package modelstore

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

const markerName = ".b70-model.json"

// StagingDirName is the directory beneath the model root that holds
// Controller-owned resumable download staging. Its contents are never final
// models and are never discovered by Scan.
const StagingDirName = ".b70ctl-staging"

type Artifact struct {
	Repo     string
	Revision string
	Path     string
}

// Managed reports whether the artifact is a Controller-managed local model:
// its directory holds a valid model marker matching the artifact identity
// exactly. It mirrors the acceptance test Remove applies before deleting, so
// only managed artifacts qualify for Controller-side lifecycle decisions such
// as orphan classification; external Hugging Face cache entries and
// unrecognized directories never do.
func (artifact Artifact) Managed() bool {
	identity, err := loadMarker(filepath.Join(artifact.Path, markerName))
	return err == nil && identity.Repo == artifact.Repo && identity.Revision == artifact.Revision
}

type RemovalStatus string

const (
	RemovalMissing         RemovalStatus = "missing"
	RemovalRemoved         RemovalStatus = "removed"
	RemovalExternalHFCache RemovalStatus = "external-hf-cache"
	RemovalMultipleCopies  RemovalStatus = "multiple-copies"
	RemovalRecognizedModel RemovalStatus = "recognized-model"
)

type marker struct {
	Repo     string `json:"repo"`
	Revision string `json:"revision"`
}

func Scan(root string) ([]Artifact, error) {
	root, err := filepath.Abs(root)
	if err != nil {
		return nil, fmt.Errorf("resolve model root: %w", err)
	}
	root = filepath.Clean(root)

	entries, err := os.ReadDir(root)
	if os.IsNotExist(err) {
		return []Artifact{}, nil
	}
	if err != nil {
		return nil, fmt.Errorf("read model root: %w", err)
	}

	artifacts := []Artifact{}
	scanEntries := make([]os.DirEntry, 0, len(entries))
	for _, entry := range entries {
		if entry.Name() == StagingDirName {
			continue
		}
		scanEntries = append(scanEntries, entry)
	}
	artifacts = append(artifacts, scanMarkers(root, scanEntries)...)
	artifacts = append(artifacts, scanHFLocalDirectories(root, scanEntries)...)
	artifacts = append(artifacts, scanHFCache(root, scanEntries)...)

	for _, entry := range scanEntries {
		if entry.Name() != "hub" || !entry.IsDir() {
			continue
		}
		hub := filepath.Join(root, entry.Name())
		hubEntries, err := os.ReadDir(hub)
		if err == nil {
			artifacts = append(artifacts, scanHFCache(hub, hubEntries)...)
		}
	}

	sort.Slice(artifacts, func(i, j int) bool {
		if artifacts[i].Repo != artifacts[j].Repo {
			return artifacts[i].Repo < artifacts[j].Repo
		}
		if artifacts[i].Revision != artifacts[j].Revision {
			return artifacts[i].Revision < artifacts[j].Revision
		}
		return artifacts[i].Path < artifacts[j].Path
	})
	deduplicated := artifacts[:0]
	for _, artifact := range artifacts {
		if len(deduplicated) > 0 && deduplicated[len(deduplicated)-1] == artifact {
			continue
		}
		deduplicated = append(deduplicated, artifact)
	}
	return deduplicated, nil
}

func Find(artifacts []Artifact, repo, revision string) (Artifact, bool) {
	var found Artifact
	matched := false
	for _, artifact := range artifacts {
		if artifact.Repo != repo || artifact.Revision != revision {
			continue
		}
		if !matched || artifact.Path < found.Path {
			found = artifact
			matched = true
		}
	}
	return found, matched
}

func Remove(root, repo, revision string) (RemovalStatus, error) {
	artifacts, err := Scan(root)
	if err != nil {
		return "", err
	}
	var matches []Artifact
	for _, artifact := range artifacts {
		if artifact.Repo == repo && artifact.Revision == revision {
			matches = append(matches, artifact)
		}
	}
	if len(matches) == 0 {
		return RemovalMissing, nil
	}
	if len(matches) > 1 {
		return RemovalMultipleCopies, nil
	}

	identity, err := loadMarker(filepath.Join(matches[0].Path, markerName))
	if err != nil || identity.Repo != repo || identity.Revision != revision {
		return RemovalExternalHFCache, nil
	}
	resolvedRoot, resolvedPath, err := safeRemovalPaths(root, matches[0].Path)
	if err != nil {
		return "", err
	}
	if resolvedPath == resolvedRoot {
		return "", errors.New("model path is the configured model root")
	}
	identity, err = loadMarker(filepath.Join(resolvedPath, markerName))
	if err != nil || identity.Repo != repo || identity.Revision != revision {
		return "", errors.New("model marker no longer matches the requested artifact")
	}
	if err := os.RemoveAll(resolvedPath); err != nil {
		return "", fmt.Errorf("remove model artifact: %w", err)
	}
	return RemovalRemoved, nil
}

// DestinationName returns the directory name Controller uses to store a
// repository's model files.
func DestinationName(repo string) string {
	return strings.ReplaceAll(repo, "/", "__")
}

// DestinationExists reports whether anything occupies the canonical
// destination directory Controller would use for repo beneath root.
func DestinationExists(root, repo string) (bool, error) {
	destination, err := destinationPath(root, repo)
	if err != nil {
		return false, err
	}
	if _, err := os.Lstat(destination); err == nil {
		return true, nil
	} else if !os.IsNotExist(err) {
		return false, fmt.Errorf("check model destination: %w", err)
	}
	return false, nil
}

// RemoveDestination removes a directory occupying the canonical destination
// Controller would use for repo when that directory is not a model Scan
// recognizes. It is the remedy for an invalid destination that blocks a
// fresh exact-revision download; recognized model directories, symbolic
// links, and anything outside the model root are refused.
func RemoveDestination(root, repo string) (RemovalStatus, error) {
	artifacts, err := Scan(root)
	if err != nil {
		return "", err
	}
	destination, err := destinationPath(root, repo)
	if err != nil {
		return "", err
	}
	info, err := os.Lstat(destination)
	if os.IsNotExist(err) {
		return RemovalMissing, nil
	}
	if err != nil {
		return "", fmt.Errorf("check model destination: %w", err)
	}
	if err := checkRemovableDestination(info); err != nil {
		return "", err
	}
	for _, artifact := range artifacts {
		if artifact.Path == destination {
			return RemovalRecognizedModel, nil
		}
	}
	resolvedRoot, resolvedPath, err := safeRemovalPaths(root, destination)
	if err != nil {
		return "", err
	}
	if resolvedPath == resolvedRoot {
		return "", errors.New("model destination is the configured model root")
	}
	info, err = os.Lstat(resolvedPath)
	if os.IsNotExist(err) {
		return RemovalMissing, nil
	}
	if err != nil {
		return "", fmt.Errorf("recheck model destination: %w", err)
	}
	if err := checkRemovableDestination(info); err != nil {
		return "", fmt.Errorf("model destination changed before removal: %w", err)
	}
	if err := os.RemoveAll(resolvedPath); err != nil {
		return "", fmt.Errorf("remove model destination: %w", err)
	}
	return RemovalRemoved, nil
}

func checkRemovableDestination(info os.FileInfo) error {
	if info.Mode()&os.ModeSymlink != 0 {
		return errors.New("model destination is a symbolic link")
	}
	if !info.IsDir() {
		return errors.New("model destination is not a directory")
	}
	return nil
}

func destinationPath(root, repo string) (string, error) {
	root, err := filepath.Abs(root)
	if err != nil {
		return "", fmt.Errorf("resolve model root: %w", err)
	}
	name := DestinationName(repo)
	if name == "" || name == "." || name == ".." || strings.ContainsRune(name, filepath.Separator) {
		return "", errors.New("repository cannot be used as a destination name")
	}
	return filepath.Clean(filepath.Join(root, name)), nil
}

// StagingPath returns the persistent staging directory Controller uses to
// prepare repo at the exact resolved revision beneath root. The key is
// derived from the destination name plus the resolved commit SHA, so the
// same exact model always finds the same staging area and different
// repos or revisions never collide.
func StagingPath(root, repo, revision string) (string, error) {
	root, err := filepath.Abs(root)
	if err != nil {
		return "", fmt.Errorf("resolve model root: %w", err)
	}
	if revision == "" || revision == "." || revision == ".." || strings.ContainsAny(revision, `/\`) {
		return "", errors.New("revision cannot be used as a staging name")
	}
	name := DestinationName(repo) + "-" + revision
	if name == "" || name == "." || name == ".." || strings.ContainsRune(name, filepath.Separator) {
		return "", errors.New("repository cannot be used as a staging name")
	}
	return filepath.Clean(filepath.Join(root, StagingDirName, name)), nil
}

func safeRemovalPaths(root, path string) (string, string, error) {
	root, err := filepath.Abs(root)
	if err != nil {
		return "", "", fmt.Errorf("resolve model root: %w", err)
	}
	root, err = filepath.EvalSymlinks(filepath.Clean(root))
	if err != nil {
		return "", "", fmt.Errorf("resolve model root: %w", err)
	}
	path, err = filepath.Abs(path)
	if err != nil {
		return "", "", fmt.Errorf("resolve model path: %w", err)
	}
	path, err = filepath.EvalSymlinks(filepath.Clean(path))
	if err != nil {
		return "", "", fmt.Errorf("resolve model path: %w", err)
	}
	relative, err := filepath.Rel(root, path)
	if err != nil {
		return "", "", fmt.Errorf("compare model path to model root: %w", err)
	}
	if relative == "." || relative == ".." || strings.HasPrefix(relative, ".."+string(filepath.Separator)) {
		return "", "", errors.New("model path is not beneath the configured model root")
	}
	return root, path, nil
}

func scanMarkers(root string, entries []os.DirEntry) []Artifact {
	var artifacts []Artifact
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		path := filepath.Join(root, entry.Name())
		identity, err := loadMarker(filepath.Join(path, markerName))
		if err != nil {
			continue
		}
		artifacts = append(artifacts, Artifact{
			Repo:     identity.Repo,
			Revision: identity.Revision,
			Path:     path,
		})
	}
	return artifacts
}

func scanHFCache(root string, entries []os.DirEntry) []Artifact {
	var artifacts []Artifact
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		repo, ok := cacheRepo(entry.Name())
		if !ok {
			continue
		}
		snapshots := filepath.Join(root, entry.Name(), "snapshots")
		revisions, err := os.ReadDir(snapshots)
		if err != nil {
			continue
		}
		for _, revision := range revisions {
			if !revision.IsDir() || revision.Name() == "" {
				continue
			}
			artifacts = append(artifacts, Artifact{
				Repo:     repo,
				Revision: revision.Name(),
				Path:     filepath.Join(snapshots, revision.Name()),
			})
		}
	}
	return artifacts
}

func scanHFLocalDirectories(root string, entries []os.DirEntry) []Artifact {
	var artifacts []Artifact
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		parts := strings.Split(entry.Name(), "__")
		if len(parts) != 2 || parts[0] == "" || parts[1] == "" {
			continue
		}
		trees := filepath.Join(root, entry.Name(), ".cache", "huggingface", "trees")
		revisions, err := os.ReadDir(trees)
		if err != nil {
			continue
		}
		for _, revision := range revisions {
			if revision.IsDir() || filepath.Ext(revision.Name()) != ".json" {
				continue
			}
			name := strings.TrimSuffix(revision.Name(), ".json")
			if name == "" {
				continue
			}
			artifacts = append(artifacts, Artifact{Repo: parts[0] + "/" + parts[1], Revision: name, Path: filepath.Join(root, entry.Name())})
		}
	}
	return artifacts
}

func cacheRepo(name string) (string, bool) {
	parts := strings.Split(strings.TrimPrefix(name, "models--"), "--")
	if !strings.HasPrefix(name, "models--") || len(parts) != 2 || parts[0] == "" || parts[1] == "" {
		return "", false
	}
	return parts[0] + "/" + parts[1], true
}

func loadMarker(path string) (marker, error) {
	file, err := os.Open(path)
	if err != nil {
		return marker{}, err
	}
	defer file.Close()

	var identity marker
	decoder := json.NewDecoder(file)
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&identity); err != nil {
		return marker{}, err
	}
	var extra any
	if err := decoder.Decode(&extra); !errors.Is(err, io.EOF) {
		return marker{}, errors.New("marker must contain one JSON object")
	}
	if strings.TrimSpace(identity.Repo) == "" || strings.TrimSpace(identity.Revision) == "" {
		return marker{}, errors.New("marker identity is incomplete")
	}
	return identity, nil
}
