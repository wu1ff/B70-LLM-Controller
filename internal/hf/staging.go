package hf

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"syscall"

	"b70ctl/internal/modelstore"
)

const (
	stagingMarkerName    = ".b70-staging.json"
	stagingMarkerSchema  = 1
	stagingMarkerPartial = stagingMarkerName + ".tmp"
	partSuffix           = ".part"
)

type stagingMarker struct {
	Schema   int    `json:"schema"`
	Repo     string `json:"repo"`
	Revision string `json:"revision"`
}

// stagingArea owns the persistent, resumable staging directory for one
// exact repo+revision while a download attempt runs. The flock guarantees a
// single writer across Controller processes and restarts; the marker is the
// provenance contract validated before any staged content is reused.
type stagingArea struct {
	dir      string
	lockPath string
	lock     *os.File
}

func acquireStagingArea(root string, repository Repository) (*stagingArea, error) {
	dir, err := modelstore.StagingPath(root, repository.Repo, repository.Revision)
	if err != nil {
		return nil, err
	}
	parent := filepath.Dir(dir)
	if info, err := os.Lstat(parent); err == nil {
		if !info.IsDir() || info.Mode()&os.ModeSymlink != 0 {
			return nil, errors.New("model staging root is not a safe directory")
		}
	} else if !os.IsNotExist(err) {
		return nil, fmt.Errorf("check model staging root: %w", err)
	}
	if err := os.MkdirAll(parent, 0o755); err != nil {
		return nil, fmt.Errorf("create model staging root: %w", err)
	}

	// Lock before touching staging content so exactly one attempt at a time
	// creates directories, writes the marker, or mutates staged files.
	lockPath := dir + ".lock"
	lock, err := os.OpenFile(lockPath, os.O_RDWR|os.O_CREATE, 0o644)
	if err != nil {
		return nil, fmt.Errorf("open model staging lock: %w", err)
	}
	if err := syscall.Flock(int(lock.Fd()), syscall.LOCK_EX|syscall.LOCK_NB); err != nil {
		lock.Close()
		if errors.Is(err, syscall.EWOULDBLOCK) {
			return nil, fmt.Errorf("%w: another download of %s is already staging this revision", ErrDownloadInProgress, repository.Repo)
		}
		return nil, fmt.Errorf("lock model staging: %w", err)
	}
	area := &stagingArea{dir: dir, lockPath: lockPath, lock: lock}

	if info, err := os.Lstat(dir); err == nil {
		if !info.IsDir() || info.Mode()&os.ModeSymlink != 0 {
			area.release()
			return nil, errors.New("model staging directory is not a safe directory")
		}
		if err := area.establishMarker(repository); err != nil {
			area.release()
			return nil, err
		}
	} else if os.IsNotExist(err) {
		if err := os.Mkdir(dir, 0o755); err != nil {
			area.release()
			return nil, fmt.Errorf("create model staging directory: %w", err)
		}
		if err := writeStagingMarker(dir, repository); err != nil {
			area.release()
			return nil, err
		}
	} else {
		area.release()
		return nil, fmt.Errorf("check model staging directory: %w", err)
	}
	return area, nil
}

// release drops the staging lock, leaving the staging directory and lock
// file in place for the next attempt.
func (area *stagingArea) release() {
	if area.lock == nil {
		return
	}
	_ = syscall.Flock(int(area.lock.Fd()), syscall.LOCK_UN)
	_ = area.lock.Close()
	area.lock = nil
}

// consumed releases the staging area after its directory was promoted to
// the final destination, removing the stale lock file and the staging root
// itself when nothing else is staging.
func (area *stagingArea) consumed() {
	if area.lock != nil {
		_ = os.Remove(area.lockPath)
	}
	area.release()
	_ = os.Remove(filepath.Dir(area.dir))
}

// establishMarker validates an existing staging directory's marker, or
// writes a fresh marker when the directory is an empty crash window.
func (area *stagingArea) establishMarker(repository Repository) error {
	found, err := validateStagingMarker(area.dir, repository)
	if err != nil || found {
		return err
	}
	fresh, err := stagingIsFresh(area.dir)
	if err != nil {
		return err
	}
	if !fresh {
		return errors.New("model staging directory has no valid staging marker")
	}
	return writeStagingMarker(area.dir, repository)
}

// validateStagingMarker reports whether dir holds a staging marker, and
// fails on malformed markers, foreign schemas, or provenance that does not
// match the exact repo+revision being requested. Mismatched staging is
// never reused and never deleted automatically.
func validateStagingMarker(dir string, repository Repository) (bool, error) {
	path := filepath.Join(dir, stagingMarkerName)
	info, err := os.Lstat(path)
	if os.IsNotExist(err) {
		return false, nil
	}
	if err != nil {
		return false, fmt.Errorf("check staging marker: %w", err)
	}
	if !info.Mode().IsRegular() {
		return false, errors.New("staging marker is not a regular file")
	}
	file, err := os.Open(path)
	if err != nil {
		return false, fmt.Errorf("read staging marker: %w", err)
	}
	defer file.Close()

	var marker stagingMarker
	decoder := json.NewDecoder(file)
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&marker); err != nil {
		return false, fmt.Errorf("staging marker is invalid: %w", err)
	}
	var extra any
	if err := decoder.Decode(&extra); !errors.Is(err, io.EOF) {
		return false, errors.New("staging marker is invalid: must contain one JSON object")
	}
	if marker.Schema != stagingMarkerSchema {
		return false, fmt.Errorf("staging marker schema %d is not supported", marker.Schema)
	}
	if marker.Repo != repository.Repo || marker.Revision != repository.Revision {
		return false, fmt.Errorf("staging directory belongs to %s at %s, not the requested model", marker.Repo, marker.Revision)
	}
	return true, nil
}

// stagingIsFresh reports whether dir holds nothing but an interrupted
// marker write, i.e. only the crash window between directory creation and
// the marker's atomic rename.
func stagingIsFresh(dir string) (bool, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return false, fmt.Errorf("read model staging directory: %w", err)
	}
	for _, entry := range entries {
		info, err := entry.Info()
		if err != nil {
			return false, fmt.Errorf("read model staging entry: %w", err)
		}
		if entry.Name() == stagingMarkerPartial && info.Mode().IsRegular() {
			continue
		}
		return false, nil
	}
	return true, nil
}

func writeStagingMarker(dir string, repository Repository) error {
	temporary := filepath.Join(dir, stagingMarkerPartial)
	if err := clearStalePartFile(temporary); err != nil {
		return err
	}
	file, err := os.OpenFile(temporary, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o644)
	if err != nil {
		return fmt.Errorf("write staging marker: %w", err)
	}
	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	err = encoder.Encode(stagingMarker{Schema: stagingMarkerSchema, Repo: repository.Repo, Revision: repository.Revision})
	closeErr := file.Close()
	if err != nil {
		return fmt.Errorf("write staging marker: %w", err)
	}
	if closeErr != nil {
		return fmt.Errorf("write staging marker: %w", closeErr)
	}
	if err := os.Rename(temporary, filepath.Join(dir, stagingMarkerName)); err != nil {
		return fmt.Errorf("write staging marker: %w", err)
	}
	return nil
}

// stagingFileTarget maps a repository file path to its staged location and
// refuses anything that would escape the staging directory.
func stagingFileTarget(dir, remotePath string) (string, error) {
	path := filepath.Join(dir, filepath.FromSlash(remotePath))
	relative, err := filepath.Rel(dir, path)
	if err != nil || relative == "." || relative == ".." || strings.HasPrefix(relative, ".."+string(filepath.Separator)) {
		return "", errors.New("staged file escapes the staging directory")
	}
	return path, nil
}

// ensureStagingParents creates or verifies every intermediate directory for
// a staged file, refusing symbolic links and non-directories so a staged
// tree can never redirect writes outside the staging area.
func ensureStagingParents(dir, target string) error {
	relative, err := filepath.Rel(dir, target)
	if err != nil {
		return fmt.Errorf("check staging directory: %w", err)
	}
	components := strings.Split(relative, string(filepath.Separator))
	current := dir
	for index := 0; index < len(components)-1; index++ {
		current = filepath.Join(current, components[index])
		info, err := os.Lstat(current)
		if os.IsNotExist(err) {
			if err := os.Mkdir(current, 0o755); err != nil && !os.IsExist(err) {
				return fmt.Errorf("create staging directory: %w", err)
			}
			continue
		}
		if err != nil {
			return fmt.Errorf("check staging directory: %w", err)
		}
		if !info.IsDir() || info.Mode()&os.ModeSymlink != 0 {
			return errors.New("staging path contains an unsafe directory entry")
		}
	}
	return nil
}

// clearStalePartFile removes an interrupted per-file download artifact.
// Only a regular file is ever removed; anything else is refused rather
// than followed.
func clearStalePartFile(path string) error {
	info, err := os.Lstat(path)
	if os.IsNotExist(err) {
		return nil
	}
	if err != nil {
		return fmt.Errorf("check partial download: %w", err)
	}
	if !info.Mode().IsRegular() {
		return errors.New("partial download artifact is not a regular file")
	}
	if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("remove partial download: %w", err)
	}
	return nil
}
