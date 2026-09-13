package packstore

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"syscall"

	"b70ctl/internal/modelpack"
)

type InstalledPack struct {
	ID      string
	Name    string
	Version string
	Source  string
}

type provenance struct {
	Source string `json:"source"`
}

const (
	SourceLocal  = "local"
	SourceRemote = "remote"
)

// ImportTempPrefix is the throwaway temporary-directory prefix Import
// creates beneath the pack store root. Startup cleanup removes crash
// leftovers with this prefix.
const ImportTempPrefix = ".import-"

func Import(sourcePath, storeRoot, source string) (*InstalledPack, error) {
	if !validSource(source) {
		return nil, errors.New("pack source must be local or remote")
	}
	manifest, err := modelpack.Load(sourcePath)
	if err != nil {
		return nil, fmt.Errorf("load model pack: %w", err)
	}
	if !safePathComponent(manifest.Version) {
		return nil, errors.New("pack version cannot be used as a directory name")
	}

	destination := filepath.Join(storeRoot, manifest.ID, manifest.Version)
	if err := claimDestination(destination, manifest.ID, manifest.Version); err != nil {
		return nil, err
	}
	if err := os.MkdirAll(storeRoot, 0o755); err != nil {
		return nil, fmt.Errorf("create pack store: %w", err)
	}

	temporary, err := os.MkdirTemp(storeRoot, ImportTempPrefix)
	if err != nil {
		return nil, fmt.Errorf("create temporary pack directory: %w", err)
	}
	defer os.RemoveAll(temporary)

	files := map[string]struct{}{
		"README.md": {},
		"pack.json": {},
	}
	for relative := range files {
		if err := copyFile(filepath.Join(sourcePath, relative), filepath.Join(temporary, relative)); err != nil {
			return nil, fmt.Errorf("copy %q: %w", relative, err)
		}
	}
	if err := writeProvenance(filepath.Join(temporary, "source.json"), source); err != nil {
		return nil, err
	}

	parent := filepath.Dir(destination)
	if err := os.MkdirAll(parent, 0o755); err != nil {
		return nil, fmt.Errorf("create pack directory: %w", err)
	}
	if err := claimDestination(destination, manifest.ID, manifest.Version); err != nil {
		return nil, err
	}
	if err := os.Rename(temporary, destination); err != nil {
		return nil, fmt.Errorf("install pack: %w", err)
	}

	return &InstalledPack{
		ID:      manifest.ID,
		Name:    manifest.Name,
		Version: manifest.Version,
		Source:  source,
	}, nil
}

func List(storeRoot string) ([]InstalledPack, error) {
	idEntries, err := os.ReadDir(storeRoot)
	if os.IsNotExist(err) {
		return []InstalledPack{}, nil
	}
	if err != nil {
		return nil, fmt.Errorf("read pack store: %w", err)
	}

	var installed []InstalledPack
	for _, idEntry := range idEntries {
		if !idEntry.IsDir() {
			continue
		}
		versions, err := os.ReadDir(filepath.Join(storeRoot, idEntry.Name()))
		if err != nil {
			continue
		}
		for _, versionEntry := range versions {
			if !versionEntry.IsDir() {
				continue
			}
			path := filepath.Join(storeRoot, idEntry.Name(), versionEntry.Name())
			manifest, err := modelpack.Load(path)
			if err != nil || manifest.ID != idEntry.Name() || manifest.Version != versionEntry.Name() {
				continue
			}
			source, err := loadProvenance(filepath.Join(path, "source.json"))
			if err != nil {
				continue
			}
			installed = append(installed, InstalledPack{
				ID:      manifest.ID,
				Name:    manifest.Name,
				Version: manifest.Version,
				Source:  source,
			})
		}
	}

	sort.Slice(installed, func(i, j int) bool {
		if installed[i].ID == installed[j].ID {
			return installed[i].Version < installed[j].Version
		}
		return installed[i].ID < installed[j].ID
	})
	return installed, nil
}

func Remove(storeRoot, packID, version string) error {
	if !safePathComponent(packID) || !safePathComponent(version) {
		return errors.New("pack identity cannot be used as a directory name")
	}

	root, err := filepath.Abs(storeRoot)
	if err != nil {
		return fmt.Errorf("resolve pack store: %w", err)
	}
	root, err = filepath.EvalSymlinks(filepath.Clean(root))
	if err != nil {
		return fmt.Errorf("resolve pack store: %w", err)
	}
	parent := filepath.Join(root, packID)
	destination := filepath.Join(parent, version)
	info, err := os.Lstat(destination)
	if os.IsNotExist(err) {
		return fmt.Errorf("pack %s %s is not installed", packID, version)
	}
	if err != nil {
		return fmt.Errorf("inspect installed pack: %w", err)
	}
	if !info.IsDir() || info.Mode()&os.ModeSymlink != 0 {
		return errors.New("installed pack path is not a directory")
	}
	parentInfo, err := os.Lstat(parent)
	if err != nil {
		return fmt.Errorf("inspect pack directory: %w", err)
	}
	if !parentInfo.IsDir() || parentInfo.Mode()&os.ModeSymlink != 0 {
		return errors.New("pack directory is not a directory")
	}

	if err := os.RemoveAll(destination); err != nil {
		return fmt.Errorf("remove installed pack: %w", err)
	}
	if err := os.Remove(parent); err != nil && !errors.Is(err, syscall.ENOTEMPTY) {
		return fmt.Errorf("remove empty pack directory: %w", err)
	}
	return nil
}

func safePathComponent(value string) bool {
	return value != "." && value != ".." && filepath.Base(value) == value
}

// claimDestination keeps Import's notion of an installed pack identical to
// List's: a destination holding a loadable, matching pack is refused, while a
// leftover directory List would skip is cleared so the import can replace it.
func claimDestination(path, id, version string) error {
	info, err := os.Lstat(path)
	if os.IsNotExist(err) {
		return nil
	}
	if err != nil {
		return fmt.Errorf("check installed pack: %w", err)
	}
	if info.IsDir() && installedPackAt(path, id, version) {
		return fmt.Errorf("pack %s %s is already installed", id, version)
	}
	if err := os.RemoveAll(path); err != nil {
		return fmt.Errorf("clear invalid installed pack: %w", err)
	}
	return nil
}

func installedPackAt(path, id, version string) bool {
	manifest, err := modelpack.Load(path)
	if err != nil || manifest.ID != id || manifest.Version != version {
		return false
	}
	_, err = loadProvenance(filepath.Join(path, "source.json"))
	return err == nil
}

func copyFile(source, destination string) error {
	input, err := os.Open(source)
	if err != nil {
		return err
	}
	defer input.Close()

	info, err := input.Stat()
	if err != nil {
		return err
	}
	if !info.Mode().IsRegular() {
		return errors.New("source is not a regular file")
	}
	if err := os.MkdirAll(filepath.Dir(destination), 0o755); err != nil {
		return err
	}
	output, err := os.OpenFile(destination, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o644)
	if err != nil {
		return err
	}
	if _, err := io.Copy(output, input); err != nil {
		output.Close()
		return err
	}
	return output.Close()
}

func writeProvenance(path, source string) error {
	file, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o644)
	if err != nil {
		return fmt.Errorf("write source.json: %w", err)
	}
	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(provenance{Source: source}); err != nil {
		file.Close()
		return fmt.Errorf("write source.json: %w", err)
	}
	if err := file.Close(); err != nil {
		return fmt.Errorf("write source.json: %w", err)
	}
	return nil
}

func loadProvenance(path string) (string, error) {
	file, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer file.Close()

	var source provenance
	decoder := json.NewDecoder(file)
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&source); err != nil {
		return "", err
	}
	var extra any
	if err := decoder.Decode(&extra); !errors.Is(err, io.EOF) {
		return "", errors.New("source.json must contain one JSON object")
	}
	if !validSource(source.Source) {
		return "", errors.New("source.json has an invalid source")
	}
	return source.Source, nil
}

func validSource(source string) bool {
	return source == SourceLocal || source == SourceRemote
}
