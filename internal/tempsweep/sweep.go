// Package tempsweep removes crash-orphaned Controller temporary
// directories from configured managed roots. It is intended to run once at
// Controller startup, before any download or import operation exists, so
// every match is a leftover from an interrupted process.
package tempsweep

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// Root pairs a Controller-managed root directory with the exact temporary
// directory prefix Controller creates directly beneath it.
type Root struct {
	Path   string
	Prefix string
}

// Sweep removes leftover temporary directories matching each root's exact
// prefix and returns the removed paths. It only ever deletes real
// directories that are direct children of a managed root; symbolic links,
// non-directories, the roots themselves, and resumable staging beneath
// modelstore.StagingDirName are never touched. Sweep is best-effort per
// root: an unreadable root is reported in the returned error while the
// remaining roots are still swept.
func Sweep(roots ...Root) ([]string, error) {
	var removed []string
	var failures []error
	for _, target := range roots {
		names, err := sweepRoot(target.Path, target.Prefix)
		removed = append(removed, names...)
		if err != nil {
			failures = append(failures, err)
		}
	}
	return removed, errors.Join(failures...)
}

func sweepRoot(root, prefix string) ([]string, error) {
	if prefix == "" || prefix == "." || prefix == ".." || strings.ContainsAny(prefix, `/\`) {
		return nil, fmt.Errorf("invalid temporary prefix %q", prefix)
	}
	root, err := filepath.Abs(root)
	if err != nil {
		return nil, fmt.Errorf("resolve managed root: %w", err)
	}
	info, err := os.Lstat(root)
	if os.IsNotExist(err) {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("read managed root: %w", err)
	}
	if !info.IsDir() || info.Mode()&os.ModeSymlink != 0 {
		return nil, nil
	}
	entries, err := os.ReadDir(root)
	if err != nil {
		return nil, fmt.Errorf("read managed root: %w", err)
	}
	var removed []string
	for _, entry := range entries {
		if !strings.HasPrefix(entry.Name(), prefix) {
			continue
		}
		entryInfo, err := entry.Info()
		if err != nil || !entryInfo.IsDir() || entryInfo.Mode()&os.ModeSymlink != 0 {
			continue
		}
		path := filepath.Join(root, entry.Name())
		if err := os.RemoveAll(path); err != nil {
			return removed, fmt.Errorf("remove leftover temporary directory: %w", err)
		}
		removed = append(removed, path)
	}
	return removed, nil
}
