// Package packupdate implements the model-pack update flow: recognizing a
// newer catalog version of an installed pack, replacing the installed version
// in a safe order, and retiring superseded versions through the shared
// uninstall engine so artifact ownership rules stay in one place.
package packupdate

import (
	"errors"
	"fmt"
	"sort"
	"strconv"
	"strings"

	"b70ctl/internal/catalog"
	"b70ctl/internal/install"
	"b70ctl/internal/packstore"
	"b70ctl/internal/uninstall"
)

// Status is the relation between one catalog entry and the installed
// versions of the same pack ID.
type Status string

const (
	// StatusNotInstalled: no version of the pack is installed.
	StatusNotInstalled Status = "not-installed"
	// StatusInstalled: the exact catalog version is installed and no older
	// version of the pack remains.
	StatusInstalled Status = "installed"
	// StatusUpdateAvailable: the pack is installed at an older version.
	StatusUpdateAvailable Status = "update-available"
	// StatusFinishUpdate: the exact catalog version is installed but older
	// versions of the pack still are too, so the update needs finishing.
	StatusFinishUpdate Status = "finish-update"
	// StatusOlderCatalog: the catalog entry is older than the installed
	// version; a downgrade is never offered.
	StatusOlderCatalog Status = "older-catalog"
	// StatusIncomparable: the installed and catalog versions cannot be
	// ordered, so exact-version behavior applies.
	StatusIncomparable Status = "incomparable"
)

// State is the classification of one catalog entry against the installed
// packs.
type State struct {
	Status Status
	// Current is the newest installed version of the same pack ID, or ""
	// when none is installed.
	Current string
	// OldVersions are the installed versions older than the catalog entry.
	OldVersions []string
	// Reason explains why versions could not be compared
	// (StatusIncomparable only).
	Reason string
}

// Classify determines how a catalog entry relates to the installed versions
// of the same pack ID. When the versions cannot be compared safely it falls
// back to exact-version behavior and reports why.
func Classify(installed []packstore.InstalledPack, entry catalog.Entry) State {
	var versions []string
	for _, pack := range installed {
		if pack.ID == entry.ID {
			versions = append(versions, pack.Version)
		}
	}
	if len(versions) == 0 {
		return State{Status: StatusNotInstalled}
	}
	exact := false
	for _, version := range versions {
		if version == entry.Version {
			exact = true
			break
		}
	}
	newest, err := newestVersion(versions)
	newestVersusEntry := 0
	if err == nil {
		newestVersusEntry, err = compareVersions(newest, entry.Version)
	}
	if err != nil {
		if exact {
			return State{Status: StatusInstalled, Current: entry.Version}
		}
		return State{
			Status: StatusIncomparable,
			Reason: fmt.Sprintf("Installed pack version %q cannot be compared with catalog version %q, so update detection is unavailable.", versions[0], entry.Version),
		}
	}
	state := State{Current: newest}
	for _, version := range versions {
		if comparison, err := compareVersions(version, entry.Version); err == nil && comparison < 0 {
			state.OldVersions = append(state.OldVersions, version)
		}
	}
	switch {
	case exact && len(state.OldVersions) > 0:
		state.Status = StatusFinishUpdate
	case exact:
		state.Status = StatusInstalled
	case newestVersusEntry < 0:
		state.Status = StatusUpdateAvailable
	default:
		state.Status = StatusOlderCatalog
	}
	return state
}

// PreparationSucceeded reports whether an install preparation result is
// complete enough to retire the previous pack version: no model or runtime
// artifact failed. Skipped gated models do not block an update; they keep the
// normal retry-from-Models path.
func PreparationSucceeded(result install.Result) bool {
	if result.HasRuntimeFailure() {
		return false
	}
	for _, item := range result.Items {
		if item.Outcome == install.Failed {
			return false
		}
	}
	return true
}

// Outcome is how far one update attempt got.
type Outcome uint8

const (
	// OutcomeUpdated: the new version is installed and every older version
	// of the pack is retired.
	OutcomeUpdated Outcome = iota
	// OutcomePrepareFailed: preparing the new version failed; the newly
	// imported version was rolled back and the previous version stays.
	OutcomePrepareFailed
	// OutcomePrepareIncomplete: the new version was imported but artifacts
	// failed; the new version was rolled back and the previous version
	// stays.
	OutcomePrepareIncomplete
	// OutcomeRetireIncomplete: the new version is installed but at least one
	// older version could not be retired.
	OutcomeRetireIncomplete
)

// Result is the outcome of one update attempt.
type Result struct {
	Outcome  Outcome
	Err      error
	Prepared install.Result
	Retired  []uninstall.Result
}

// Apply runs one pack update in the safe order: prepare — which imports the
// new version into the pack store and then verifies its artifacts — must
// complete successfully before any older version is retired. When prepare
// fails or reports failed artifacts, the newly imported version is removed
// again so the previous version stays active and the update can be retried;
// no older version is retired in that case. A version of the pack that
// already existed before the attempt is never rolled back.
func Apply(storeRoot, dataRoot, modelRoot, packID, newVersion string, prepare func() (install.Result, error)) Result {
	preexisting := versionInstalled(storeRoot, packID, newVersion)
	prepared, err := prepare()
	if err != nil {
		result := Result{Outcome: OutcomePrepareFailed, Err: err, Prepared: prepared}
		if !preexisting {
			wrapRollback(storeRoot, packID, newVersion, &result)
		}
		return result
	}
	if !PreparationSucceeded(prepared) {
		result := Result{Outcome: OutcomePrepareIncomplete, Prepared: prepared}
		if !preexisting {
			wrapRollback(storeRoot, packID, newVersion, &result)
		}
		return result
	}
	retired, err := RetireOlder(storeRoot, dataRoot, modelRoot, packID, newVersion)
	if err != nil {
		return Result{Outcome: OutcomeRetireIncomplete, Err: err, Prepared: prepared, Retired: retired}
	}
	return Result{Outcome: OutcomeUpdated, Prepared: prepared, Retired: retired}
}

// RetireOlder removes every installed version of packID strictly older than
// keepVersion through the shared uninstall engine, so model and runtime
// retention follow the existing ownership rules: artifacts referenced by any
// remaining installed pack (including keepVersion) stay, and artifacts
// b70ctl cannot prove it owns are never deleted. Newer versions and versions
// that cannot be ordered with keepVersion are never touched.
func RetireOlder(storeRoot, dataRoot, modelRoot, packID, keepVersion string) ([]uninstall.Result, error) {
	installed, err := packstore.List(storeRoot)
	if err != nil {
		return nil, err
	}
	var older []string
	for _, pack := range installed {
		if pack.ID != packID || pack.Version == keepVersion {
			continue
		}
		comparison, err := compareVersions(pack.Version, keepVersion)
		if err != nil {
			return nil, fmt.Errorf("installed version %s of pack %s cannot be compared with version %s and was left installed: %w", pack.Version, packID, keepVersion, err)
		}
		if comparison < 0 {
			older = append(older, pack.Version)
		}
	}
	// All entries compared cleanly with keepVersion, so they are dotted
	// numeric versions and order among themselves oldest first; the string
	// fallback is unreachable belt-and-braces.
	sort.Slice(older, func(i, j int) bool {
		comparison, err := compareVersions(older[i], older[j])
		if err != nil {
			return older[i] < older[j]
		}
		return comparison < 0
	})
	var results []uninstall.Result
	for _, version := range older {
		result, err := uninstall.Run(storeRoot, dataRoot, modelRoot, packID, version, uninstall.DeleteEverything)
		if err != nil {
			return results, fmt.Errorf("remove old pack version %s %s: %w", packID, version, err)
		}
		results = append(results, result)
	}
	return results, nil
}

// wrapRollback removes a version the failed attempt imported and records the
// removal failure alongside the original error.
func wrapRollback(storeRoot, packID, version string, result *Result) {
	if err := rollbackNewVersion(storeRoot, packID, version); err != nil {
		result.Err = errors.Join(result.Err, err)
	}
}

// rollbackNewVersion removes a partially updated pack version that did not
// exist before the update attempt. A version that was never imported is not
// an error.
func rollbackNewVersion(storeRoot, packID, version string) error {
	if !versionInstalled(storeRoot, packID, version) {
		return nil
	}
	if err := packstore.Remove(storeRoot, packID, version); err != nil {
		return fmt.Errorf("remove incomplete new pack version %s %s: %w", packID, version, err)
	}
	return nil
}

func versionInstalled(storeRoot, packID, version string) bool {
	installed, err := packstore.List(storeRoot)
	if err != nil {
		return false
	}
	for _, pack := range installed {
		if pack.ID == packID && pack.Version == version {
			return true
		}
	}
	return false
}

// compareVersions orders two dotted numeric versions (for example "1.0.0"
// and "1.0.1") component by component, treating missing components as zero.
// Any empty or non-numeric component makes the comparison fail; callers then
// fall back to exact-version behavior instead of guessing.
func compareVersions(a, b string) (int, error) {
	if a == "" || b == "" {
		return 0, errors.New("version is empty")
	}
	left := strings.Split(a, ".")
	right := strings.Split(b, ".")
	for index := 0; index < len(left) || index < len(right); index++ {
		var leftValue, rightValue int
		var err error
		if index < len(left) {
			leftValue, err = versionComponent(left[index])
			if err != nil {
				return 0, fmt.Errorf("version %q: %w", a, err)
			}
		}
		if index < len(right) {
			rightValue, err = versionComponent(right[index])
			if err != nil {
				return 0, fmt.Errorf("version %q: %w", b, err)
			}
		}
		if leftValue != rightValue {
			if leftValue < rightValue {
				return -1, nil
			}
			return 1, nil
		}
	}
	return 0, nil
}

func versionComponent(value string) (int, error) {
	if value == "" {
		return 0, errors.New("version component is empty")
	}
	for _, character := range value {
		if character < '0' || character > '9' {
			return 0, errors.New("version component is not numeric")
		}
	}
	parsed, err := strconv.Atoi(value)
	if err != nil {
		return 0, errors.New("version component is out of range")
	}
	return parsed, nil
}

func newestVersion(versions []string) (string, error) {
	newest := versions[0]
	for _, version := range versions[1:] {
		comparison, err := compareVersions(version, newest)
		if err != nil {
			return "", err
		}
		if comparison > 0 {
			newest = version
		}
	}
	return newest, nil
}
