package tui

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"b70ctl/internal/catalog"
	"b70ctl/internal/config"
	"b70ctl/internal/install"
	"b70ctl/internal/modelpack"
	"b70ctl/internal/modelstore"
	"b70ctl/internal/packstore"
	"b70ctl/internal/packupdate"
	"b70ctl/internal/uninstall"
)

var updateTestDigest = "sha256:" + strings.Repeat("c", 64)

func updateTestModel(id, name, repo, revision string) modelpack.Model {
	return modelpack.Model{
		ID: id, Name: name, Kind: "target", Repo: repo, Revision: revision,
		Files:     []modelpack.ModelFile{{Path: "model.bin"}},
		MountPath: "/models/target",
		Launch:    modelpack.LaunchBlock{DockerArgs: []string{}, Environment: map[string]string{}, CommandArgs: []string{}},
	}
}

func updateTestManifest(version, repo, revision string) modelpack.Manifest {
	model := updateTestModel("target", "Test Target", repo, revision)
	empty := modelpack.LaunchBlock{DockerArgs: []string{}, Environment: map[string]string{}, CommandArgs: []string{}}
	return modelpack.Manifest{
		SchemaVersion: 1, ID: "update-pack", Name: "Update Pack", Version: version,
		Models: []modelpack.Model{model},
		Runtimes: []modelpack.Runtime{{
			ID: "runtime", Image: "example/runtime:1", Digest: updateTestDigest,
			ContainerPort: 8000, HealthPath: "/health",
			Launch: modelpack.RuntimeLaunch{DockerArgs: []string{}, Environment: map[string]string{}, Command: []string{"serve"}},
		}},
		Modes:    []modelpack.Mode{{ID: "base", DisplayName: "Base", Assistants: []string{}, Launch: empty}},
		Profiles: []modelpack.Profile{{ID: "profile", ModelID: "target", RuntimeID: "runtime", Cards: 1, TensorParallel: 1, Context: 4096, Mode: "base", Launch: empty}},
	}
}

// writeUpdatePackSource writes a loadable pack directory (the stand-in for a
// downloaded catalog archive) and returns its path and manifest.
func writeUpdatePackSource(t *testing.T, manifest modelpack.Manifest) (string, *modelpack.Manifest) {
	t.Helper()
	source := t.TempDir()
	writeTuiPack(t, source, manifest)
	return source, &manifest
}

func importUpdatePack(t *testing.T, storeRoot string, manifest modelpack.Manifest) {
	t.Helper()
	source, _ := writeUpdatePackSource(t, manifest)
	if _, err := packstore.Import(source, storeRoot, packstore.SourceLocal); err != nil {
		t.Fatal(err)
	}
}

func writeUpdateManagedModel(t *testing.T, root, repo, revision string) string {
	t.Helper()
	directory := modelstore.DestinationName(repo)
	writePresentModelAt(t, root, directory, repo, revision, 6)
	return filepath.Join(root, directory)
}

// updateTestApp builds an app with isolated config/data roots so update flows
// never touch the real Hugging Face token or data directory.
func updateTestApp(t *testing.T, events []byte, modelsRoot, packsRoot, dataRoot string) (*app, *os.File) {
	t.Helper()
	t.Setenv("XDG_CONFIG_HOME", filepath.Join(t.TempDir(), "config"))
	t.Setenv("XDG_DATA_HOME", filepath.Join(t.TempDir(), "data"))
	t.Setenv("HF_TOKEN", "")
	application, output := uninstallTestApp(t, events)
	application.config = config.Config{ModelDirectory: modelsRoot, DefaultAccess: config.AccessLocal, DefaultPort: 8000}
	application.paths = config.Paths{PacksRoot: packsRoot, DataRoot: dataRoot}
	return application, output
}

func updateDockerStub(t *testing.T) {
	t.Helper()
	bin := t.TempDir()
	script := `#!/bin/sh
if [ "$1" = "inspect" ]; then
  echo "Error: No such object: b70ctl-runtime" >&2
  exit 1
fi
if [ "$1" = "image" ] && [ "$2" = "inspect" ]; then
  echo '"` + updateTestDigest + `"'
  exit 0
fi
if [ "$1" = "image" ] && [ "$2" = "rm" ]; then
  echo "Untagged: $3"
  exit 0
fi
exit 2
`
	path := filepath.Join(bin, "docker")
	if err := os.WriteFile(path, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", bin+string(os.PathListSeparator)+os.Getenv("PATH"))
}

func TestCatalogEntryDetailShowsUpdateTransition(t *testing.T) {
	installed := []packstore.InstalledPack{{ID: "pack", Version: "1.0.0"}}
	detail := stripANSI(catalogEntryDetail(catalog.Entry{ID: "pack", Version: "1.0.1"}, packupdate.Classify(installed, catalog.Entry{ID: "pack", Version: "1.0.1"})))
	for _, want := range []string{"v1.0.0 → v1.0.1", "● UPDATE AVAILABLE"} {
		if !strings.Contains(detail, want) {
			t.Fatalf("update detail missing %q: %q", want, detail)
		}
	}
}

func TestCatalogEntryDetailFinishUpdateMentionsOldVersion(t *testing.T) {
	installed := []packstore.InstalledPack{{ID: "pack", Version: "1.0.0"}, {ID: "pack", Version: "1.0.1"}}
	detail := stripANSI(catalogEntryDetail(catalog.Entry{ID: "pack", Version: "1.0.1"}, packupdate.Classify(installed, catalog.Entry{ID: "pack", Version: "1.0.1"})))
	for _, want := range []string{"v1.0.1 (old 1.0.0)", "● FINISH UPDATE"} {
		if !strings.Contains(detail, want) {
			t.Fatalf("finish-update detail missing %q: %q", want, detail)
		}
	}
}

func TestCatalogEntryDetailDowngradeNotOfferedAsUpdate(t *testing.T) {
	installed := []packstore.InstalledPack{{ID: "pack", Version: "1.0.1"}}
	detail := stripANSI(catalogEntryDetail(catalog.Entry{ID: "pack", Version: "1.0.0"}, packupdate.Classify(installed, catalog.Entry{ID: "pack", Version: "1.0.0"})))
	if strings.Contains(detail, "UPDATE AVAILABLE") || !strings.Contains(detail, "NEWER INSTALLED") {
		t.Fatalf("older catalog entry must not offer an update: %q", detail)
	}
}

func TestPackUpdateSummaryUpdated(t *testing.T) {
	manifest := &modelpack.Manifest{Version: "1.0.1"}
	outcome := packupdate.Result{
		Outcome: packupdate.OutcomeUpdated,
		Retired: []uninstall.Result{{Items: []uninstall.Item{{Artifact: "Old Target", Outcome: "Removed"}}}},
	}
	got := strings.Join(packUpdateSummary(manifest, "1.0.0", outcome, false), "\n")
	for _, want := range []string{"Pack updated to 1.0.1.", "Old Target\nRemoved", "Press Enter to continue."} {
		if !strings.Contains(got, want) {
			t.Fatalf("updated summary missing %q: %q", want, got)
		}
	}
}

func TestPackUpdateSummaryRetireIncompletePointsToFinishUpdate(t *testing.T) {
	manifest := &modelpack.Manifest{Version: "1.0.1"}
	outcome := packupdate.Result{Outcome: packupdate.OutcomeRetireIncomplete, Err: fmt.Errorf("Stop this model before uninstalling the pack.")}
	got := strings.Join(packUpdateSummary(manifest, "1.0.0", outcome, false), "\n")
	for _, want := range []string{"Pack updated to 1.0.1.", "Finish the update later from Available Packs."} {
		if !strings.Contains(got, want) {
			t.Fatalf("retire-incomplete summary missing %q: %q", want, got)
		}
	}
}

func TestPackUpdateSummaryIncompleteKeepsPreviousVersion(t *testing.T) {
	manifest := &modelpack.Manifest{Version: "1.0.1"}
	outcome := packupdate.Result{
		Outcome:  packupdate.OutcomePrepareIncomplete,
		Prepared: install.Result{Items: []install.Item{{Artifact: install.Artifact{Name: "New Target"}, Outcome: install.Failed, Reason: "network unavailable"}}},
	}
	got := strings.Join(packUpdateSummary(manifest, "1.0.0", outcome, false), "\n")
	for _, want := range []string{
		"Update did not complete.",
		"Version 1.0.0 is still installed.",
		"New Target\nFailed — network unavailable",
		"Retry missing models later from Models.",
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("incomplete summary missing %q: %q", want, got)
		}
	}
}

func TestUpdateConfirmationShowsVersionsAndDefaultsToNo(t *testing.T) {
	application, output := updateTestApp(t, []byte{'\n'}, t.TempDir(), t.TempDir(), t.TempDir())
	manifest := updateTestManifest("1.0.1", "example/target", "rev-new")
	plan := install.Plan{SelectedTargetIDs: []string{"target"}}
	yes, err := application.confirmPackUpdate(&manifest, plan, "1.0.0")
	if err != nil || yes {
		t.Fatalf("confirmPackUpdate() = %v, %v", yes, err)
	}
	got := stripANSI(readTestOutput(t, output))
	for _, want := range []string{"Update Update Pack to 1.0.1?", "CURRENT", "v1.0.0", "AVAILABLE", "v1.0.1", "▸ NO"} {
		if !strings.Contains(got, want) {
			t.Fatalf("update confirmation missing %q: %q", want, got)
		}
	}
}

// TestUpdatePackEndToEnd drives the real update wrapper: confirmation,
// preparation through the normal install flow (new version imported into the
// pack store, artifacts verified), retirement of the old version, and the
// result screen. The new revision is already on disk — exactly the hotfix
// situation — so no download happens; the runtime image is served by the
// docker stub.
func TestUpdatePackEndToEnd(t *testing.T) {
	models := t.TempDir()
	packs := t.TempDir()
	data := t.TempDir()
	updateDockerStub(t)
	importUpdatePack(t, packs, updateTestManifest("1.0.0", "example/target", "rev-old"))
	writeUpdateManagedModel(t, models, "example/target", "rev-new")
	newManifest := updateTestManifest("1.0.1", "example/target", "rev-new")
	source, manifest := writeUpdatePackSource(t, newManifest)

	events := []byte{27, '[', 'B', '\n', '\n'} // confirm Yes, continue past result
	application, output := updateTestApp(t, events, models, packs, data)
	inventory, err := modelstore.Scan(models)
	if err != nil {
		t.Fatal(err)
	}
	plan, err := install.BuildPlan(manifest, []string{"target"}, inventory)
	if err != nil {
		t.Fatal(err)
	}
	if err := application.updatePack(source, manifest, plan, "1.0.0"); err != nil {
		t.Fatalf("updatePack() error = %v", err)
	}
	got := stripANSI(readTestOutput(t, output))
	if !strings.Contains(got, "Pack updated to 1.0.1.") {
		t.Fatalf("update result not shown: %q", lastFrame(got))
	}
	installed, err := packstore.List(packs)
	if err != nil || len(installed) != 1 || installed[0].Version != "1.0.1" {
		t.Fatalf("installed packs after update = %#v, %v", installed, err)
	}
	if _, err := os.Stat(filepath.Join(models, modelstore.DestinationName("example/target"))); err != nil {
		t.Fatalf("new model revision was not retained: %v", err)
	}
	entries := inventoryEntries(t, application)
	if _, found := findEntry(entries, "example/target", "rev-new"); !found {
		t.Fatalf("models inventory lacks the new revision: %#v", entries)
	}
	if _, found := findEntry(entries, "example/target", "rev-old"); found {
		t.Fatalf("models inventory kept the retired revision: %#v", entries)
	}
}

// TestFinishUpdateScreenRetiresOldVersionWithoutRedownload reproduces the
// dual-version state (both pack versions installed, newest already current,
// only the new revision on disk): the finish-update screen must retire the
// old version without acquiring anything.
func TestFinishUpdateScreenRetiresOldVersionWithoutRedownload(t *testing.T) {
	models := t.TempDir()
	packs := t.TempDir()
	data := t.TempDir()
	updateDockerStub(t)
	importUpdatePack(t, packs, updateTestManifest("1.0.0", "example/target", "rev-old"))
	importUpdatePack(t, packs, updateTestManifest("1.0.1", "example/target", "rev-new"))
	writeUpdateManagedModel(t, models, "example/target", "rev-new")
	newPackBytes, err := os.ReadFile(filepath.Join(packs, "update-pack", "1.0.1", "pack.json"))
	if err != nil {
		t.Fatal(err)
	}

	events := []byte{'\n', 27, '[', 'B', '\n', '\n'} // Remove Old Version, confirm Yes, continue
	application, output := updateTestApp(t, events, models, packs, data)
	entry := catalog.Entry{ID: "update-pack", Name: "Update Pack", Version: "1.0.1"}
	if err := application.packFinishUpdateScreen(entry, packupdate.Classify(mustListInstalled(t, packs), entry)); err != nil {
		t.Fatalf("packFinishUpdateScreen() error = %v", err)
	}
	got := stripANSI(readTestOutput(t, output))
	for _, want := range []string{"Remove old pack version 1.0.0?", "OLD VERSION REMOVED", "Old pack version removed.", "Current version: 1.0.1"} {
		if !strings.Contains(got, want) {
			t.Fatalf("finish-update screen missing %q: %q", want, lastFrame(got))
		}
	}
	installed, err := packstore.List(packs)
	if err != nil || len(installed) != 1 || installed[0].Version != "1.0.1" {
		t.Fatalf("installed packs after cleanup = %#v, %v", installed, err)
	}
	kept, err := os.ReadFile(filepath.Join(packs, "update-pack", "1.0.1", "pack.json"))
	if err != nil || string(kept) != string(newPackBytes) {
		t.Fatalf("current version changed during cleanup: %v", err)
	}
	if _, err := os.Stat(filepath.Join(models, modelstore.DestinationName("example/target"))); err != nil {
		t.Fatalf("current model revision was removed: %v", err)
	}
	entries := inventoryEntries(t, application)
	if _, found := findEntry(entries, "example/target", "rev-old"); found {
		t.Fatalf("dead old revision still inventoried: %#v", entries)
	}
}

func mustListInstalled(t *testing.T, storeRoot string) []packstore.InstalledPack {
	t.Helper()
	installed, err := packstore.List(storeRoot)
	if err != nil {
		t.Fatal(err)
	}
	return installed
}

// TestModelsInventoryDropsOldRepositoryRevisionAfterUpdate covers the original
// acceptance observation: with both pack versions installed the Models screen
// lists the repository twice (old revision Missing, new revision Present);
// after retiring the old version only the new revision remains.
func TestModelsInventoryDropsOldRepositoryRevisionAfterUpdate(t *testing.T) {
	models := t.TempDir()
	packs := t.TempDir()
	data := t.TempDir()
	updateDockerStub(t)
	importUpdatePack(t, packs, updateTestManifest("1.0.0", "example/target", "rev-old"))
	importUpdatePack(t, packs, updateTestManifest("1.0.1", "example/target", "rev-new"))
	writeUpdateManagedModel(t, models, "example/target", "rev-new")
	application, _ := updateTestApp(t, nil, models, packs, data)

	entries := inventoryEntries(t, application)
	if _, found := findEntry(entries, "example/target", "rev-old"); !found {
		t.Fatalf("dual-version state lacks the old requirement row: %#v", entries)
	}
	if _, err := packupdate.RetireOlder(packs, data, models, "update-pack", "1.0.1"); err != nil {
		t.Fatalf("RetireOlder() error = %v", err)
	}
	entries = inventoryEntries(t, application)
	if entry, found := findEntry(entries, "example/target", "rev-new"); !found || entry.orphan || entry.readiness.State != modelstore.Present {
		t.Fatalf("new revision not a Present requirement row: %#v, %v", entry, found)
	}
	if _, found := findEntry(entries, "example/target", "rev-old"); found {
		t.Fatalf("old revision row survived the update: %#v", entries)
	}
	if len(entries) != 1 {
		t.Fatalf("inventory = %#v", entries)
	}
}
