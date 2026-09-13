package tui

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"b70ctl/internal/config"
	"b70ctl/internal/hf"
	"b70ctl/internal/modelpack"
	"b70ctl/internal/modelstore"
	"b70ctl/internal/packstore"
	"b70ctl/internal/runtime"
)

func TestModelScreenActionsFollowState(t *testing.T) {
	tests := []struct {
		name      string
		build     func(t *testing.T, root string) modelpack.Model
		actions   []string
		forbidden []string
		hint      string
	}{
		{
			name:      "present exposes delete",
			build:     func(t *testing.T, root string) modelpack.Model { return writePresentModel(t, root) },
			actions:   []string{"DELETE"},
			forbidden: []string{"REPLACE", "DOWNLOAD"},
		},
		{
			name:      "incomplete exposes replace",
			build:     func(t *testing.T, root string) modelpack.Model { return writeIncompleteModel(t, root) },
			actions:   []string{"REPLACE"},
			forbidden: []string{"DELETE", "DOWNLOAD"},
		},
		{
			name:      "missing exposes download",
			build:     func(t *testing.T, root string) modelpack.Model { return tuiTestModel() },
			actions:   []string{"DOWNLOAD"},
			forbidden: []string{"DELETE", "REPLACE"},
		},
		{
			name: "missing with occupied destination exposes replace",
			build: func(t *testing.T, root string) modelpack.Model {
				model := tuiTestModel()
				writeTestModelFile(t, filepath.Join(root, "example__model", "junk.bin"), "junk")
				return model
			},
			actions:   []string{"REPLACE"},
			forbidden: []string{"DELETE", "DOWNLOAD"},
			hint:      "occupied by an unrecognized directory",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			models := t.TempDir()
			model := test.build(t, models)
			application, output := modelActionTestApp(t, []byte{27}, models, t.TempDir())
			if err := application.modelScreen(model, assessFixture(t, models, model)); err != nil {
				t.Fatal(err)
			}
			got := stripANSI(readTestOutput(t, output))
			for _, action := range test.actions {
				if !strings.Contains(got, action) {
					t.Fatalf("model screen for %q missing action %q:\n%s", test.name, action, got)
				}
			}
			for _, action := range test.forbidden {
				if strings.Contains(got, action) {
					t.Fatalf("model screen for %q shows action %q:\n%s", test.name, action, got)
				}
			}
			if test.hint != "" && !strings.Contains(got, test.hint) {
				t.Fatalf("model screen for %q missing hint %q:\n%s", test.name, test.hint, got)
			}
		})
	}
}

func TestDeleteConfirmedRemovesOnlySelectedModel(t *testing.T) {
	models := t.TempDir()
	model := writePresentModel(t, models)
	writePresentModelAt(t, models, "other", "example/other", "87654321", 6)
	application, output := modelActionTestApp(t, deleteFlowEvents, models, t.TempDir())
	installTuiPack(t, application.paths.PacksRoot, "pack-a", "Pack A", model)

	if err := application.modelScreen(model, assessFixture(t, models, model)); err != nil {
		t.Fatal(err)
	}
	target := filepath.Join(models, "example__model")
	if _, err := os.Stat(target); !os.IsNotExist(err) {
		t.Fatalf("model was not deleted: %v", err)
	}
	if _, err := os.Stat(filepath.Join(models, "other")); err != nil {
		t.Fatalf("unrelated model was removed: %v", err)
	}
	got := stripANSI(readTestOutput(t, output))
	for _, want := range []string{"Model deleted.", "● MISSING", "DOWNLOAD"} {
		if !strings.Contains(got, want) {
			t.Fatalf("delete flow output missing %q:\n%s", want, got)
		}
	}
}

func TestDeleteRefusedForExternalHFCacheModel(t *testing.T) {
	models := t.TempDir()
	model := tuiTestModel()
	snapshot := filepath.Join(models, "hub", "models--example--model", "snapshots", model.Revision)
	writeTestModelFile(t, filepath.Join(snapshot, "model.bin"), "weight")
	application, output := modelActionTestApp(t, deleteFlowEvents, models, t.TempDir())
	installTuiPack(t, application.paths.PacksRoot, "pack-a", "Pack A", model)

	if err := application.modelScreen(model, assessFixture(t, models, model)); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(snapshot, "model.bin")); err != nil {
		t.Fatalf("Hugging Face snapshot was removed: %v", err)
	}
	if want := "Not deleted — external Hugging Face cache."; !strings.Contains(stripANSI(readTestOutput(t, output)), want) {
		t.Fatalf("delete flow did not report the retained cache:\n%s", readTestOutput(t, output))
	}
}

func TestDeleteRefusedForMultipleLocalCopies(t *testing.T) {
	models := t.TempDir()
	model := tuiTestModel()
	writePresentModelAt(t, models, "first", model.Repo, model.Revision, 6)
	writePresentModelAt(t, models, "second", model.Repo, model.Revision, 6)
	application, output := modelActionTestApp(t, deleteFlowEvents, models, t.TempDir())
	installTuiPack(t, application.paths.PacksRoot, "pack-a", "Pack A", model)

	if err := application.modelScreen(model, assessFixture(t, models, model)); err != nil {
		t.Fatal(err)
	}
	for _, copy := range []string{"first", "second"} {
		if _, err := os.Stat(filepath.Join(models, copy)); err != nil {
			t.Fatalf("copy %q was removed: %v", copy, err)
		}
	}
	if want := "Not deleted — multiple local copies found."; !strings.Contains(stripANSI(readTestOutput(t, output)), want) {
		t.Fatalf("delete flow did not report multiple copies:\n%s", readTestOutput(t, output))
	}
}

func TestDeleteRefusedWhileModelIsStartingOrRunning(t *testing.T) {
	tests := []struct {
		name  string
		state runtime.State
	}{
		{name: "starting", state: runtime.StateStarting},
		{name: "running", state: runtime.StateRunning},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			models := t.TempDir()
			model := writePresentModel(t, models)
			application, output := modelActionTestApp(t, []byte{'\n', 27}, models, t.TempDir())
			installTuiPack(t, application.paths.PacksRoot, "pack-a", "Pack A", model)
			stubRuntimeStatus(t, runtime.StatusResult{
				State: test.state, PackID: "pack-a", PackVersion: "1.0.0", ProfileID: "profile",
			}, nil)

			if err := application.modelScreen(model, assessFixture(t, models, model)); err != nil {
				t.Fatal(err)
			}
			if _, err := os.Stat(filepath.Join(models, "example__model")); err != nil {
				t.Fatalf("model was deleted: %v", err)
			}
			got := stripANSI(readTestOutput(t, output))
			if !strings.Contains(got, "Stop it first") {
				t.Fatalf("delete was not refused for a %s model:\n%s", test.name, got)
			}
			if strings.Contains(got, "CONFIRM") {
				t.Fatalf("confirmation was offered for an in-use model:\n%s", got)
			}
		})
	}
}

func TestDeleteAllowedWhenRunningPackDoesNotMountModel(t *testing.T) {
	models := t.TempDir()
	model := writePresentModel(t, models)
	application, output := modelActionTestApp(t, deleteFlowEvents, models, t.TempDir())
	installTuiPack(t, application.paths.PacksRoot, "pack-a", "Pack A", model)
	installTuiPack(t, application.paths.PacksRoot, "pack-b", "Pack B", tuiTestModelAt("example/unmounted", "87654321"))
	stubRuntimeStatus(t, runtime.StatusResult{
		State: runtime.StateRunning, PackID: "pack-b", PackVersion: "1.0.0", ProfileID: "profile",
	}, nil)

	if err := application.modelScreen(model, assessFixture(t, models, model)); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(models, "example__model")); !os.IsNotExist(err) {
		t.Fatalf("unmounted model was not deleted: %v", err)
	}
	if want := "Model deleted."; !strings.Contains(stripANSI(readTestOutput(t, output)), want) {
		t.Fatalf("delete of an unmounted model did not succeed:\n%s", readTestOutput(t, output))
	}
}

func TestDeleteRefusedForModelSharedByInstalledPacks(t *testing.T) {
	models := t.TempDir()
	model := writePresentModel(t, models)
	application, output := modelActionTestApp(t, []byte{'\n', 27}, models, t.TempDir())
	installTuiPack(t, application.paths.PacksRoot, "pack-a", "Pack A", model)
	installTuiPack(t, application.paths.PacksRoot, "pack-b", "Pack B", tuiTestModelAt(model.Repo, model.Revision))

	if err := application.modelScreen(model, assessFixture(t, models, model)); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(models, "example__model")); err != nil {
		t.Fatalf("shared model was removed: %v", err)
	}
	got := stripANSI(readTestOutput(t, output))
	if !strings.Contains(got, "shared by 2 installed packs") {
		t.Fatalf("delete was not refused for a shared checkpoint:\n%s", got)
	}
	if strings.Contains(got, "CONFIRM") {
		t.Fatalf("confirmation was offered for a shared checkpoint:\n%s", got)
	}
}

func TestDeleteRemoveFailureSurfacesErrorWithoutClaimingSuccess(t *testing.T) {
	if os.Geteuid() == 0 {
		t.Skip("permission-based failure injection does not work as root")
	}
	models := t.TempDir()
	model := writePresentModel(t, models)
	application, output := modelActionTestApp(t, deleteFlowEvents, models, t.TempDir())
	installTuiPack(t, application.paths.PacksRoot, "pack-a", "Pack A", model)
	if err := os.Chmod(models, 0o555); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { os.Chmod(models, 0o755) })

	if err := application.modelScreen(model, assessFixture(t, models, model)); err != nil {
		t.Fatal(err)
	}
	got := stripANSI(readTestOutput(t, output))
	if !strings.Contains(got, "Not deleted — remove model artifact") {
		t.Fatalf("remove failure did not surface a useful error:\n%s", got)
	}
	if strings.Contains(got, "Model deleted.") {
		t.Fatalf("failed delete claimed success:\n%s", got)
	}
}

func TestReplaceIncompleteModelRedownloadsAndEndsPresent(t *testing.T) {
	models := t.TempDir()
	model := writeIncompleteModel(t, models)
	application, output := modelActionTestApp(t, replaceFlowEvents, models, t.TempDir())
	installTuiPack(t, application.paths.PacksRoot, "pack-a", "Pack A", model)
	stubHFClient(t, fakeHFClientFor(model))

	if err := application.modelScreen(model, assessFixture(t, models, model)); err != nil {
		t.Fatal(err)
	}
	destination := filepath.Join(models, "example__model")
	contents, err := os.ReadFile(filepath.Join(destination, "model.bin"))
	if err != nil || string(contents) != "weight" {
		t.Fatalf("replaced model.bin = %q, %v", contents, err)
	}
	if _, err := os.Stat(filepath.Join(destination, ".b70-model.json")); err != nil {
		t.Fatalf("replaced model has no marker: %v", err)
	}
	got := stripANSI(readTestOutput(t, output))
	for _, want := range []string{"DOWNLOAD COMPLETE", "● PRESENT"} {
		if !strings.Contains(got, want) {
			t.Fatalf("replace flow output missing %q:\n%s", want, got)
		}
	}
}

func TestReplaceClearsOccupiedDestinationForMissingModel(t *testing.T) {
	models := t.TempDir()
	model := tuiTestModel()
	junk := filepath.Join(models, "example__model")
	writeTestModelFile(t, filepath.Join(junk, "junk.bin"), "junk")
	application, output := modelActionTestApp(t, replaceFlowEvents, models, t.TempDir())
	stubHFClient(t, fakeHFClientFor(model))

	if err := application.modelScreen(model, assessFixture(t, models, model)); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(junk, "junk.bin")); !os.IsNotExist(err) {
		t.Fatalf("occupied destination junk survived: %v", err)
	}
	contents, err := os.ReadFile(filepath.Join(junk, "model.bin"))
	if err != nil || string(contents) != "weight" {
		t.Fatalf("re-downloaded model.bin = %q, %v", contents, err)
	}
	if want := "● PRESENT"; !strings.Contains(stripANSI(readTestOutput(t, output)), want) {
		t.Fatalf("replace flow did not end Present:\n%s", readTestOutput(t, output))
	}
}

func TestReplaceRefusedWhenDestinationHoldsRecognizedModel(t *testing.T) {
	models := t.TempDir()
	model := tuiTestModel()
	squatter := tuiTestModelAt("example/squatter", "87654321")
	writePresentModelAt(t, models, "example__model", squatter.Repo, squatter.Revision, 6)
	application, output := modelActionTestApp(t, deleteFlowEvents, models, t.TempDir())

	if err := application.modelScreen(model, assessFixture(t, models, model)); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(models, "example__model", ".b70-model.json")); err != nil {
		t.Fatalf("recognized destination model was removed: %v", err)
	}
	got := stripANSI(readTestOutput(t, output))
	if !strings.Contains(got, "destination holds another recognized model") {
		t.Fatalf("replace did not refuse a recognized destination:\n%s", got)
	}
	if strings.Contains(got, "DOWNLOAD COMPLETE") {
		t.Fatalf("refused replace claimed a download:\n%s", got)
	}
}

func TestReplaceFailedRedownloadDoesNotClaimSuccess(t *testing.T) {
	models := t.TempDir()
	model := writeIncompleteModel(t, models)
	application, output := modelActionTestApp(t, replaceFlowEvents, models, t.TempDir())
	stubHFClient(t, &fakeHFClient{installErr: hf.ErrNetworkUnavailable})

	if err := application.modelScreen(model, assessFixture(t, models, model)); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(models, "example__model")); !os.IsNotExist(err) {
		t.Fatalf("failed re-download left a model behind: %v", err)
	}
	got := stripANSI(readTestOutput(t, output))
	for _, want := range []string{"network unavailable", "● MISSING"} {
		if !strings.Contains(got, want) {
			t.Fatalf("failed replace output missing %q:\n%s", want, got)
		}
	}
	if strings.Contains(got, "DOWNLOAD COMPLETE") || strings.Contains(got, "● PRESENT") {
		t.Fatalf("failed replace claimed success:\n%s", got)
	}
}

func TestPackRequiringCountsOnlyExactIdentities(t *testing.T) {
	packs := []loadedPack{
		{manifest: &modelpack.Manifest{Models: []modelpack.Model{tuiTestModel()}}},
		{manifest: &modelpack.Manifest{Models: []modelpack.Model{tuiTestModelAt("example/model", "11111111")}}},
	}
	if got := packsRequiring(packs, "example/model", tuiTestRevision); len(got) != 1 {
		t.Fatalf("packsRequiring() matched %d packs for the exact revision", len(got))
	}
	if got := packsRequiring(packs, "example/model", "11111111"); len(got) != 1 {
		t.Fatalf("packsRequiring() matched %d packs for the alternate revision", len(got))
	}
}

const tuiTestRevision = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"

// deleteFlowEvents activates the Delete action, confirms the destructive
// prompt, then leaves the screen.
var deleteFlowEvents = []byte{'\n', 27, '[', 'B', '\n', 27}

// replaceFlowEvents activates the Replace action, confirms the destructive
// replacement, confirms the download, continues past the completion screen,
// then leaves the screen.
var replaceFlowEvents = []byte{'\n', 27, '[', 'B', '\n', 27, '[', 'B', '\n', '\n', 27}

type fakeHFClient struct {
	repository hf.Repository
	installErr error
}

func fakeHFClientFor(model modelpack.Model) *fakeHFClient {
	var size int64 = 6
	return &fakeHFClient{
		repository: hf.Repository{Repo: model.Repo, Revision: model.Revision, Files: []hf.File{
			{Path: "model.bin", Size: size, SizeKnown: true},
		}},
	}
}

func (client *fakeHFClient) Inspect(context.Context, string, string) (hf.Repository, error) {
	return client.repository, nil
}

func (client *fakeHFClient) Install(_ context.Context, modelRoot string, repository hf.Repository, progress func(hf.Progress)) (hf.Result, error) {
	if client.installErr != nil {
		return hf.Result{}, client.installErr
	}
	destination := filepath.Join(modelRoot, modelstore.DestinationName(repository.Repo))
	if err := os.MkdirAll(destination, 0o755); err != nil {
		return hf.Result{}, err
	}
	for index, file := range repository.Files {
		path := filepath.Join(destination, filepath.FromSlash(file.Path))
		contents := "weight"[:min(int(file.Size), len("weight"))]
		if err := os.WriteFile(path, []byte(contents), 0o644); err != nil {
			return hf.Result{}, err
		}
		if progress != nil {
			progress(hf.Progress{CurrentFile: file.Path, FileNumber: index + 1, TotalFiles: len(repository.Files)})
		}
	}
	marker := fmt.Sprintf(`{"repo":%q,"revision":%q}`, repository.Repo, repository.Revision)
	if err := os.WriteFile(filepath.Join(destination, ".b70-model.json"), []byte(marker), 0o644); err != nil {
		return hf.Result{}, err
	}
	return hf.Result{Path: destination}, nil
}

func stubHFClient(t *testing.T, client hfClient) {
	t.Helper()
	original := newHFClient
	newHFClient = func(string) hfClient { return client }
	t.Cleanup(func() { newHFClient = original })
}

func stubRuntimeStatus(t *testing.T, result runtime.StatusResult, statusErr error) {
	t.Helper()
	original := runtimeStatus
	runtimeStatus = func() (runtime.StatusResult, error) { return result, statusErr }
	t.Cleanup(func() { runtimeStatus = original })
}

func modelActionTestApp(t *testing.T, events []byte, modelsRoot, packsRoot string) (*app, *os.File) {
	t.Helper()
	t.Setenv("XDG_CONFIG_HOME", filepath.Join(t.TempDir(), "config"))
	application, output := uninstallTestApp(t, events)
	application.config = config.Config{ModelDirectory: modelsRoot, DefaultAccess: config.AccessLocal, DefaultPort: 8000}
	application.paths = config.Paths{PacksRoot: packsRoot}
	stubRuntimeStatus(t, runtime.StatusResult{State: runtime.StateStopped}, nil)
	return application, output
}

func assessFixture(t *testing.T, modelsRoot string, model modelpack.Model) modelstore.ReadinessResult {
	t.Helper()
	artifacts, err := modelstore.Scan(modelsRoot)
	if err != nil {
		t.Fatal(err)
	}
	return modelstore.Assess(artifacts, model)
}

func tuiTestModel() modelpack.Model {
	return tuiTestModelAt("example/model", tuiTestRevision)
}

func tuiTestModelAt(repo, revision string) modelpack.Model {
	size := int64(6)
	return modelpack.Model{
		ID: "target", Name: "Test Model", Kind: "target", Repo: repo, Revision: revision,
		Files:     []modelpack.ModelFile{{Path: "model.bin", Size: &size}},
		MountPath: "/models/target",
		Launch:    modelpack.LaunchBlock{DockerArgs: []string{}, Environment: map[string]string{}, CommandArgs: []string{}},
	}
}

func writePresentModel(t *testing.T, root string) modelpack.Model {
	t.Helper()
	model := tuiTestModel()
	writePresentModelAt(t, root, modelstore.DestinationName(model.Repo), model.Repo, model.Revision, 6)
	return model
}

func writeIncompleteModel(t *testing.T, root string) modelpack.Model {
	t.Helper()
	model := tuiTestModel()
	directory := filepath.Join(root, modelstore.DestinationName(model.Repo))
	writeTestModelMarker(t, directory, model.Repo, model.Revision)
	writeTestModelFile(t, filepath.Join(directory, "model.bin"), "bad")
	return model
}

func writePresentModelAt(t *testing.T, root, directory, repo, revision string, size int) {
	t.Helper()
	path := filepath.Join(root, directory)
	writeTestModelMarker(t, path, repo, revision)
	writeTestModelFile(t, filepath.Join(path, "model.bin"), "weight"[:size])
}

func installTuiPack(t *testing.T, storeRoot, id, name string, model modelpack.Model) {
	t.Helper()
	source := t.TempDir()
	empty := modelpack.LaunchBlock{DockerArgs: []string{}, Environment: map[string]string{}, CommandArgs: []string{}}
	manifest := modelpack.Manifest{
		SchemaVersion: 1, ID: id, Name: name, Version: "1.0.0",
		Models: []modelpack.Model{model},
		Runtimes: []modelpack.Runtime{{
			ID: "runtime", Image: "example/runtime:1", Digest: "sha256:" + strings.Repeat("a", 64),
			ContainerPort: 8000, HealthPath: "/health",
			Launch: modelpack.RuntimeLaunch{DockerArgs: []string{}, Environment: map[string]string{}, Command: []string{"serve"}},
		}},
		Modes:    []modelpack.Mode{{ID: "base", DisplayName: "Base", Assistants: []string{}, Launch: empty}},
		Profiles: []modelpack.Profile{{ID: "profile", ModelID: "target", RuntimeID: "runtime", Cards: 1, TensorParallel: 1, Context: 4096, Mode: "base", Launch: empty}},
	}
	writeTuiPack(t, source, manifest)
	if _, err := packstore.Import(source, storeRoot, packstore.SourceLocal); err != nil {
		t.Fatal(err)
	}
}

func writeTuiPack(t *testing.T, root string, manifest modelpack.Manifest) {
	t.Helper()
	writeTestModelFile(t, filepath.Join(root, "README.md"), "# Test pack\n")
	data, err := json.Marshal(manifest)
	if err != nil {
		t.Fatal(err)
	}
	writeTestModelFile(t, filepath.Join(root, "pack.json"), string(data))
}

func writeTestModelMarker(t *testing.T, directory, repo, revision string) {
	t.Helper()
	writeTestModelFile(t, filepath.Join(directory, ".b70-model.json"), fmt.Sprintf(`{"repo":%q,"revision":%q}`, repo, revision))
}

func writeTestModelFile(t *testing.T, path, contents string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(contents), 0o644); err != nil {
		t.Fatal(err)
	}
}
