package uninstall

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"b70ctl/internal/modelpack"
	"b70ctl/internal/modelstore"
	"b70ctl/internal/packstore"
	"b70ctl/internal/runtime"
)

func mustRecordOwnership(t *testing.T, dataRoot, digest, image string) {
	t.Helper()
	if err := runtime.RecordOwnership(dataRoot, digest, image); err != nil {
		t.Fatal(err)
	}
}

func TestKeepModelsSupportsPackReinstallAndArtifactReuse(t *testing.T) {
	store := t.TempDir()
	data := t.TempDir()
	models := t.TempDir()
	source := t.TempDir()
	digest := "sha256:" + strings.Repeat("a", 64)
	manifest := uninstallManifest("qwen-test", "Qwen Test", "example/model", "revision-1", digest)
	writePack(t, source, manifest)
	modelPath := filepath.Join(models, "example__model")
	writeTestFile(t, filepath.Join(modelPath, ".b70-model.json"), `{"repo":"example/model","revision":"revision-1"}`)
	writeTestFile(t, filepath.Join(modelPath, "weights.bin"), "weights")
	installDockerStub(t, digest, false, false)
	mustRecordOwnership(t, data, digest, "example/runtime:1")

	if _, err := packstore.Import(source, store, packstore.SourceLocal); err != nil {
		t.Fatal(err)
	}
	before, err := modelstore.Scan(models)
	if err != nil {
		t.Fatal(err)
	}
	artifact, present := modelstore.Find(before, "example/model", "revision-1")
	if !present {
		t.Fatal("model was not present before uninstall")
	}

	result, err := Run(store, data, models, manifest.ID, manifest.Version, KeepModels)
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	wantItems := []Item{
		{Artifact: "Test Model", Outcome: "Retained — keeping models"},
		{Artifact: "Runtime image", Outcome: "Removed"},
	}
	if !result.PackRemoved || !reflect.DeepEqual(result.Items, wantItems) {
		t.Fatalf("Run() = %#v, want items %#v", result, wantItems)
	}
	owned, err := runtime.LoadOwnership(data)
	if err != nil {
		t.Fatal(err)
	}
	if _, exists := owned[digest]; exists {
		t.Fatal("ownership record survived the removed image")
	}

	if _, err := packstore.Import(source, store, packstore.SourceLocal); err != nil {
		t.Fatalf("reinstall failed: %v", err)
	}
	after, err := modelstore.Scan(models)
	if err != nil {
		t.Fatal(err)
	}
	reused, present := modelstore.Find(after, "example/model", "revision-1")
	if !present || reused.Path != artifact.Path {
		t.Fatalf("reinstalled pack did not reuse model: %#v, %v", reused, present)
	}
}

func TestDeleteEverythingRetainsArtifactsSharedByAnotherPack(t *testing.T) {
	store := t.TempDir()
	models := t.TempDir()
	digest := "sha256:" + strings.Repeat("b", 64)
	for _, id := range []string{"first-pack", "second-pack"} {
		source := t.TempDir()
		manifest := uninstallManifest(id, id, "example/shared", "revision-1", digest)
		writePack(t, source, manifest)
		if _, err := packstore.Import(source, store, packstore.SourceLocal); err != nil {
			t.Fatal(err)
		}
	}
	modelPath := filepath.Join(models, "shared")
	writeTestFile(t, filepath.Join(modelPath, ".b70-model.json"), `{"repo":"example/shared","revision":"revision-1"}`)
	installDockerStub(t, digest, false, false)

	result, err := Run(store, t.TempDir(), models, "first-pack", "1.0.0", DeleteEverything)
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	for _, item := range result.Items {
		if item.Outcome != "Retained — used by another installed pack" {
			t.Fatalf("result item = %#v", item)
		}
	}
	if _, err := os.Stat(modelPath); err != nil {
		t.Fatalf("shared model was removed: %v", err)
	}
	installed, err := packstore.List(store)
	if err != nil || len(installed) != 1 || installed[0].ID != "second-pack" {
		t.Fatalf("installed packs = %#v, %v", installed, err)
	}
}

func TestRunningTargetPackRefusesUninstall(t *testing.T) {
	store := t.TempDir()
	models := t.TempDir()
	digest := "sha256:" + strings.Repeat("c", 64)
	source := t.TempDir()
	manifest := uninstallManifest("running-pack", "Running Pack", "example/model", "revision-1", digest)
	writePack(t, source, manifest)
	if _, err := packstore.Import(source, store, packstore.SourceLocal); err != nil {
		t.Fatal(err)
	}
	installDockerStub(t, digest, true, false)

	_, err := Run(store, t.TempDir(), models, manifest.ID, manifest.Version, DeleteEverything)
	if err == nil || err.Error() != "Stop this model before uninstalling the pack." {
		t.Fatalf("Run() error = %v", err)
	}
	installed, listErr := packstore.List(store)
	if listErr != nil || len(installed) != 1 {
		t.Fatalf("pack was removed after refusal: %#v, %v", installed, listErr)
	}
}

func TestBuildPlanUsesExactModelIdentityAndRuntimeDigest(t *testing.T) {
	digestA := "sha256:" + strings.Repeat("a", 64)
	digestB := "sha256:" + strings.Repeat("b", 64)
	target := uninstallManifest("target", "Target", "example/model", "revision-1", digestA)

	t.Run("exact identities are shared regardless of local IDs", func(t *testing.T) {
		other := uninstallManifest("other", "Other", "example/model", "revision-1", digestA)
		other.Models[0].ID = "different-model-id"
		other.Runtimes[0].ID = "different-runtime-id"
		models, runtimes := buildPlan(&target, []*modelpack.Manifest{&other})
		if len(models) != 1 || !models[0].shared || len(runtimes) != 1 || !runtimes[0].shared {
			t.Fatalf("buildPlan() = %#v, %#v", models, runtimes)
		}
	})

	t.Run("matching IDs do not share different immutable identities", func(t *testing.T) {
		other := uninstallManifest("other", "Other", "example/model", "revision-2", digestB)
		models, runtimes := buildPlan(&target, []*modelpack.Manifest{&other})
		if len(models) != 1 || models[0].shared || len(runtimes) != 1 || runtimes[0].shared {
			t.Fatalf("buildPlan() = %#v, %#v", models, runtimes)
		}
	})
}

func TestBuildPlanDeduplicatesExactArtifacts(t *testing.T) {
	digest := "sha256:" + strings.Repeat("d", 64)
	target := uninstallManifest("target", "Target", "example/model", "revision-1", digest)
	duplicateModel := target.Models[0]
	duplicateModel.ID = "assistant"
	duplicateModel.Kind = "assistant"
	target.Models = append(target.Models, duplicateModel)
	duplicateRuntime := target.Runtimes[0]
	duplicateRuntime.ID = "second-runtime"
	duplicateRuntime.Image = "example/second-tag:1"
	target.Runtimes = append(target.Runtimes, duplicateRuntime)

	models, runtimes := buildPlan(&target, nil)
	if len(models) != 1 || len(runtimes) != 1 {
		t.Fatalf("buildPlan() produced %d model and %d runtime cleanups", len(models), len(runtimes))
	}
}

func TestDockerCleanupRefusalDoesNotRestoreRemovedPack(t *testing.T) {
	store := t.TempDir()
	data := t.TempDir()
	models := t.TempDir()
	digest := "sha256:" + strings.Repeat("e", 64)
	source := t.TempDir()
	manifest := uninstallManifest("refused-pack", "Refused Pack", "example/model", "revision-1", digest)
	writePack(t, source, manifest)
	if _, err := packstore.Import(source, store, packstore.SourceLocal); err != nil {
		t.Fatal(err)
	}
	installDockerStub(t, digest, false, true)
	mustRecordOwnership(t, data, digest, "example/runtime:1")

	result, err := Run(store, data, models, manifest.ID, manifest.Version, KeepModels)
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if got := result.Items[len(result.Items)-1].Outcome; got != "Retained — Docker reports image is in use" {
		t.Fatalf("runtime outcome = %q", got)
	}
	owned, err := runtime.LoadOwnership(data)
	if err != nil {
		t.Fatal(err)
	}
	if _, exists := owned[digest]; !exists {
		t.Fatal("ownership record was dropped while the image remained")
	}
	installed, err := packstore.List(store)
	if err != nil || len(installed) != 0 {
		t.Fatalf("removed pack was restored: %#v, %v", installed, err)
	}
}

func uninstallManifest(id, name, repo, revision, digest string) modelpack.Manifest {
	empty := modelpack.LaunchBlock{DockerArgs: []string{}, Environment: map[string]string{}, CommandArgs: []string{}}
	return modelpack.Manifest{
		SchemaVersion: 1,
		ID:            id,
		Name:          name,
		Version:       "1.0.0",
		Models: []modelpack.Model{
			{ID: "target", Name: "Test Model", Kind: "target", Repo: repo, Revision: revision, Files: []modelpack.ModelFile{{Path: "model.bin"}}, MountPath: "/models/target", Launch: empty},
		},
		Runtimes: []modelpack.Runtime{
			{ID: "runtime", Image: "example/runtime:1", Digest: digest, ContainerPort: 8000, HealthPath: "/health", Launch: modelpack.RuntimeLaunch{DockerArgs: []string{}, Environment: map[string]string{}, Command: []string{"serve"}}},
		},
		Modes: []modelpack.Mode{{ID: "base", DisplayName: "Base", Assistants: []string{}, Launch: empty}},
		Profiles: []modelpack.Profile{
			{ID: "profile", ModelID: "target", RuntimeID: "runtime", Cards: 1, TensorParallel: 1, Context: 4096, Mode: "base", Launch: empty},
		},
	}
}

func writePack(t *testing.T, root string, manifest modelpack.Manifest) {
	t.Helper()
	writeTestFile(t, filepath.Join(root, "README.md"), "# Test pack\n")
	data, err := json.Marshal(manifest)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "pack.json"), data, 0o644); err != nil {
		t.Fatal(err)
	}
}

func writeTestFile(t *testing.T, path, contents string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(contents), 0o644); err != nil {
		t.Fatal(err)
	}
}

func installDockerStub(t *testing.T, digest string, running, refuseRemoval bool) {
	t.Helper()
	installDockerStubWithLog(t, digest, running, refuseRemoval, "")
}

// installLoggingDockerStub installs the Docker stub and returns a function
// counting stub invocations whose arguments start with the given words.
func installLoggingDockerStub(t *testing.T, digest string, running, refuseRemoval bool) func(...string) int {
	t.Helper()
	logPath := filepath.Join(t.TempDir(), "docker.log")
	installDockerStubWithLog(t, digest, running, refuseRemoval, logPath)
	return func(words ...string) int {
		data, err := os.ReadFile(logPath)
		if err != nil {
			t.Fatal(err)
		}
		count := 0
		for _, line := range strings.Split(strings.TrimSpace(string(data)), "\n") {
			fields := strings.Fields(line)
			if len(fields) < len(words) {
				continue
			}
			matches := true
			for index, word := range words {
				if fields[index] != word {
					matches = false
					break
				}
			}
			if matches {
				count++
			}
		}
		return count
	}
}

func installDockerStubWithLog(t *testing.T, digest string, running, refuseRemoval bool, logPath string) {
	t.Helper()
	bin := t.TempDir()
	script := `#!/bin/sh
if [ -n "` + logPath + `" ]; then
  echo "$@" >> "` + logPath + `"
fi
if [ "$1" = "inspect" ]; then
`
	if running {
		script += `cat <<'JSON'
[{"Config":{"Labels":{"b70ctl.managed":"true","b70ctl.pack_id":"running-pack","b70ctl.pack_version":"1.0.0","b70ctl.profile_id":"profile","b70ctl.health_path":"/health"}},"Image":"` + digest + `","State":{"Running":true,"ExitCode":0},"NetworkSettings":{"Ports":{"8000/tcp":[{"HostIp":"127.0.0.1","HostPort":"65534"}]}}}]
JSON
exit 0
`
	} else {
		script += `echo "Error: No such object: b70ctl-runtime" >&2
exit 1
`
	}
	script += fmt.Sprintf(`fi
if [ "$1" = "image" ] && [ "$2" = "inspect" ]; then
  echo '"%s"'
  exit 0
fi
if [ "$1" = "image" ] && [ "$2" = "rm" ]; then
`, digest)
	if refuseRemoval {
		script += `  echo "Error response from daemon: conflict: unable to remove repository reference - image is being used by stopped container" >&2
  exit 1
`
	} else {
		script += `  echo "Untagged: $3"
  exit 0
`
	}
	script += `fi
exit 2
`
	path := filepath.Join(bin, "docker")
	if err := os.WriteFile(path, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", bin+string(os.PathListSeparator)+os.Getenv("PATH"))
}
