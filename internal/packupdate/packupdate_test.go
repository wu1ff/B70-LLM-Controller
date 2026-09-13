package packupdate

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"b70ctl/internal/catalog"
	"b70ctl/internal/install"
	"b70ctl/internal/modelpack"
	"b70ctl/internal/modelstore"
	"b70ctl/internal/packstore"
	"b70ctl/internal/runtime"
)

var testDigest = "sha256:" + strings.Repeat("a", 64)

func packModel(id, name, repo, revision string) modelpack.Model {
	return modelpack.Model{
		ID: id, Name: name, Kind: "target", Repo: repo, Revision: revision,
		Files:     []modelpack.ModelFile{{Path: "model.bin"}},
		MountPath: "/models/target",
		Launch:    modelpack.LaunchBlock{DockerArgs: []string{}, Environment: map[string]string{}, CommandArgs: []string{}},
	}
}

func packManifest(id, name, version string, models []modelpack.Model, digest string) modelpack.Manifest {
	empty := modelpack.LaunchBlock{DockerArgs: []string{}, Environment: map[string]string{}, CommandArgs: []string{}}
	return modelpack.Manifest{
		SchemaVersion: 1, ID: id, Name: name, Version: version, Models: models,
		Runtimes: []modelpack.Runtime{{
			ID: "runtime", Image: "example/runtime:1", Digest: digest,
			ContainerPort: 8000, HealthPath: "/health",
			Launch: modelpack.RuntimeLaunch{DockerArgs: []string{}, Environment: map[string]string{}, Command: []string{"serve"}},
		}},
		Modes:    []modelpack.Mode{{ID: "base", DisplayName: "Base", Assistants: []string{}, Launch: empty}},
		Profiles: []modelpack.Profile{{ID: "profile", ModelID: models[0].ID, RuntimeID: "runtime", Cards: 1, TensorParallel: 1, Context: 4096, Mode: "base", Launch: empty}},
	}
}

func importPack(t *testing.T, storeRoot string, manifest modelpack.Manifest) {
	t.Helper()
	source := t.TempDir()
	writePackSource(t, source, manifest)
	if _, err := packstore.Import(source, storeRoot, packstore.SourceLocal); err != nil {
		t.Fatal(err)
	}
}

func writePackSource(t *testing.T, root string, manifest modelpack.Manifest) {
	t.Helper()
	writeFile(t, filepath.Join(root, "README.md"), "# Test pack\n")
	data, err := json.Marshal(manifest)
	if err != nil {
		t.Fatal(err)
	}
	writeFile(t, filepath.Join(root, "pack.json"), string(data))
}

func writeFile(t *testing.T, path, contents string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(contents), 0o644); err != nil {
		t.Fatal(err)
	}
}

// writeManagedModel places a Controller-managed model (matching marker) in
// the model store.
func writeManagedModel(t *testing.T, root, repo, revision string) string {
	t.Helper()
	directory := filepath.Join(root, modelstore.DestinationName(repo))
	writeFile(t, filepath.Join(directory, ".b70-model.json"), fmt.Sprintf(`{"repo":%q,"revision":%q}`, repo, revision))
	writeFile(t, filepath.Join(directory, "model.bin"), "weights-"+revision)
	return directory
}

// writeExternalModel places a model in the external Hugging Face cache
// layout, which Controller never deletes.
func writeExternalModel(t *testing.T, root, repo, revision string) string {
	t.Helper()
	snapshot := filepath.Join(root, "models--"+strings.ReplaceAll(repo, "/", "--"), "snapshots", revision)
	writeFile(t, filepath.Join(snapshot, "model.bin"), "weights-"+revision)
	return snapshot
}

// installDockerStub intercepts the docker binary. When runningPackID is set,
// the managed container reports that pack version as running; image inspect
// always reports the digest present and image rm succeeds.
func installDockerStub(t *testing.T, digest, runningPackID, runningPackVersion string) {
	t.Helper()
	container := `echo "Error: No such object: b70ctl-runtime" >&2
exit 1
`
	if runningPackID != "" {
		container = fmt.Sprintf(`cat <<'JSON'
[{"Config":{"Labels":{"b70ctl.managed":"true","b70ctl.pack_id":%q,"b70ctl.pack_version":%q,"b70ctl.profile_id":"profile","b70ctl.health_path":"/health"}},"Image":"`+digest+`","State":{"Running":true,"ExitCode":0},"NetworkSettings":{"Ports":{"8000/tcp":[{"HostIp":"127.0.0.1","HostPort":"65534"}]}}}]
JSON
exit 0
`, runningPackID, runningPackVersion)
	}
	script := `#!/bin/sh
if [ "$1" = "inspect" ]; then
` + container + `fi
if [ "$1" = "image" ] && [ "$2" = "inspect" ]; then
  echo '"` + digest + `"'
  exit 0
fi
if [ "$1" = "image" ] && [ "$2" = "rm" ]; then
  echo "Untagged: $3"
  exit 0
fi
exit 2
`
	bin := t.TempDir()
	path := filepath.Join(bin, "docker")
	if err := os.WriteFile(path, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", bin+string(os.PathListSeparator)+os.Getenv("PATH"))
}

func installedPacks(t *testing.T, versions ...string) []packstore.InstalledPack {
	t.Helper()
	installed := make([]packstore.InstalledPack, 0, len(versions))
	for _, version := range versions {
		installed = append(installed, packstore.InstalledPack{ID: "qwen-test", Name: "Qwen Test", Version: version})
	}
	return installed
}

func entry(version string) catalog.Entry {
	return catalog.Entry{ID: "qwen-test", Name: "Qwen Test", Version: version}
}

func TestCompareVersions(t *testing.T) {
	cases := []struct {
		a, b string
		want int
	}{
		{"1.0.0", "1.0.1", -1},
		{"1.0.1", "1.0.0", 1},
		{"1.0.0", "1.0.0", 0},
		{"1.0", "1.0.0", 0},
		{"1.10.0", "1.9.0", 1},
		{"2.0.0", "1.99.99", 1},
		{"0.9.0", "1.0.0", -1},
	}
	for _, testCase := range cases {
		got, err := compareVersions(testCase.a, testCase.b)
		if err != nil || got != testCase.want {
			t.Fatalf("compareVersions(%q, %q) = %d, %v; want %d", testCase.a, testCase.b, got, err, testCase.want)
		}
	}
	for _, invalid := range [][2]string{{"1.0.0-beta", "1.0.0"}, {"", "1.0.0"}, {"1.0.0", ""}, {"1.x.0", "1.0.0"}, {"1..0", "1.0.0"}, {"1.0.0", "v1.0.0"}} {
		if _, err := compareVersions(invalid[0], invalid[1]); err == nil {
			t.Fatalf("compareVersions(%q, %q) unexpectedly succeeded", invalid[0], invalid[1])
		}
	}
}

func TestClassifyUpdateStates(t *testing.T) {
	cases := []struct {
		name      string
		installed []string
		entry     string
		want      Status
		current   string
		old       []string
	}{
		{name: "not installed", installed: nil, entry: "1.0.1", want: StatusNotInstalled},
		{name: "older version installed", installed: []string{"1.0.0"}, entry: "1.0.1", want: StatusUpdateAvailable, current: "1.0.0"},
		{name: "exact version installed", installed: []string{"1.0.1"}, entry: "1.0.1", want: StatusInstalled, current: "1.0.1"},
		{name: "catalog older than installed", installed: []string{"1.0.1"}, entry: "1.0.0", want: StatusOlderCatalog, current: "1.0.1"},
		{name: "multiple old versions one target", installed: []string{"0.9.0", "1.0.0"}, entry: "1.0.1", want: StatusUpdateAvailable, current: "1.0.0", old: []string{"0.9.0", "1.0.0"}},
		{name: "newest installed with old remaining", installed: []string{"1.0.0", "1.0.1"}, entry: "1.0.1", want: StatusFinishUpdate, current: "1.0.1", old: []string{"1.0.0"}},
		{name: "exact plus newer installed", installed: []string{"1.0.1", "1.0.2"}, entry: "1.0.1", want: StatusInstalled, current: "1.0.2"},
	}
	for _, testCase := range cases {
		state := Classify(installedPacks(t, testCase.installed...), entry(testCase.entry))
		if state.Status != testCase.want || state.Current != testCase.current {
			t.Fatalf("%s: Classify() = %+v; want status %s current %s", testCase.name, state, testCase.want, testCase.current)
		}
		if len(testCase.old) > 0 && strings.Join(state.OldVersions, ",") != strings.Join(testCase.old, ",") {
			t.Fatalf("%s: OldVersions = %v; want %v", testCase.name, state.OldVersions, testCase.old)
		}
	}
}

func TestClassifyIncomparableFallsBackToExactVersion(t *testing.T) {
	state := Classify(installedPacks(t, "1.0.0-beta"), entry("1.0.0"))
	if state.Status != StatusIncomparable || state.Reason == "" {
		t.Fatalf("incomparable state = %+v", state)
	}
	exact := Classify(installedPacks(t, "1.0.0-beta"), entry("1.0.0-beta"))
	if exact.Status != StatusInstalled {
		t.Fatalf("exact incomparable version not reported installed: %+v", exact)
	}
}

func TestClassifyCatalogRefreshTransition(t *testing.T) {
	installed := installedPacks(t, "1.0.0")
	if state := Classify(installed, entry("1.0.0")); state.Status != StatusInstalled {
		t.Fatalf("pre-refresh state = %+v", state)
	}
	if state := Classify(installed, entry("1.0.1")); state.Status != StatusUpdateAvailable || state.Current != "1.0.0" {
		t.Fatalf("post-refresh state = %+v", state)
	}
}

func TestPreparationSucceeded(t *testing.T) {
	failedItem := install.Result{Items: []install.Item{{Outcome: install.Failed}}}
	if PreparationSucceeded(failedItem) {
		t.Fatal("failed model artifact must block retirement")
	}
	failedRuntime := install.Result{RuntimeItems: []install.RuntimeItem{{Outcome: runtime.AcquisitionFailed}}}
	if PreparationSucceeded(failedRuntime) {
		t.Fatal("failed runtime acquisition must block retirement")
	}
	skippedGated := install.Result{Items: []install.Item{{Outcome: install.Skipped}}}
	if !PreparationSucceeded(skippedGated) {
		t.Fatal("skipped gated model must not block retirement")
	}
	if !PreparationSucceeded(install.Result{}) {
		t.Fatal("empty preparation must count as succeeded")
	}
}

// applyFixture installs one old pack version (1.0.0) referencing a shared
// model and an old-only model, plus the b70ctl-owned shared runtime image.
func applyFixture(t *testing.T) (store, data, models string) {
	t.Helper()
	store, data, models = t.TempDir(), t.TempDir(), t.TempDir()
	oldManifest := packManifest("qwen-test", "Qwen Test", "1.0.0", []modelpack.Model{
		packModel("shared", "Shared", "example/shared", "rev-shared"),
		packModel("old-only", "Old Only", "example/old", "rev-old"),
	}, testDigest)
	importPack(t, store, oldManifest)
	writeManagedModel(t, models, "example/shared", "rev-shared")
	writeManagedModel(t, models, "example/old", "rev-old")
	installDockerStub(t, testDigest, "", "")
	if err := runtime.RecordOwnership(data, testDigest, "example/runtime:1"); err != nil {
		t.Fatal(err)
	}
	return store, data, models
}

func newVersionManifest() modelpack.Manifest {
	return packManifest("qwen-test", "Qwen Test", "1.0.1", []modelpack.Model{
		packModel("shared", "Shared", "example/shared", "rev-shared"),
		packModel("new-only", "New Only", "example/new", "rev-new"),
	}, testDigest)
}

// importNewVersion stands in for install.Prepare's import step: it installs
// the new pack version into the store, exactly as the real prepare does
// before any artifact work.
func importNewVersion(t *testing.T, store string) {
	t.Helper()
	importPack(t, store, newVersionManifest())
}

func storeVersions(t *testing.T, store, packID string) []string {
	t.Helper()
	installed, err := packstore.List(store)
	if err != nil {
		t.Fatal(err)
	}
	var versions []string
	for _, pack := range installed {
		if pack.ID == packID {
			versions = append(versions, pack.Version)
		}
	}
	return versions
}

func TestApplyInstallsNewVersionBeforeRetiringOld(t *testing.T) {
	store, data, models := applyFixture(t)
	oldStillInstalledWhenPreparing := false
	result := Apply(store, data, models, "qwen-test", "1.0.1", func() (install.Result, error) {
		oldStillInstalledWhenPreparing = len(storeVersions(t, store, "qwen-test")) == 1 &&
			storeVersions(t, store, "qwen-test")[0] == "1.0.0"
		importNewVersion(t, store)
		return install.Result{}, nil
	})
	if result.Outcome != OutcomeUpdated || result.Err != nil {
		t.Fatalf("Apply() = %+v", result)
	}
	if !oldStillInstalledWhenPreparing {
		t.Fatal("old version was not installed while the new version was prepared")
	}
	if versions := storeVersions(t, store, "qwen-test"); len(versions) != 1 || versions[0] != "1.0.1" {
		t.Fatalf("installed versions after update = %v", versions)
	}
}

func TestApplyPrepareFailurePreservesOldPack(t *testing.T) {
	store, data, models := applyFixture(t)
	failure := fmt.Errorf("download failed")
	result := Apply(store, data, models, "qwen-test", "1.0.1", func() (install.Result, error) {
		return install.Result{}, failure
	})
	if result.Outcome != OutcomePrepareFailed || result.Err == nil {
		t.Fatalf("Apply() = %+v", result)
	}
	if versions := storeVersions(t, store, "qwen-test"); len(versions) != 1 || versions[0] != "1.0.0" {
		t.Fatalf("old version did not stay installed: %v", versions)
	}
}

func TestApplyPrepareFailureAfterImportRollsBackNewVersion(t *testing.T) {
	store, data, models := applyFixture(t)
	result := Apply(store, data, models, "qwen-test", "1.0.1", func() (install.Result, error) {
		importNewVersion(t, store)
		return install.Result{}, fmt.Errorf("access verification failed")
	})
	if result.Outcome != OutcomePrepareFailed {
		t.Fatalf("Apply() = %+v", result)
	}
	if versions := storeVersions(t, store, "qwen-test"); len(versions) != 1 || versions[0] != "1.0.0" {
		t.Fatalf("half-installed new version was not rolled back: %v", versions)
	}
}

func TestApplyIncompletePreparationRollsBackAndKeepsOld(t *testing.T) {
	store, data, models := applyFixture(t)
	result := Apply(store, data, models, "qwen-test", "1.0.1", func() (install.Result, error) {
		importNewVersion(t, store)
		return install.Result{Items: []install.Item{{Outcome: install.Failed, Reason: "network unavailable"}}}, nil
	})
	if result.Outcome != OutcomePrepareIncomplete {
		t.Fatalf("Apply() = %+v", result)
	}
	if versions := storeVersions(t, store, "qwen-test"); len(versions) != 1 || versions[0] != "1.0.0" {
		t.Fatalf("incomplete new version was not rolled back: %v", versions)
	}
}

func TestApplyNeverRollsBackPreexistingNewVersion(t *testing.T) {
	store, data, models := applyFixture(t)
	importNewVersion(t, store) // dual-version state predates the attempt
	result := Apply(store, data, models, "qwen-test", "1.0.1", func() (install.Result, error) {
		return install.Result{}, fmt.Errorf("prepare failed")
	})
	if result.Outcome != OutcomePrepareFailed {
		t.Fatalf("Apply() = %+v", result)
	}
	versions := storeVersions(t, store, "qwen-test")
	if len(versions) != 2 {
		t.Fatalf("preexisting versions must be untouched, got %v", versions)
	}
}

func TestRetireOlderRemovesOlderVersionsAndKeepsNewer(t *testing.T) {
	store, data, models := applyFixture(t)
	importPack(t, store, packManifest("qwen-test", "Qwen Test", "0.9.0", []modelpack.Model{
		packModel("shared", "Shared", "example/shared", "rev-shared"),
	}, testDigest))
	importNewVersion(t, store)
	importPack(t, store, packManifest("qwen-test", "Qwen Test", "1.0.2", []modelpack.Model{
		packModel("shared", "Shared", "example/shared", "rev-shared"),
	}, testDigest))

	results, err := RetireOlder(store, data, models, "qwen-test", "1.0.1")
	if err != nil {
		t.Fatalf("RetireOlder() error = %v", err)
	}
	if len(results) != 2 {
		t.Fatalf("retired results = %d, want 2 (0.9.0 and 1.0.0)", len(results))
	}
	versions := storeVersions(t, store, "qwen-test")
	if len(versions) != 2 || versions[0] != "1.0.1" || versions[1] != "1.0.2" {
		t.Fatalf("installed versions after retirement = %v, want [1.0.1 1.0.2]", versions)
	}
}

func TestRetireOlderArtifactRetention(t *testing.T) {
	store, data, models := applyFixture(t)
	importNewVersion(t, store)
	writeManagedModel(t, models, "example/new", "rev-new")

	results, err := RetireOlder(store, data, models, "qwen-test", "1.0.1")
	if err != nil {
		t.Fatalf("RetireOlder() error = %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("retired results = %d", len(results))
	}
	sharedModel := filepath.Join(models, modelstore.DestinationName("example/shared"))
	if _, err := os.Stat(sharedModel); err != nil {
		t.Fatalf("model shared by old and new versions was removed: %v", err)
	}
	oldOnlyModel := filepath.Join(models, modelstore.DestinationName("example/old"))
	if _, err := os.Stat(oldOnlyModel); !os.IsNotExist(err) {
		t.Fatalf("old-only Controller-owned model was retained: %v", err)
	}
	outcomes := ""
	for _, result := range results {
		for _, item := range result.Items {
			outcomes += item.Artifact + "=" + item.Outcome + "\n"
		}
	}
	for _, want := range []string{
		"Shared=Retained — used by another installed pack",
		"Old Only=Removed",
		"Runtime image=Retained — used by another installed pack",
	} {
		if !strings.Contains(outcomes, want) {
			t.Fatalf("retirement outcomes missing %q:\n%s", want, outcomes)
		}
	}
}

func TestRetireOlderRetainsExternalAndUnownedArtifacts(t *testing.T) {
	store := t.TempDir()
	data := t.TempDir()
	models := t.TempDir()
	// Old-only model in the external Hugging Face cache layout, and an
	// old-only runtime digest with no b70ctl ownership record: neither may
	// be deleted.
	unownedDigest := "sha256:" + strings.Repeat("b", 64)
	oldManifest := packManifest("qwen-test", "Qwen Test", "1.0.0", []modelpack.Model{
		packModel("external", "External", "example/external", "rev-external"),
	}, unownedDigest)
	importPack(t, store, oldManifest)
	external := writeExternalModel(t, models, "example/external", "rev-external")
	importNewVersion(t, store)
	installDockerStub(t, testDigest, "", "")

	results, err := RetireOlder(store, data, models, "qwen-test", "1.0.1")
	if err != nil {
		t.Fatalf("RetireOlder() error = %v", err)
	}
	if _, err := os.Stat(external); err != nil {
		t.Fatalf("external Hugging Face cache model was removed: %v", err)
	}
	outcomes := ""
	for _, result := range results {
		for _, item := range result.Items {
			outcomes += item.Artifact + "=" + item.Outcome + "\n"
		}
	}
	if !strings.Contains(outcomes, "Retained — external Hugging Face cache") {
		t.Fatalf("external model retention not reported:\n%s", outcomes)
	}
	if !strings.Contains(outcomes, "Retained — it was not acquired by b70ctl") {
		t.Fatalf("unowned runtime retention not reported:\n%s", outcomes)
	}
}

func TestRetireOlderDualStateWithoutRedownload(t *testing.T) {
	store, data, models := applyFixture(t)
	importNewVersion(t, store)
	newPackJSON, err := os.ReadFile(filepath.Join(store, "qwen-test", "1.0.1", "pack.json"))
	if err != nil {
		t.Fatal(err)
	}

	results, err := RetireOlder(store, data, models, "qwen-test", "1.0.1")
	if err != nil {
		t.Fatalf("RetireOlder() error = %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("retired results = %d", len(results))
	}
	versions := storeVersions(t, store, "qwen-test")
	if len(versions) != 1 || versions[0] != "1.0.1" {
		t.Fatalf("installed versions after cleanup = %v, want [1.0.1]", versions)
	}
	after, err := os.ReadFile(filepath.Join(store, "qwen-test", "1.0.1", "pack.json"))
	if err != nil {
		t.Fatal(err)
	}
	if string(after) != string(newPackJSON) {
		t.Fatal("kept pack version content changed during cleanup")
	}
}

func TestRetireOlderRefusesWhileOldVersionServes(t *testing.T) {
	store, data, models := applyFixture(t)
	importNewVersion(t, store)
	installDockerStub(t, testDigest, "qwen-test", "1.0.0")

	if _, err := RetireOlder(store, data, models, "qwen-test", "1.0.1"); err == nil || !strings.Contains(err.Error(), "Stop this model") {
		t.Fatalf("RetireOlder() error = %v, want running-model refusal", err)
	}
	versions := storeVersions(t, store, "qwen-test")
	if len(versions) != 2 {
		t.Fatalf("refused retirement still changed the store: %v", versions)
	}
}

func TestRetireOlderLeavesIncomparableVersionsInstalled(t *testing.T) {
	store, data, models := applyFixture(t)
	importPack(t, store, packManifest("qwen-test", "Qwen Test", "1.0.0-beta", []modelpack.Model{
		packModel("shared", "Shared", "example/shared", "rev-shared"),
	}, testDigest))
	importNewVersion(t, store)

	_, err := RetireOlder(store, data, models, "qwen-test", "1.0.1")
	if err == nil || !strings.Contains(err.Error(), "cannot be compared") {
		t.Fatalf("RetireOlder() error = %v, want incomparable-version refusal", err)
	}
	versions := storeVersions(t, store, "qwen-test")
	if len(versions) != 3 {
		t.Fatalf("incomparable version was not left installed: %v", versions)
	}
}
