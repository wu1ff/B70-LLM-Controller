package install

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"b70ctl/internal/catalog"
	"b70ctl/internal/hf"
	"b70ctl/internal/modelpack"
	"b70ctl/internal/modelstore"
	"b70ctl/internal/packstore"
	runtimebackend "b70ctl/internal/runtime"
)

const (
	publicRevision    = "1111111111111111111111111111111111111111"
	gatedRevision     = "2222222222222222222222222222222222222222"
	assistantRevision = "3333333333333333333333333333333333333333"
)

func TestBuildPlanUsesSelectedTargetsAndTheirAssistants(t *testing.T) {
	manifest := syntheticManifest()
	targets := Targets(manifest)
	if got := modelIDs(targets); !reflect.DeepEqual(got, []string{"target-public", "target-gated", "target-unrelated"}) {
		t.Fatalf("Targets() = %v", got)
	}

	root := t.TempDir()
	public := filepath.Join(root, "public")
	shared := filepath.Join(root, "shared")
	writeMarker(t, public, "example/public", publicRevision)
	writeMarker(t, shared, "example/shared", assistantRevision)
	inventory := []modelstore.Artifact{
		{Repo: "example/public", Revision: publicRevision, Path: public},
		{Repo: "example/shared", Revision: assistantRevision, Path: shared},
	}
	plan, err := BuildPlan(manifest, []string{"target-public", "target-gated"}, inventory)
	if err != nil {
		t.Fatal(err)
	}
	if got := artifactIDs(plan.Artifacts); !reflect.DeepEqual(got, []string{"target-public", "assistant-shared", "target-gated"}) {
		t.Fatalf("planned artifacts = %v", got)
	}
	if plan.Artifacts[0].State != Present || plan.Artifacts[1].State != Present || plan.Artifacts[2].State != Missing {
		t.Fatalf("planned states = %#v", plan.Artifacts)
	}
	if got := plan.Counts(); got != (Counts{SelectedTargets: 2, Existing: 2, Downloads: 1}) {
		t.Fatalf("Counts() = %#v", got)
	}
}

func TestBuildPlanAllowsNoTargetsAndRejectsAssistantSelection(t *testing.T) {
	manifest := syntheticManifest()
	plan, err := BuildPlan(manifest, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(plan.SelectedTargetIDs) != 0 || len(plan.Artifacts) != 0 || plan.Counts() != (Counts{}) {
		t.Fatalf("empty plan = %#v", plan)
	}
	if _, err := BuildPlan(manifest, []string{"assistant-shared"}, nil); err == nil {
		t.Fatal("BuildPlan accepted an assistant selection")
	}
}

func TestBuildPlanDeduplicatesSharedAssistantAndExcludesUnrelatedModels(t *testing.T) {
	plan, err := BuildPlan(syntheticManifest(), []string{"target-public", "target-gated"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if got := artifactIDs(plan.Artifacts); !reflect.DeepEqual(got, []string{"target-public", "assistant-shared", "target-gated"}) {
		t.Fatalf("planned artifacts = %v", got)
	}
	for _, artifact := range plan.Artifacts {
		if artifact.ModelID == "target-unrelated" || artifact.ModelID == "assistant-unrelated" {
			t.Fatalf("unrelated artifact included: %#v", artifact)
		}
	}
	if len(plan.Runtimes) != 1 || plan.Runtimes[0].Digest != syntheticManifest().Runtimes[0].Digest {
		t.Fatalf("deduplicated runtimes = %#v", plan.Runtimes)
	}
}

func TestBuildPlanUsesModeAssistants(t *testing.T) {
	manifest := syntheticManifest()
	empty := modelpack.LaunchBlock{DockerArgs: []string{}, Environment: map[string]string{}, CommandArgs: []string{}}
	manifest.Modes = append(manifest.Modes,
		modelpack.Mode{ID: "base", DisplayName: "Base", Assistants: []string{}, Launch: empty},
		modelpack.Mode{ID: "mtp", DisplayName: "MTP1", Assistants: []string{}, Launch: empty},
		modelpack.Mode{ID: "dflash2", DisplayName: "dFlash2", Assistants: []string{"assistant-shared"}, Launch: empty},
	)
	manifest.Profiles = []modelpack.Profile{
		{ID: "base", ModelID: "target-public", RuntimeID: "runtime", Cards: 1, TensorParallel: 1, Context: 4096, Mode: "base", Launch: empty},
		{ID: "mtp", ModelID: "target-public", RuntimeID: "runtime", Cards: 1, TensorParallel: 1, Context: 8192, Mode: "mtp", Launch: empty},
		{ID: "dflash2", ModelID: "target-public", RuntimeID: "runtime", Cards: 1, TensorParallel: 1, Context: 16384, Mode: "dflash2", Launch: empty},
	}
	plan, err := BuildPlan(manifest, []string{"target-public"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if got := artifactIDs(plan.Artifacts); !reflect.DeepEqual(got, []string{"target-public", "assistant-shared"}) {
		t.Fatalf("mode-derived artifacts = %v", got)
	}
	manifest.Profiles = manifest.Profiles[:2]
	plan, err = BuildPlan(manifest, []string{"target-public"}, nil)
	if err != nil || !reflect.DeepEqual(artifactIDs(plan.Artifacts), []string{"target-public"}) {
		t.Fatalf("Base/MTP plan = %#v, %v", plan, err)
	}
}

func TestMissingGatedArtifactsCanBeSkipped(t *testing.T) {
	plan, err := BuildPlan(syntheticManifest(), []string{"target-gated"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !plan.HasMissingGated() {
		t.Fatal("missing gated target was not detected")
	}
	plan.SkipMissingGated("Hugging Face token not configured")
	if plan.HasMissingGated() || plan.Counts() != (Counts{SelectedTargets: 1, Downloads: 1, Skipped: 1}) {
		t.Fatalf("skipped plan = %#v, counts %#v", plan, plan.Counts())
	}
	if plan.Artifacts[0].SkipReason == "" {
		t.Fatal("gated target was not skipped")
	}
}

func TestPrepareContinuesAfterAccessFailureAndKeepsSuccesses(t *testing.T) {
	manifest := syntheticManifest()
	source := writePack(t, manifest)
	store := t.TempDir()
	models := t.TempDir()
	plan, err := BuildPlan(manifest, []string{"target-public", "target-gated"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	plan.Artifacts = []Artifact{plan.Artifacts[0], plan.Artifacts[2], plan.Artifacts[1]}

	var requested []string
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		repo := request.URL.Query().Get("repo")
		requested = append(requested, request.URL.Path+":"+repo)
		if repo == "example/gated" {
			writer.WriteHeader(http.StatusForbidden)
			return
		}
		fmt.Fprint(writer, "model")
	}))
	defer server.Close()

	runtimeCalls := 0
	ops := operations{
		inspect: func(ctx context.Context, repo, revision string) (hf.Repository, error) {
			request, _ := http.NewRequestWithContext(ctx, http.MethodGet, server.URL+"/inspect?repo="+repo, nil)
			response, err := http.DefaultClient.Do(request)
			if err != nil {
				return hf.Repository{}, err
			}
			response.Body.Close()
			if response.StatusCode == http.StatusForbidden {
				return hf.Repository{}, hf.ErrAccessDenied
			}
			return hf.Repository{Repo: repo, Revision: revision, Files: []hf.File{{Path: "model.bin", Size: 5, SizeKnown: true}}}, nil
		},
		install: func(ctx context.Context, root string, repository hf.Repository, progress func(hf.Progress)) (hf.Result, error) {
			request, _ := http.NewRequestWithContext(ctx, http.MethodGet, server.URL+"/download?repo="+repository.Repo, nil)
			response, err := http.DefaultClient.Do(request)
			if err != nil {
				return hf.Result{}, err
			}
			response.Body.Close()
			if response.StatusCode != http.StatusOK {
				return hf.Result{}, hf.ErrDownloadFailed
			}
			progress(hf.Progress{CurrentFile: "model.bin", FileNumber: 1, TotalFiles: 1, BytesDownloaded: 5, TotalBytes: 5, TotalKnown: true})
			path := filepath.Join(root, strings.ReplaceAll(repository.Repo, "/", "__"))
			writeMarker(t, path, repository.Repo, repository.Revision)
			return hf.Result{Path: path, BytesDownloaded: 5}, nil
		},
		acquire: func(context.Context, modelpack.Runtime, func(runtimebackend.PullProgress)) runtimebackend.AcquisitionResult {
			runtimeCalls++
			return runtimebackend.AcquisitionResult{Outcome: runtimebackend.AcquisitionFailed, Reason: "Docker pull failed", Err: errors.New("pull failed")}
		},
	}

	result, err := prepare(context.Background(), source, store, models, plan, ops, packstore.SourceLocal, nil)
	if err != nil {
		t.Fatal(err)
	}
	if got := outcomes(result.Items); !reflect.DeepEqual(got, []Outcome{Downloaded, Failed, Downloaded}) {
		t.Fatalf("outcomes = %v", got)
	}
	if !errors.Is(result.Items[1].Err, hf.ErrAccessDenied) || !result.HasAccessIssue() {
		t.Fatalf("access failure was not retained: %#v", result.Items[1])
	}
	if runtimeCalls != 1 || len(result.RuntimeItems) != 1 || !result.HasRuntimeFailure() {
		t.Fatalf("runtime preparation = %#v, calls = %d", result.RuntimeItems, runtimeCalls)
	}
	if got := requested; !reflect.DeepEqual(got, []string{
		"/inspect:example/public", "/download:example/public",
		"/inspect:example/gated",
		"/inspect:example/shared", "/download:example/shared",
	}) {
		t.Fatalf("requests = %v", got)
	}
	for _, identity := range [][2]string{{"example/public", publicRevision}, {"example/shared", assistantRevision}} {
		if _, found := modelstore.Find(result.Inventory, identity[0], identity[1]); !found {
			t.Fatalf("successful artifact %v missing from final scan: %#v", identity, result.Inventory)
		}
	}
	installed, err := packstore.List(store)
	if err != nil || len(installed) != 1 {
		t.Fatalf("installed pack after partial failure = %#v, %v", installed, err)
	}
}

func TestPrepareReusesExactArtifactAndHonorsSkippedArtifact(t *testing.T) {
	manifest := syntheticManifest()
	source := writePack(t, manifest)
	store := t.TempDir()
	models := t.TempDir()
	writeMarker(t, filepath.Join(models, "existing"), "example/shared", assistantRevision)
	plan, err := BuildPlan(manifest, []string{"target-gated"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	plan.SkipMissingGated("Hugging Face token not configured")
	called := false
	ops := operations{
		inspect: func(context.Context, string, string) (hf.Repository, error) {
			called = true
			return hf.Repository{}, errors.New("unexpected inspect")
		},
		install: func(context.Context, string, hf.Repository, func(hf.Progress)) (hf.Result, error) {
			called = true
			return hf.Result{}, errors.New("unexpected install")
		},
		acquire: func(context.Context, modelpack.Runtime, func(runtimebackend.PullProgress)) runtimebackend.AcquisitionResult {
			return runtimebackend.AcquisitionResult{Outcome: runtimebackend.AcquisitionReused}
		},
	}
	result, err := prepare(context.Background(), source, store, models, plan, ops, packstore.SourceLocal, nil)
	if err != nil {
		t.Fatal(err)
	}
	if called || !reflect.DeepEqual(outcomes(result.Items), []Outcome{Skipped, Reused}) {
		t.Fatalf("result = %#v, downloader called = %v", result, called)
	}
	if len(result.RuntimeItems) != 1 || result.RuntimeItems[0].Outcome != runtimebackend.AcquisitionReused {
		t.Fatalf("runtime result = %#v", result.RuntimeItems)
	}
}

func TestPrepareFailsIncompleteArtifactAndContinues(t *testing.T) {
	manifest := syntheticManifest()
	source := writePack(t, manifest)
	store := t.TempDir()
	models := t.TempDir()
	incomplete := filepath.Join(models, "incomplete")
	if err := os.MkdirAll(incomplete, 0o755); err != nil {
		t.Fatal(err)
	}
	writeJSON(t, filepath.Join(incomplete, ".b70-model.json"), map[string]string{"repo": "example/public", "revision": publicRevision})
	writeMarker(t, filepath.Join(models, "assistant"), "example/shared", assistantRevision)
	inventory, err := modelstore.Scan(models)
	if err != nil {
		t.Fatal(err)
	}
	plan, err := BuildPlan(manifest, []string{"target-public"}, inventory)
	if err != nil {
		t.Fatal(err)
	}
	if plan.Artifacts[0].State != Incomplete || plan.Artifacts[1].State != Present || plan.Counts() != (Counts{SelectedTargets: 1, Existing: 1, Incomplete: 1}) {
		t.Fatalf("plan = %#v, counts = %#v", plan, plan.Counts())
	}
	called := false
	result, err := prepare(context.Background(), source, store, models, plan, operations{
		inspect: func(context.Context, string, string) (hf.Repository, error) {
			called = true
			return hf.Repository{}, errors.New("unexpected inspect")
		},
		install: func(context.Context, string, hf.Repository, func(hf.Progress)) (hf.Result, error) {
			called = true
			return hf.Result{}, errors.New("unexpected install")
		},
		acquire: func(context.Context, modelpack.Runtime, func(runtimebackend.PullProgress)) runtimebackend.AcquisitionResult {
			return runtimebackend.AcquisitionResult{Outcome: runtimebackend.AcquisitionReused}
		},
	}, packstore.SourceLocal, nil)
	if err != nil {
		t.Fatal(err)
	}
	if called || !reflect.DeepEqual(outcomes(result.Items), []Outcome{Failed, Reused}) || result.Items[0].Reason != "local model is incomplete" {
		t.Fatalf("result = %#v, downloader called = %v", result, called)
	}
	installed, err := packstore.List(store)
	if err != nil || len(installed) != 1 {
		t.Fatalf("installed pack after incomplete model = %#v, %v", installed, err)
	}
}

func TestPrepareRequiresDownloadedPackCompleteness(t *testing.T) {
	manifest := syntheticManifest()
	source := writePack(t, manifest)
	plan, err := BuildPlan(manifest, []string{"target-public"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	plan.Artifacts = plan.Artifacts[:1]
	models := t.TempDir()
	result, err := prepare(context.Background(), source, t.TempDir(), models, plan, operations{
		inspect: func(context.Context, string, string) (hf.Repository, error) {
			return hf.Repository{Repo: "example/public", Revision: publicRevision, Files: []hf.File{{Path: "model.bin"}}}, nil
		},
		install: func(_ context.Context, root string, repository hf.Repository, _ func(hf.Progress)) (hf.Result, error) {
			path := filepath.Join(root, "downloaded")
			if err := os.MkdirAll(path, 0o755); err != nil {
				return hf.Result{}, err
			}
			writeJSON(t, filepath.Join(path, ".b70-model.json"), map[string]string{"repo": repository.Repo, "revision": repository.Revision})
			return hf.Result{Path: path}, nil
		},
		acquire: func(context.Context, modelpack.Runtime, func(runtimebackend.PullProgress)) runtimebackend.AcquisitionResult {
			return runtimebackend.AcquisitionResult{Outcome: runtimebackend.AcquisitionReused}
		},
	}, packstore.SourceLocal, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Items) != 1 || result.Items[0].Outcome != Failed || result.Items[0].Reason != "downloaded model is incomplete" {
		t.Fatalf("result = %#v", result)
	}
}

func TestPrepareRefusesDuplicateInstalledPack(t *testing.T) {
	manifest := syntheticManifest()
	source := writePack(t, manifest)
	store := t.TempDir()
	if _, err := packstore.Import(source, store, packstore.SourceLocal); err != nil {
		t.Fatal(err)
	}
	_, err := prepare(context.Background(), source, store, t.TempDir(), Plan{}, operations{}, packstore.SourceLocal, nil)
	if err == nil || err.Error() != "pack synthetic-pack 1.0.0 is already installed" {
		t.Fatalf("prepare duplicate error = %v", err)
	}
}

func TestRemoteCatalogAcquisitionUsesExistingPreparationFlow(t *testing.T) {
	manifest := syntheticManifest()
	source := writePack(t, manifest)
	archive := packArchive(t, source)
	hash := sha256.Sum256(archive)
	var server *httptest.Server
	server = httptest.NewTLSServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		switch request.URL.Path {
		case "/index.json":
			fmt.Fprintf(writer, `{"schema_version":1,"packs":[{"id":%q,"name":%q,"version":%q,"archive_url":%q,"sha256":%q}]}`,
				manifest.ID, manifest.Name, manifest.Version, server.URL+"/pack.tar.gz", "sha256:"+hex.EncodeToString(hash[:]))
		case "/pack.tar.gz":
			writer.Write(archive)
		default:
			writer.WriteHeader(http.StatusNotFound)
		}
	}))
	defer server.Close()
	client := &catalog.Client{URL: server.URL + "/index.json", HTTPClient: server.Client()}
	dataRoot := t.TempDir()
	remote, err := client.Refresh(context.Background(), dataRoot)
	if err != nil || remote.Cached || len(remote.Catalog.Packs) != 1 {
		t.Fatalf("Refresh() = %#v, %v", remote, err)
	}
	acquired, err := client.Acquire(context.Background(), dataRoot, remote.Catalog.Packs[0])
	if err != nil {
		t.Fatal(err)
	}
	defer acquired.Close()

	models := t.TempDir()
	writeMarker(t, filepath.Join(models, "target"), "example/public", publicRevision)
	writeMarker(t, filepath.Join(models, "assistant"), "example/shared", assistantRevision)
	inventory, err := modelstore.Scan(models)
	if err != nil {
		t.Fatal(err)
	}
	plan, err := BuildPlan(acquired.Manifest, []string{"target-public"}, inventory)
	if err != nil {
		t.Fatal(err)
	}
	runtimeCalls := 0
	ops := operations{
		inspect: func(context.Context, string, string) (hf.Repository, error) {
			return hf.Repository{}, errors.New("existing model was inspected")
		},
		install: func(context.Context, string, hf.Repository, func(hf.Progress)) (hf.Result, error) {
			return hf.Result{}, errors.New("existing model was downloaded")
		},
		acquire: func(context.Context, modelpack.Runtime, func(runtimebackend.PullProgress)) runtimebackend.AcquisitionResult {
			runtimeCalls++
			return runtimebackend.AcquisitionResult{Outcome: runtimebackend.AcquisitionReused}
		},
	}
	store := t.TempDir()
	result, err := prepare(context.Background(), acquired.Path, store, models, plan, ops, packstore.SourceRemote, nil)
	if err != nil {
		t.Fatal(err)
	}
	if result.Pack.Source != packstore.SourceRemote || !reflect.DeepEqual(outcomes(result.Items), []Outcome{Reused, Reused}) || runtimeCalls != 1 {
		t.Fatalf("remote preparation = %#v, runtime calls = %d", result, runtimeCalls)
	}
	if _, err := prepare(context.Background(), acquired.Path, store, models, plan, ops, packstore.SourceRemote, nil); err == nil {
		t.Fatal("second exact remote install was accepted")
	}

	localResult, err := prepare(context.Background(), source, t.TempDir(), t.TempDir(), Plan{}, operations{}, packstore.SourceLocal, nil)
	if err != nil {
		t.Fatal(err)
	}
	if localResult.Pack.Source != packstore.SourceLocal {
		t.Fatalf("local source = %q", localResult.Pack.Source)
	}
}

func syntheticManifest() *modelpack.Manifest {
	empty := modelpack.LaunchBlock{DockerArgs: []string{}, Environment: map[string]string{}, CommandArgs: []string{}}
	return &modelpack.Manifest{
		SchemaVersion: 1, ID: "synthetic-pack", Name: "Synthetic Pack", Version: "1.0.0",
		Models: []modelpack.Model{
			{ID: "target-public", Name: "Public Target", Kind: "target", Repo: "example/public", Revision: publicRevision, Files: []modelpack.ModelFile{{Path: "model.bin"}}, MountPath: "/models/public", Launch: empty},
			{ID: "target-gated", Name: "Gated Target", Kind: "target", Repo: "example/gated", Revision: gatedRevision, Gated: true, Files: []modelpack.ModelFile{{Path: "model.bin"}}, MountPath: "/models/gated", Launch: empty},
			{ID: "target-unrelated", Name: "Unrelated Target", Kind: "target", Repo: "example/unrelated", Revision: publicRevision, Files: []modelpack.ModelFile{{Path: "model.bin"}}, MountPath: "/models/unrelated", Launch: empty},
			{ID: "assistant-shared", Name: "Shared Assistant", Kind: "assistant", Repo: "example/shared", Revision: assistantRevision, Files: []modelpack.ModelFile{{Path: "model.bin"}}, MountPath: "/models/shared", Launch: empty},
			{ID: "assistant-duplicate", Name: "Duplicate Assistant", Kind: "assistant", Repo: "example/shared", Revision: assistantRevision, Files: []modelpack.ModelFile{{Path: "model.bin"}}, MountPath: "/models/duplicate", Launch: empty},
			{ID: "assistant-unrelated", Name: "Unrelated Assistant", Kind: "assistant", Repo: "example/unused", Revision: assistantRevision, Files: []modelpack.ModelFile{{Path: "model.bin"}}, MountPath: "/models/unused", Launch: empty},
		},
		Runtimes: []modelpack.Runtime{{ID: "runtime", Image: "example/runtime:1", Digest: "sha256:" + strings.Repeat("a", 64), ContainerPort: 8000, HealthPath: "/health", Launch: modelpack.RuntimeLaunch{DockerArgs: []string{}, Environment: map[string]string{}, Command: []string{"serve"}}}},
		Modes: []modelpack.Mode{
			{ID: "shared", DisplayName: "Shared", Assistants: []string{"assistant-shared"}, Launch: empty},
			{ID: "duplicate", DisplayName: "Duplicate", Assistants: []string{"assistant-duplicate"}, Launch: empty},
			{ID: "unrelated", DisplayName: "Unrelated", Assistants: []string{"assistant-unrelated"}, Launch: empty},
		},
		Profiles: []modelpack.Profile{
			{ID: "public", ModelID: "target-public", RuntimeID: "runtime", Cards: 1, TensorParallel: 1, Context: 4096, Mode: "shared", Launch: empty},
			{ID: "public-dedup", ModelID: "target-public", RuntimeID: "runtime", Cards: 2, TensorParallel: 2, Context: 4096, Mode: "duplicate", Launch: empty},
			{ID: "gated", ModelID: "target-gated", RuntimeID: "runtime", Cards: 1, TensorParallel: 1, Context: 8192, Mode: "shared", Launch: empty},
			{ID: "unrelated", ModelID: "target-unrelated", RuntimeID: "runtime", Cards: 1, TensorParallel: 1, Context: 16384, Mode: "unrelated", Launch: empty},
		},
	}
}

func writePack(t *testing.T, manifest *modelpack.Manifest) string {
	t.Helper()
	directory := t.TempDir()
	if err := os.WriteFile(filepath.Join(directory, "README.md"), []byte("# Synthetic\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	writeJSON(t, filepath.Join(directory, "pack.json"), manifest)
	return directory
}

func packArchive(t *testing.T, root string) []byte {
	t.Helper()
	var buffer bytes.Buffer
	compressed := gzip.NewWriter(&buffer)
	archive := tar.NewWriter(compressed)
	err := filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if info.IsDir() {
			return nil
		}
		relative, err := filepath.Rel(root, path)
		if err != nil {
			return err
		}
		data, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		if err := archive.WriteHeader(&tar.Header{Name: filepath.ToSlash(relative), Typeflag: tar.TypeReg, Mode: 0o644, Size: int64(len(data))}); err != nil {
			return err
		}
		_, err = archive.Write(data)
		return err
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := archive.Close(); err != nil {
		t.Fatal(err)
	}
	if err := compressed.Close(); err != nil {
		t.Fatal(err)
	}
	return buffer.Bytes()
}

func writeMarker(t *testing.T, directory, repo, revision string) {
	t.Helper()
	if err := os.MkdirAll(directory, 0o755); err != nil {
		t.Fatal(err)
	}
	writeJSON(t, filepath.Join(directory, ".b70-model.json"), map[string]string{"repo": repo, "revision": revision})
	if err := os.WriteFile(filepath.Join(directory, "model.bin"), []byte("model"), 0o644); err != nil {
		t.Fatal(err)
	}
}

func writeJSON(t *testing.T, path string, value any) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	data, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}
}

func modelIDs(models []modelpack.Model) []string {
	ids := make([]string, len(models))
	for index, model := range models {
		ids[index] = model.ID
	}
	return ids
}

func artifactIDs(artifacts []Artifact) []string {
	ids := make([]string, len(artifacts))
	for index, artifact := range artifacts {
		ids[index] = artifact.ModelID
	}
	return ids
}

func outcomes(items []Item) []Outcome {
	values := make([]Outcome, len(items))
	for index, item := range items {
		values[index] = item.Outcome
	}
	return values
}
