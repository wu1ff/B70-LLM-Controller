package modelpack

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"testing"
)

// Compare the pack's v6 launch with the retained qualified vLLM 0.33 profile.
func TestFlashNextDiskQualifiedProfiles(t *testing.T) {
	root := filepath.Join("..", "..", "model-packs", "Qwen3.8-Flash-Next")
	manifest, err := Load(root)
	if err != nil {
		t.Fatal(err)
	}
	const image = "ghcr.io/wu1ff/qwen38-flashnext-b70"
	const digest = "sha256:2fa91d1ac8c17ad866d9ae7eaea83ac1b1666ffd0ccceb7685564f55528c47af"
	if manifest.Version != "1.0.6" || len(manifest.Runtimes) != 1 || len(manifest.Modes) != 1 || len(manifest.Profiles) != 1 {
		t.Fatal("DFlash2 v6 must retain one runtime, mode and qualified profile")
	}
	runtime := manifest.Runtimes[0]
	if runtime.Digest != digest || runtime.Registry != image+"@"+digest || runtime.Image != image+":dflash2-v033" {
		t.Fatalf("unexpected v6 runtime authority: %#v", runtime)
	}
	if len(manifest.Models) != 2 || manifest.Models[0].Revision != "4c67bf686b7f7fd386bae6b07ab59e8ff1d5b897" || manifest.Models[1].Revision != "a22a1744e85c17364dcc54a6a420af7135d4c7f6" {
		t.Fatal("qualified target or BF16 assistant revision changed")
	}
	profile := manifest.Profiles[0]
	if profile.ID != "w4a16-dflash2-tp4-262144" || profile.Mode != "dflash2" || profile.Cards != 4 || profile.TensorParallel != 4 || profile.Context != 262144 {
		t.Fatalf("unexpected v6 profile: %#v", profile)
	}

	var authority struct {
		Environment map[string]string `json:"env"`
		Command     []string          `json:"command"`
	}
	data, err := os.ReadFile(filepath.Join(root, "patches", "production", "runtime", "profile.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(data, &authority); err != nil {
		t.Fatal(err)
	}
	// CONTROLLER_COMPATIBILITY.json records these production omissions.
	diagnostic := slices.Index(authority.Command, "--worker-extension-cls")
	if diagnostic < 0 || diagnostic+1 >= len(authority.Command) || authority.Command[diagnostic+1] != "parity_worker.ParityWorker" {
		t.Fatal("unexpected worker extension in the frozen profile")
	}
	wantCommand := slices.Delete(authority.Command, diagnostic, diagnostic+2)
	delete(authority.Environment, "PYTHONPATH")
	delete(authority.Environment, "VLLM_SERVER_DEV_MODE")

	resolved, err := Resolve(manifest, profile)
	if err != nil {
		t.Fatal(err)
	}
	serve, err := ServeCommand(resolved, "/models")
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(serve, wantCommand) {
		t.Fatalf("production command = %q, want %q", serve, wantCommand)
	}
	if !reflect.DeepEqual(resolved.Environment, authority.Environment) {
		t.Fatalf("production environment = %v, want %v", resolved.Environment, authority.Environment)
	}

	wantDockerArgs := []string{"--pull", "never", "--device=/dev/dri:/dev/dri", "--ipc=private", "--shm-size=32g", "--security-opt", "seccomp=unconfined"}
	if !slices.Equal(resolved.DockerArgs, wantDockerArgs) || resolved.ContainerPort != 8000 || resolved.HealthPath != "/health" ||
		resolved.ImageEntrypoint || resolved.PersistentJITCache {
		t.Fatal("v6 Docker arguments, listener, health check or runtime options changed")
	}
	wantMounts := []Mount{
		{ModelID: "qwen38-flashnext-w4a16", ContainerPath: "/models"},
		{ModelID: "qwen38-flashnext-dflash2-bf16", ContainerPath: "/draft"},
	}
	if !slices.Equal(resolved.Mounts, wantMounts) || resolved.Environment["TMPDIR"] != "/work/tmp" {
		t.Fatal("v6 checkpoint mounts or writable work directory contract changed")
	}

	speculationIndex := slices.Index(serve, "--speculative-config")
	if speculationIndex < 0 || speculationIndex+1 >= len(serve) {
		t.Fatal("missing speculative configuration")
	}
	var speculation struct {
		Method    string `json:"method"`
		Proposals int    `json:"num_speculative_tokens"`
		Rejection string `json:"rejection_sample_method"`
	}
	if err := json.Unmarshal([]byte(serve[speculationIndex+1]), &speculation); err != nil {
		t.Fatal(err)
	}
	if speculation.Method != "dflash" || speculation.Proposals != 7 || speculation.Rejection != "standard" {
		t.Fatalf("unexpected speculative configuration: %+v", speculation)
	}
}
