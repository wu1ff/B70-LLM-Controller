package runtime

import (
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"b70ctl/internal/modelpack"
	"b70ctl/internal/modelstore"
)

func TestBuildLaunchComposesUnifiedContractDeterministically(t *testing.T) {
	manifest := launchManifest()
	models := launchModels(t)
	launch, err := BuildLaunch(manifest, manifest.Profiles[0], models, AccessLocal, 8123)
	if err != nil {
		t.Fatal(err)
	}
	want := []string{
		"run", "-d", "--name", ManagedContainer,
		"--publish", "127.0.0.1:8123:8000",
		"--label", "b70ctl.managed=true",
		"--label", "b70ctl.pack_id=qwen-test",
		"--label", "b70ctl.pack_version=1.2.3",
		"--label", "b70ctl.profile_id=dflash-4",
		"--label", "b70ctl.health_path=/health",
		"--runtime-a", "--runtime-b", "--model", "--mode", "--profile-a", "--profile-b",
		"--mount", "type=bind,source=" + models[1].Path + ",target=/models/target,readonly",
		"--mount", "type=bind,source=" + models[0].Path + ",target=/models/draft,readonly",
		"--env", "ALPHA=first", "--env", "MODE=dflash", "--env", "ZETA=last",
		"sha256:" + strings.Repeat("a", 64),
		"vllm", "serve", "/models/target", "--speculative-model", "/models/draft", "--tp", "4",
	}
	if !reflect.DeepEqual(launch.DockerArgs, want) {
		t.Fatalf("BuildLaunch() args = %#v\nwant = %#v", launch.DockerArgs, want)
	}
	again, err := BuildLaunch(manifest, manifest.Profiles[0], models, AccessLocal, 8123)
	if err != nil || !reflect.DeepEqual(again, launch) {
		t.Fatalf("second BuildLaunch() = %#v, %v", again, err)
	}
}

func TestBuildLaunchLANAndIncompleteModel(t *testing.T) {
	manifest := launchManifest()
	models := launchModels(t)
	launch, err := BuildLaunch(manifest, manifest.Profiles[0], models, AccessLAN, 9000)
	if err != nil || launch.BindAddress != "0.0.0.0" || launch.DockerArgs[5] != "0.0.0.0:9000:8000" {
		t.Fatalf("BuildLaunch() = %#v, %v", launch, err)
	}
	if err := os.Remove(filepath.Join(models[1].Path, "model.bin")); err != nil {
		t.Fatal(err)
	}
	if _, err := BuildLaunch(manifest, manifest.Profiles[0], models, AccessLocal, 8123); err == nil || err.Error() != `model "target" is incomplete` {
		t.Fatalf("BuildLaunch() error = %v", err)
	}
}

func launchManifest() *modelpack.Manifest {
	empty := func() modelpack.LaunchBlock {
		return modelpack.LaunchBlock{DockerArgs: []string{}, Environment: map[string]string{}, CommandArgs: []string{}}
	}
	return &modelpack.Manifest{
		ID: "qwen-test", Version: "1.2.3",
		Models: []modelpack.Model{
			{ID: "target", Kind: "target", Repo: "example/target", Revision: "target-rev", Files: []modelpack.ModelFile{{Path: "model.bin"}}, MountPath: "/models/target", Launch: modelpack.LaunchBlock{DockerArgs: []string{"--model"}, Environment: map[string]string{}, CommandArgs: []string{"/models/target"}}},
			{ID: "draft", Kind: "assistant", Repo: "example/draft", Revision: "draft-rev", Files: []modelpack.ModelFile{{Path: "model.bin"}}, MountPath: "/models/draft", Launch: empty()},
		},
		Runtimes: []modelpack.Runtime{{ID: "runtime", Image: "mutable:tag", Digest: "sha256:" + strings.Repeat("a", 64), ContainerPort: 8000, HealthPath: "/health", Launch: modelpack.RuntimeLaunch{DockerArgs: []string{"--runtime-a", "--runtime-b"}, Environment: map[string]string{"ZETA": "last", "ALPHA": "first"}, Command: []string{"vllm", "serve"}}}},
		Modes:    []modelpack.Mode{{ID: "dflash2", DisplayName: "dFlash2", Assistants: []string{"draft"}, Launch: modelpack.LaunchBlock{DockerArgs: []string{"--mode"}, Environment: map[string]string{"MODE": "dflash"}, CommandArgs: []string{"--speculative-model", "/models/draft"}}}},
		Profiles: []modelpack.Profile{{ID: "dflash-4", ModelID: "target", RuntimeID: "runtime", Cards: 4, TensorParallel: 4, Context: 4096, Mode: "dflash2", Launch: modelpack.LaunchBlock{DockerArgs: []string{"--profile-a", "--profile-b"}, Environment: map[string]string{}, CommandArgs: []string{"--tp", "4"}}}},
	}
}

func launchModels(t *testing.T) []modelstore.Artifact {
	t.Helper()
	root := t.TempDir()
	draft, target := filepath.Join(root, "draft"), filepath.Join(root, "target")
	for _, path := range []string{draft, target} {
		if err := os.MkdirAll(path, 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(path, "model.bin"), []byte("model"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return []modelstore.Artifact{{Repo: "example/draft", Revision: "draft-rev", Path: draft}, {Repo: "example/target", Revision: "target-rev", Path: target}}
}
