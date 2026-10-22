package modelpack

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

// Preserve the accepted serving contract while adapting only storage and port
// ownership to Controller. No server or GPU is started by this check.
func TestFlashNextDFlash2QualifiedProfile(t *testing.T) {
	root := filepath.Join("..", "..", "model-packs", "Qwen3.8-Flash-Next")
	manifest, err := Load(root)
	if err != nil {
		t.Fatal(err)
	}
	if manifest.Version != "2.0.0-dev.1" || len(manifest.Models) != 2 || len(manifest.Runtimes) != 1 || len(manifest.Modes) != 1 || len(manifest.Profiles) != 1 {
		t.Fatal("expected one local DFlash2 runtime/profile and target/assistant")
	}
	runtime := manifest.Runtimes[0]
	if runtime.Digest != "sha256:6e2ab4e686680a5208ff0119c4637341cba81942bd39b6576ea14dc8f547b83b" || runtime.Registry != "" || runtime.Image != "qwen38-flashnext-dflash2:integrated-serving-v1" {
		t.Fatalf("runtime must retain accepted local identity without a public registry: %#v", runtime)
	}
	target, draft := manifest.Models[0], manifest.Models[1]
	if target.Revision != "4c67bf686b7f7fd386bae6b07ab59e8ff1d5b897" || target.Repo != "Intel/Qwen3.8-Flash-Next-W4A16-AutoRound" || target.MountPath != "/models" || len(target.Files) != 29 {
		t.Fatal("target identity changed")
	}
	if draft.Kind != "assistant" || draft.Revision != "a22a1744e85c17364dcc54a6a420af7135d4c7f6" || draft.Repo != "incoai/Qwen3.8-Flash-Next-DFlash2" || draft.MountPath != "/draft" || len(draft.Files) != 6 {
		t.Fatal("draft identity changed")
	}
	var inventory struct {
		Model Model `json:"model"`
	}
	readFlashNextJSON(t, filepath.Join(root, "patches/production/manifests/target-pinned-inventory.json"), &inventory)
	if !reflect.DeepEqual(target.Files, inventory.Model.Files) {
		t.Fatal("target file inventory differs from pinned authority")
	}
	var api struct {
		SHA      string `json:"sha"`
		Siblings []struct {
			Path string `json:"rfilename"`
			Size int64  `json:"size"`
		} `json:"siblings"`
	}
	readFlashNextJSON(t, filepath.Join(root, "patches/production/manifests/draft-pinned-api.json"), &api)
	for i, f := range draft.Files {
		if api.SHA != draft.Revision || f.Path != api.Siblings[i].Path || f.Size == nil || *f.Size != api.Siblings[i].Size {
			t.Fatal("draft file inventory differs from pinned authority")
		}
	}
	profile := manifest.Profiles[0]
	if profile.ID != "w4a16-dflash2-tp4-262144" || profile.Cards != 4 || profile.TensorParallel != 4 || profile.Context != 262144 || profile.Mode != "dflash2" || !reflect.DeepEqual(manifest.Modes[0].Assistants, []string{draft.ID}) {
		t.Fatal("DFlash2 profile changed")
	}
	var authority struct {
		Config struct {
			Command     []string `json:"Cmd"`
			Environment []string `json:"Env"`
		} `json:"Config"`
	}
	readFlashNextJSON(t, filepath.Join(root, "patches/production/manifests/serving-contract.json"), &authority)
	resolved, err := Resolve(manifest, profile)
	if err != nil {
		t.Fatal(err)
	}
	serve, err := ServeCommand(resolved, target.MountPath)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(flashNextCommandFlags(t, serve), flashNextCommandFlags(t, authority.Config.Command)) {
		t.Fatal("flags differ from frozen accepted command")
	}
	acceptedEnv := map[string]string{}
	for _, s := range authority.Config.Environment {
		k, v, _ := strings.Cut(s, "=")
		acceptedEnv[k] = v
	}
	for k, v := range resolved.Environment {
		if acceptedEnv[k] != v {
			t.Fatalf("environment differs: %s", k)
		}
	}
	for _, k := range []string{"VLLM_USE_V2_MODEL_RUNNER", "VLLM_XPU_DFLASH_FIXED_SPLITS", "PLE_TABLE_PATH", "LD_PRELOAD", "B70_PLE_DISK_BF16", "PYTHONPATH", "ZE_AFFINITY_MASK"} {
		if resolved.Environment[k] == "" {
			t.Fatalf("missing serving override %s", k)
		}
	}
	wantArgs := []string{"--pull", "never", "--device=/dev/dri:/dev/dri", "--ipc=host", "--security-opt", "seccomp=unconfined", "--security-opt", "label=disable"}
	wantArgs = append(wantArgs, fixedDeviceBind[:]...)
	wantArgs = append(wantArgs, fixedDFlashWorkVolume[:]...)
	if !reflect.DeepEqual(resolved.DockerArgs, wantArgs) {
		t.Fatal("device/work-volume contract differs")
	}
	if !reflect.DeepEqual(resolved.Mounts, []Mount{{ModelID: target.ID, ContainerPath: "/models"}, {ModelID: draft.ID, ContainerPath: "/draft"}}) {
		t.Fatal("target and assistant mounts differ")
	}
}

func readFlashNextJSON(t *testing.T, path string, value any) {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if err = json.Unmarshal(data, value); err != nil {
		t.Fatal(err)
	}
}
func flashNextCommandFlags(t *testing.T, command []string) map[string]string {
	t.Helper()
	if len(command) < 3 || command[0] != "vllm" || command[1] != "serve" {
		t.Fatalf("invalid serve command %v", command)
	}
	flags := map[string]string{"model": command[2]}
	for i := 3; i < len(command); i++ {
		flag := command[i]
		if !strings.HasPrefix(flag, "--") {
			t.Fatalf("unexpected positional argument %q", flag)
		}
		if _, exists := flags[flag]; exists {
			t.Fatalf("duplicate flag %q", flag)
		}
		value := "true"
		if i+1 < len(command) && !strings.HasPrefix(command[i+1], "--") {
			i++
			value = command[i]
		}
		flags[flag] = value
	}
	return flags
}
