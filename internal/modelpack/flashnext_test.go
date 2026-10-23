package modelpack

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

// Compare the published pack against its exact qualified standalone launch contract.
func TestFlashNextDiskQualifiedProfiles(t *testing.T) {
	root := filepath.Join("..", "..", "model-packs", "Qwen3.8-Flash-Next")
	manifest, err := Load(root)
	if err != nil {
		t.Fatal(err)
	}
	digest := "sha256:6e2ab4e686680a5208ff0119c4637341cba81942bd39b6576ea14dc8f547b83b"
	if manifest.Version != "1.0.5" || len(manifest.Runtimes) != 1 || len(manifest.Modes) != 1 || len(manifest.Profiles) != 1 {
		t.Fatal("DFlash2 release must retain one runtime and one qualified seven-proposal profile")
	}
	runtime := manifest.Runtimes[0]
	if runtime.Digest != digest || runtime.Registry != "ghcr.io/wu1ff/qwen38-flashnext-b70@"+digest || runtime.Image != "ghcr.io/wu1ff/qwen38-flashnext-b70:dflash2" {
		t.Fatalf("unexpected runtime authority: %#v", runtime)
	}
	if len(manifest.Models) != 2 || manifest.Models[0].Revision != "4c67bf686b7f7fd386bae6b07ab59e8ff1d5b897" || manifest.Models[1].Revision != "a22a1744e85c17364dcc54a6a420af7135d4c7f6" {
		t.Fatal("qualified target or BF16 assistant identity changed")
	}
	seen := make(map[string]bool)
	for _, profile := range manifest.Profiles {
		t.Run(profile.ID, func(t *testing.T) {
			if profile.Mode != "dflash2" {
				t.Fatalf("unexpected mode %q", profile.Mode)
			}
			validContext := profile.Context == 262144
			wantID := fmt.Sprintf("w4a16-%s-tp4-%d", profile.Mode, profile.Context)
			if !validContext || profile.ID != wantID || seen[wantID] || profile.Cards != 4 || profile.TensorParallel != 4 {
				t.Fatalf("invalid or duplicate profile: %#v", profile)
			}
			seen[wantID] = true
			var authority struct {
				Environment map[string]string `json:"environment"`
				Command     []string          `json:"command"`
				DockerArgs  []string          `json:"docker_args"`
				Mounts      []Mount           `json:"model_mounts"`
			}
			data, err := os.ReadFile(filepath.Join(root, "patches", "production", "manifests", "qualified-launch.json"))
			if err != nil {
				t.Fatal(err)
			}
			if err := json.Unmarshal(data, &authority); err != nil {
				t.Fatal(err)
			}
			resolved, err := Resolve(manifest, profile)
			if err != nil {
				t.Fatal(err)
			}
			serve, err := ServeCommand(resolved, manifest.Models[0].MountPath)
			if err != nil {
				t.Fatal(err)
			}
			want := flashNextCommandFlags(t, authority.Command)
			want["--max-model-len"] = fmt.Sprint(profile.Context)
			if got := flashNextCommandFlags(t, serve); !reflect.DeepEqual(got, want) {
				t.Fatalf("command differs from qualified %s: got %v, want %v", profile.Mode, got, want)
			}
			if !reflect.DeepEqual(resolved.Environment, authority.Environment) {
				t.Fatalf("environment differs from qualified %s", profile.Mode)
			}
			if !reflect.DeepEqual(resolved.DockerArgs, authority.DockerArgs) || resolved.ContainerPort != 8000 || resolved.HealthPath != "/health" {
				t.Fatal("qualified device binds, port or health contract changed")
			}
			if !reflect.DeepEqual(resolved.Mounts, authority.Mounts) || len(resolved.Mounts) != 2 || resolved.Mounts[0].ContainerPath != "/models" || resolved.Mounts[1].ContainerPath != "/draft" {
				t.Fatal("qualified target and assistant mounts changed")
			}
			if resolved.Environment["TMPDIR"] != "/tmp" || strings.Contains(strings.Join(resolved.DockerArgs, " "), "/work") {
				t.Fatal("/work must use the writable container filesystem and temporary files must use /tmp")
			}
			var speculation struct {
				Method    string `json:"method"`
				Proposals int    `json:"num_speculative_tokens"`
			}
			if err := json.Unmarshal([]byte(want["--speculative-config"]), &speculation); err != nil || speculation.Method != "dflash" || speculation.Proposals != 7 {
				t.Fatal("qualified seven-proposal DFlash2 configuration changed")
			}
		})
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
