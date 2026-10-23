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
	digest := "sha256:dc25c8cd205a828ad4df3ae9aa3e9505f0a51550010ac9a115336473ca268086"
	if manifest.Version != "1.0.4" || len(manifest.Runtimes) != 1 || len(manifest.Modes) != 1 || len(manifest.Profiles) != 1 {
		t.Fatal("Intel release must retain one runtime and one qualified dynamic MTP5 profile")
	}
	runtime := manifest.Runtimes[0]
	if runtime.Digest != digest || runtime.Registry != "ghcr.io/wu1ff/qwen38-flashnext-b70@"+digest || runtime.Image != "ghcr.io/wu1ff/qwen38-flashnext-b70:disk" {
		t.Fatalf("unexpected runtime authority: %#v", runtime)
	}
	seen := make(map[string]bool)
	for _, profile := range manifest.Profiles {
		t.Run(profile.ID, func(t *testing.T) {
			if profile.Mode != "mtp5-dynamic" {
				t.Fatalf("unexpected mode %q", profile.Mode)
			}
			validContext := profile.Context == 262144
			wantID := fmt.Sprintf("w4a16-%s-tp4-%d", profile.Mode, profile.Context)
			if !validContext || profile.ID != wantID || seen[wantID] || profile.Cards != 4 || profile.TensorParallel != 4 {
				t.Fatalf("invalid or duplicate profile: %#v", profile)
			}
			seen[wantID] = true
			var authority struct {
				Environment   map[string]string `json:"env"`
				Command       []string          `json:"command"`
				DockerOptions []string          `json:"docker_options"`
				Mounts        []string          `json:"mounts"`
			}
			data, err := os.ReadFile(filepath.Join(root, "patches", "disk", "profiles", profile.Mode+".json"))
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
			options := append(authority.DockerOptions, "--mount", authority.Mounts[0])
			if !reflect.DeepEqual(resolved.DockerArgs, options) || resolved.ContainerPort != 8000 || resolved.HealthPath != "/health" {
				t.Fatal("qualified device binds, port or health contract changed")
			}
			if !reflect.DeepEqual(resolved.Mounts, []Mount{{ModelID: manifest.Models[0].ID, ContainerPath: manifest.Models[0].MountPath}}) {
				t.Fatal("qualified checkpoint mount changed")
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
