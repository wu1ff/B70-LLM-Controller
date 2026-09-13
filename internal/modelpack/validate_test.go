package modelpack

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func TestLoadUnifiedPackAndResolve(t *testing.T) {
	dir := t.TempDir()
	manifest := validManifest()
	writePack(t, dir, manifest)
	loaded, err := Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(loaded.Models) != 2 || len(loaded.Runtimes) != 1 || len(loaded.Modes) != 2 || len(loaded.Profiles) != 2 {
		t.Fatalf("Load() counts = %d/%d/%d/%d", len(loaded.Models), len(loaded.Runtimes), len(loaded.Modes), len(loaded.Profiles))
	}
	resolved, err := Resolve(loaded, loaded.Profiles[1])
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(resolved.DockerArgs, []string{"--runtime", "--model", "--mode", "--profile"}) ||
		!reflect.DeepEqual(resolved.Command, []string{"serve", "--draft", "/models/draft", "--tp", "1"}) ||
		!reflect.DeepEqual(resolved.Mounts, []Mount{{ModelID: "target", ContainerPath: "/models/target"}, {ModelID: "draft", ContainerPath: "/models/draft"}}) {
		t.Fatalf("Resolve() = %#v", resolved)
	}
}

func TestLoadRejectsUnknownFieldsAtEveryLayer(t *testing.T) {
	for _, target := range []string{"pack", "runtime", "model", "mode", "profile", "launch"} {
		t.Run(target, func(t *testing.T) {
			dir := t.TempDir()
			writePack(t, dir, validManifest())
			document := readDocument(t, dir)
			var object map[string]any
			switch target {
			case "pack":
				object = document
			case "runtime":
				object = document["runtimes"].([]any)[0].(map[string]any)
			case "model":
				object = document["models"].([]any)[0].(map[string]any)
			case "mode":
				object = document["modes"].([]any)[0].(map[string]any)
			case "profile":
				object = document["profiles"].([]any)[0].(map[string]any)
			case "launch":
				object = document["profiles"].([]any)[0].(map[string]any)["launch"].(map[string]any)
			}
			object["unexpected"] = true
			writeJSON(t, filepath.Join(dir, "pack.json"), document)
			assertLoadError(t, dir, `unknown field "unexpected"`)
		})
	}
}

func TestLoadValidatesLaunchBlocksAndControllerFlags(t *testing.T) {
	for _, layer := range []string{"runtime", "model", "mode", "profile"} {
		for _, forbidden := range []string{"--name=bad", "--rm", "-p8000:8000", "--publish", "--network=host", "-v/tmp:/tmp", "--mount=x", "-eA=B", "--env-file=x", "--label=x"} {
			t.Run(layer+"/"+forbidden, func(t *testing.T) {
				dir := t.TempDir()
				manifest := validManifest()
				switch layer {
				case "runtime":
					manifest.Runtimes[0].Launch.DockerArgs = []string{forbidden}
				case "model":
					manifest.Models[0].Launch.DockerArgs = []string{forbidden}
				case "mode":
					manifest.Modes[0].Launch.DockerArgs = []string{forbidden}
				case "profile":
					manifest.Profiles[0].Launch.DockerArgs = []string{forbidden}
				}
				writePack(t, dir, manifest)
				assertLoadError(t, dir, "is controlled by B70 LLM Controller")
			})
		}
	}

	t.Run("missing launch field", func(t *testing.T) {
		dir := t.TempDir()
		writePack(t, dir, validManifest())
		document := readDocument(t, dir)
		delete(document["modes"].([]any)[0].(map[string]any)["launch"].(map[string]any), "command_args")
		writeJSON(t, filepath.Join(dir, "pack.json"), document)
		assertLoadError(t, dir, "command_args list is required")
	})
}

func TestLoadAllowsOnlyFixedDeviceBind(t *testing.T) {
	t.Run("exact fixed bind", func(t *testing.T) {
		dir := t.TempDir()
		manifest := validManifest()
		manifest.Runtimes[0].Launch.DockerArgs = fixedDeviceBind[:]
		writePack(t, dir, manifest)
		if _, err := Load(dir); err != nil {
			t.Fatal(err)
		}
	})
	rejected := []struct {
		name string
		args []string
	}{
		{"bare mount flag", []string{"--mount"}},
		{"trailing bare mount flag", []string{"--device=/dev/dri:/dev/dri", "--mount"}},
		{"equals form", []string{"--mount=type=bind,source=/dev/dri/by-path,target=/dev/dri/by-path,readonly"}},
		{"other source", []string{"--mount", "type=bind,source=/dev/dri,target=/dev/dri,readonly"}},
		{"other target", []string{"--mount", "type=bind,source=/dev/dri/by-path,target=/mnt/by-path,readonly"}},
		{"writable", []string{"--mount", "type=bind,source=/dev/dri/by-path,target=/dev/dri/by-path"}},
		{"reordered fields", []string{"--mount", "type=bind,readonly,source=/dev/dri/by-path,target=/dev/dri/by-path"}},
		{"extra field", []string{"--mount", "type=bind,source=/dev/dri/by-path,target=/dev/dri/by-path,readonly,consistency=delegated"}},
		{"volume flag", []string{"--volume", "/dev/dri/by-path:/dev/dri/by-path:ro"}},
	}
	for _, test := range rejected {
		t.Run(test.name, func(t *testing.T) {
			dir := t.TempDir()
			manifest := validManifest()
			manifest.Runtimes[0].Launch.DockerArgs = test.args
			writePack(t, dir, manifest)
			assertLoadError(t, dir, "is controlled by B70 LLM Controller")
		})
	}
}

func TestLoadRejectsInvalidResolvedContracts(t *testing.T) {
	t.Run("duplicate environment", func(t *testing.T) {
		dir := t.TempDir()
		manifest := validManifest()
		manifest.Models[0].Launch.Environment["RUNTIME"] = "duplicate"
		writePack(t, dir, manifest)
		assertLoadError(t, dir, `duplicate environment key "RUNTIME"`)
	})
	t.Run("selected mount conflict", func(t *testing.T) {
		dir := t.TempDir()
		manifest := validManifest()
		manifest.Models[1].MountPath = manifest.Models[0].MountPath
		writePack(t, dir, manifest)
		assertLoadError(t, dir, "conflict at mount path")
	})
	t.Run("mutually exclusive targets share mount", func(t *testing.T) {
		dir := t.TempDir()
		manifest := validManifest()
		second := manifest.Models[0]
		second.ID, second.Name, second.Repo = "target-two", "Target Two", "example/target-two"
		manifest.Models = append(manifest.Models, second)
		profile := manifest.Profiles[0]
		profile.ID, profile.ModelID, profile.Context = "target-two", second.ID, 8192
		manifest.Profiles = append(manifest.Profiles, profile)
		writePack(t, dir, manifest)
		if _, err := Load(dir); err != nil {
			t.Fatal(err)
		}
	})
}

func TestLoadValidatesReferencesAndQualifications(t *testing.T) {
	tests := []struct {
		name string
		want string
		edit func(*Manifest)
	}{
		{"mode assistant missing", "references unknown model", func(m *Manifest) { m.Modes[1].Assistants = []string{"missing"} }},
		{"mode assistant target", "is not an assistant", func(m *Manifest) { m.Modes[1].Assistants = []string{"target"} }},
		{"target assistant", "is not a target", func(m *Manifest) { m.Profiles[0].ModelID = "draft" }},
		{"runtime missing", `unknown runtime "missing"`, func(m *Manifest) { m.Profiles[0].RuntimeID = "missing" }},
		{"mode missing", `unknown mode "missing"`, func(m *Manifest) { m.Profiles[0].Mode = "missing" }},
		{"cards", "cards must be", func(m *Manifest) { m.Profiles[0].Cards = 3 }},
		{"tensor parallel", "tensor_parallel must be", func(m *Manifest) { m.Profiles[0].TensorParallel = 3 }},
		{"context", "context must be positive", func(m *Manifest) { m.Profiles[0].Context = 0 }},
		{"duplicate tuple", "duplicate qualified profile", func(m *Manifest) { p := m.Profiles[0]; p.ID = "copy"; m.Profiles = append(m.Profiles, p) }},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			dir := t.TempDir()
			manifest := validManifest()
			test.edit(&manifest)
			writePack(t, dir, manifest)
			assertLoadError(t, dir, test.want)
		})
	}
}

func TestLoadRequiresOnlyREADMEAndManifest(t *testing.T) {
	dir := t.TempDir()
	writePack(t, dir, validManifest())
	if _, err := Load(dir); err != nil {
		t.Fatal(err)
	}
	entries, err := os.ReadDir(dir)
	if err != nil || len(entries) != 2 {
		t.Fatalf("pack files = %#v, %v", entries, err)
	}
}

func validManifest() Manifest {
	block := func(arg string) LaunchBlock {
		return LaunchBlock{DockerArgs: []string{arg}, Environment: map[string]string{}, CommandArgs: []string{}}
	}
	return Manifest{
		SchemaVersion: 1, ID: "test-pack", Name: "Test Pack", Version: "1.0.0",
		Models: []Model{
			{ID: "target", Name: "Target", Kind: "target", Repo: "example/target", Revision: "revision-1", Files: []ModelFile{{Path: "model.bin"}}, MountPath: "/models/target", Launch: block("--model")},
			{ID: "draft", Name: "Draft", Kind: "assistant", Repo: "example/draft", Revision: "revision-2", Files: []ModelFile{{Path: "model.bin"}}, MountPath: "/models/draft", Launch: LaunchBlock{DockerArgs: []string{}, Environment: map[string]string{}, CommandArgs: []string{}}},
		},
		Runtimes: []Runtime{{ID: "runtime", Image: "local/runtime:1", Digest: "sha256:" + strings.Repeat("a", 64), ContainerPort: 8000, HealthPath: "/health", Launch: RuntimeLaunch{DockerArgs: []string{"--runtime"}, Environment: map[string]string{"RUNTIME": "yes"}, Command: []string{"serve"}}}},
		Modes: []Mode{
			{ID: "base", DisplayName: "Base", Assistants: []string{}, Launch: LaunchBlock{DockerArgs: []string{}, Environment: map[string]string{}, CommandArgs: []string{}}},
			{ID: "draft", DisplayName: "Draft", Assistants: []string{"draft"}, Launch: LaunchBlock{DockerArgs: []string{"--mode"}, Environment: map[string]string{"MODE": "draft"}, CommandArgs: []string{"--draft", "/models/draft"}}},
		},
		Profiles: []Profile{
			{ID: "base-1", ModelID: "target", RuntimeID: "runtime", Cards: 1, TensorParallel: 1, Context: 4096, Mode: "base", Launch: LaunchBlock{DockerArgs: []string{}, Environment: map[string]string{}, CommandArgs: []string{"/models/target"}}},
			{ID: "draft-1", ModelID: "target", RuntimeID: "runtime", Cards: 1, TensorParallel: 1, Context: 4096, Mode: "draft", Launch: LaunchBlock{DockerArgs: []string{"--profile"}, Environment: map[string]string{"PROFILE": "draft"}, CommandArgs: []string{"--tp", "1"}}},
		},
	}
}

func writePack(t *testing.T, dir string, manifest Manifest) {
	t.Helper()
	if err := os.WriteFile(filepath.Join(dir, "README.md"), []byte("# Test\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	writeJSON(t, filepath.Join(dir, "pack.json"), manifest)
}

func readDocument(t *testing.T, dir string) map[string]any {
	t.Helper()
	data, err := os.ReadFile(filepath.Join(dir, "pack.json"))
	if err != nil {
		t.Fatal(err)
	}
	var document map[string]any
	if err := json.Unmarshal(data, &document); err != nil {
		t.Fatal(err)
	}
	return document
}

func writeJSON(t *testing.T, path string, value any) {
	t.Helper()
	data, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}
}

func assertLoadError(t *testing.T, dir, want string) {
	t.Helper()
	_, err := Load(dir)
	if err == nil || !strings.Contains(err.Error(), want) {
		t.Fatalf("Load() error = %v, want containing %q", err, want)
	}
}
