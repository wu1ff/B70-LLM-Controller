package modelpack

import (
	"path/filepath"
	"testing"
)

func TestImageEntrypointManifest(t *testing.T) {
	manifest := entrypointManifest()
	dir := t.TempDir()
	writePack(t, dir, manifest)
	loaded, err := Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	resolved, err := Resolve(loaded, loaded.Profiles[1])
	if err != nil || !resolved.ImageEntrypoint || !resolved.PersistentJITCache || len(resolved.Command) != 0 {
		t.Fatalf("Resolve() = %#v, %v", resolved, err)
	}
}

func TestRejectsConflictingRuntimeCapabilities(t *testing.T) {
	for _, tc := range []struct {
		name, want string
		edit       func(*Manifest)
	}{
		{"runtime command", "requires an empty composed command", func(m *Manifest) { m.Runtimes[0].Launch.Command = []string{"vllm", "serve"} }},
		{"model command", "requires an empty composed command", func(m *Manifest) { m.Models[0].Launch.CommandArgs = []string{"/models/target"} }},
		{"mode command", "requires an empty composed command", func(m *Manifest) { m.Modes[1].Launch.CommandArgs = []string{"--draft"} }},
		{"profile command", "requires an empty composed command", func(m *Manifest) { m.Profiles[1].Launch.CommandArgs = []string{"--port"} }},
		{"entrypoint override", "cannot be combined with --entrypoint", func(m *Manifest) { m.Profiles[1].Launch.DockerArgs = []string{"--entrypoint=/bin/sh"} }},
		{"cache model mount", "mount conflicts", func(m *Manifest) { m.Models[1].MountPath = "/cache/draft" }},
		{"cache tmpfs mount", "tmpfs mount conflicts", func(m *Manifest) { m.Runtimes[0].Launch.DockerArgs = []string{"--tmpfs", "/cache:rw"} }},
		{"arbitrary bind", "is controlled by B70 LLM Controller", func(m *Manifest) { m.Runtimes[0].Launch.DockerArgs = []string{"--mount=type=bind,src=/tmp,dst=/cache"} }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			manifest := entrypointManifest()
			tc.edit(&manifest)
			dir := t.TempDir()
			writePack(t, dir, manifest)
			assertLoadError(t, dir, tc.want)
		})
	}
}

func TestRejectsInvalidRuntimeCapabilityTypes(t *testing.T) {
	dir := t.TempDir()
	writePack(t, dir, entrypointManifest())
	document := readDocument(t, dir)
	runtime := document["runtimes"].([]any)[0].(map[string]any)
	for _, field := range []string{"image_entrypoint", "persistent_jit_cache"} {
		for _, value := range []any{nil, "true", 1} {
			runtime[field] = value
			writeJSON(t, filepath.Join(dir, "pack.json"), document)
			assertLoadError(t, dir, "runtime capability must be a JSON boolean")
		}
		runtime[field] = true
	}
}

func TestLoadControllerVersionRequirement(t *testing.T) {
	manifest := entrypointManifest()
	manifest.MinControllerVersion = "1.0.2"
	dir := t.TempDir()
	writePack(t, dir, manifest)
	loaded, err := Load(dir)
	if err != nil || loaded.CatalogName() != manifest.Name+" (Controller 1.0.2+)" {
		t.Fatalf("Load() = %#v, %v", loaded, err)
	}
	manifest.MinControllerVersion = "1.999.0"
	writePack(t, dir, manifest)
	assertLoadError(t, dir, "requires Controller 1.999.0 or newer (running ")
	document := readDocument(t, dir)
	for _, invalid := range []any{"no", "1.2", "1.2.0-rc1", "", nil} {
		document["min_controller_version"] = invalid
		writeJSON(t, filepath.Join(dir, "pack.json"), document)
		assertLoadError(t, dir, "must be a stable version")
	}
}

func TestMinimumControllerVersionComparison(t *testing.T) {
	for _, tc := range []struct {
		installed string
		allowed   bool
	}{
		{"1.0.1", false}, {"1.0.2-rc1", false}, {"1.0.2", true}, {"1.0.10", true}, {"1.1.0", true}, {"2.0.0", true},
	} {
		if err := requireControllerVersion("1.0.2", tc.installed); (err == nil) != tc.allowed {
			t.Errorf("Controller %s: %v", tc.installed, err)
		}
	}
}

func entrypointManifest() Manifest {
	manifest := validManifest()
	manifest.Runtimes[0].ImageEntrypoint = true
	manifest.Runtimes[0].PersistentJITCache = true
	manifest.Runtimes[0].Launch.Command = []string{}
	for index := range manifest.Models {
		manifest.Models[index].Launch.CommandArgs = []string{}
	}
	for index := range manifest.Modes {
		manifest.Modes[index].Launch.CommandArgs = []string{}
	}
	for index := range manifest.Profiles {
		manifest.Profiles[index].Launch.CommandArgs = []string{}
	}
	return manifest
}
