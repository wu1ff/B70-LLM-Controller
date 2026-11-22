package runtime

import (
	"encoding/csv"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"testing"
)

func TestBuildNativeLaunchReusesCacheAcrossManagedReplacement(t *testing.T) {
	dataHome := filepath.Join(t.TempDir(), "data,local")
	t.Setenv("XDG_DATA_HOME", dataHome)
	bin := t.TempDir()
	t.Setenv("PATH", bin)
	query := filepath.Join(bin, "dpkg-query")
	writeRuntimeVersion := func(version string) {
		t.Helper()
		if err := os.WriteFile(query, []byte("#!/bin/sh\nprintf 'installed\\tlibze1:amd64\\t%s\\n' '"+version+"'\n"), 0o700); err != nil {
			t.Fatal(err)
		}
	}
	writeRuntimeVersion("1.2.3")

	manifest := launchManifest()
	declared := &manifest.Runtimes[0]
	declared.ImageEntrypoint = true
	declared.PersistentJITCache = true
	declared.ContainerPort, declared.HealthPath = 18093, "/ready"
	declared.Launch.Command = nil
	manifest.Models[0].Launch.CommandArgs = nil
	manifest.Modes[0].Launch.CommandArgs = nil
	manifest.Profiles[0].Launch.CommandArgs = nil
	models := launchModels(t)
	port := unusedPort(t)
	build := func(access Access) Launch {
		t.Helper()
		launch, err := BuildLaunch(manifest, manifest.Profiles[0], models, access, port)
		if err != nil {
			t.Fatal(err)
		}
		return launch
	}
	local := build(AccessLocal)
	if local.BindAddress != "127.0.0.1" || local.HealthPath != "/ready" || local.ContainerPort != 18093 ||
		local.DockerArgs[len(local.DockerArgs)-1] != declared.Digest || slices.Contains(local.DockerArgs, "vllm") {
		t.Fatalf("native launch = %#v", local)
	}
	directory := cacheSource(t, local)
	if !strings.HasPrefix(directory, filepath.Join(dataHome, "b70ctl", "jit-cache")+"/") {
		t.Fatalf("cache path = %q", directory)
	}
	entry := filepath.Join(directory, "compiled-object")
	if err := os.WriteFile(entry, []byte("cache"), 0o600); err != nil {
		t.Fatal(err)
	}

	lan := build(AccessLAN)
	if lan.BindAddress != "0.0.0.0" || cacheSource(t, lan) != directory {
		t.Fatalf("LAN launch = %#v", lan)
	}
	var calls []string
	setDockerHook(t, func(args ...string) ([]byte, error) {
		calls = append(calls, args[0])
		if args[0] == "inspect" {
			return inspectJSON(false, 0, port), nil
		}
		return nil, nil
	})
	if err := Start(lan); err != nil || !reflect.DeepEqual(calls, []string{"inspect", "rm", "run"}) {
		t.Fatalf("mock replacement calls = %v, error = %v", calls, err)
	}
	if _, err := os.Stat(entry); err != nil {
		t.Fatalf("cache after replacement: %v", err)
	}

	writeRuntimeVersion("1.2.4")
	updatedHost := cacheSource(t, build(AccessLocal))
	if updatedHost == directory {
		t.Fatal("Intel runtime change reused the old cache")
	}
	if entries, err := os.ReadDir(updatedHost); err != nil || len(entries) != 0 {
		t.Fatalf("cache for changed runtime contains %v, error = %v", entries, err)
	}
	declared.Digest = "sha256:" + strings.Repeat("b", 64)
	if updatedImage := cacheSource(t, build(AccessLocal)); updatedImage == updatedHost || updatedImage == directory {
		t.Fatal("image change reused the old cache")
	}
}

func cacheSource(t *testing.T, launch Launch) string {
	t.Helper()
	for index, arg := range launch.DockerArgs {
		if arg != "--mount" {
			continue
		}
		fields, err := csv.NewReader(strings.NewReader(launch.DockerArgs[index+1])).Read()
		if err == nil && len(fields) == 3 && fields[0] == "type=bind" && fields[2] == "target=/cache" {
			return strings.TrimPrefix(fields[1], "source=")
		}
	}
	t.Fatalf("no writable cache bind in %v", launch.DockerArgs)
	return ""
}
