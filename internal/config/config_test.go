package config

import (
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func TestDefaults(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	got, err := Default()
	if err != nil {
		t.Fatal(err)
	}
	want := Config{
		ModelDirectory: filepath.Join(home, "b70ctl-models"),
		DefaultAccess:  AccessLocal,
		DefaultPort:    8000,
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("Default() = %#v, want %#v", got, want)
	}
}

func TestSaveLoad(t *testing.T) {
	path := filepath.Join(t.TempDir(), "nested", "config.json")
	want := Config{ModelDirectory: "/srv/models", DefaultAccess: AccessLAN, DefaultPort: 18000}
	if err := Save(path, want); err != nil {
		t.Fatal(err)
	}
	got, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("Load() = %#v, want %#v", got, want)
	}
	contents, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasSuffix(string(contents), "\n") {
		t.Fatal("saved config has no trailing newline")
	}
	if strings.Contains(string(contents), "hf_token") || strings.Contains(string(contents), "stored-token") {
		t.Fatal("normal config contains Hugging Face token data")
	}
}

func TestSaveLoadWithLastRun(t *testing.T) {
	path := filepath.Join(t.TempDir(), "config.json")
	want := Config{
		ModelDirectory: "/srv/models",
		DefaultAccess:  AccessLAN,
		DefaultPort:    18000,
		LastRun:        &RunSelection{ModelID: "qwen38-27b-uncensored-fp8", Cards: 4, Context: 262144, Mode: "dflash2"},
	}
	if err := Save(path, want); err != nil {
		t.Fatal(err)
	}
	got, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("Load() = %#v, want %#v", got, want)
	}
	contents, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	for _, field := range []string{`"last_run"`, `"model_id": "qwen38-27b-uncensored-fp8"`, `"cards": 4`, `"context": 262144`, `"mode": "dflash2"`} {
		if !strings.Contains(string(contents), field) {
			t.Fatalf("saved config missing %s: %s", field, contents)
		}
	}
}

func TestSaveOmitsLastRunWhenNeverSet(t *testing.T) {
	path := filepath.Join(t.TempDir(), "config.json")
	if err := Save(path, Config{ModelDirectory: "/srv/models", DefaultAccess: AccessLocal, DefaultPort: 8000}); err != nil {
		t.Fatal(err)
	}
	contents, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(contents), "last_run") {
		t.Fatalf("saved config contains last_run without a stored preference: %s", contents)
	}
}

func TestLoadLegacyConfigWithoutLastRun(t *testing.T) {
	path := filepath.Join(t.TempDir(), "config.json")
	body := `{"model_directory":"/models","default_access":"local","default_port":8000}`
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	got, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	if got.LastRun != nil {
		t.Fatalf("Load() last run = %#v, want nil", got.LastRun)
	}
	if got.ModelDirectory != "/models" || got.DefaultAccess != AccessLocal || got.DefaultPort != 8000 {
		t.Fatalf("Load() = %#v", got)
	}
}

func TestLoadKeepsUnresolvableLastRun(t *testing.T) {
	path := filepath.Join(t.TempDir(), "config.json")
	body := `{"model_directory":"/models","default_access":"local","default_port":8000,"last_run":{"model_id":"gone","cards":0,"context":0,"mode":""}}`
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	got, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	if got.LastRun == nil || got.LastRun.ModelID != "gone" {
		t.Fatalf("Load() last run = %#v, want preserved stale value", got.LastRun)
	}
}

func TestLoadRejectsInvalidValues(t *testing.T) {
	tests := []struct {
		name string
		body string
		want string
	}{
		{name: "access", body: `{"model_directory":"/models","default_access":"public","default_port":8000}`, want: "access"},
		{name: "port", body: `{"model_directory":"/models","default_access":"local","default_port":80}`, want: "port"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "config.json")
			if err := os.WriteFile(path, []byte(test.body), 0o644); err != nil {
				t.Fatal(err)
			}
			if _, err := Load(path); err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("Load() error = %v", err)
			}
		})
	}
}

func TestResolvePaths(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("XDG_CONFIG_HOME", "")
	t.Setenv("XDG_DATA_HOME", "")
	got, err := ResolvePaths()
	if err != nil {
		t.Fatal(err)
	}
	if got.ConfigFile != filepath.Join(home, ".config", "b70ctl", "config.json") ||
		got.DataRoot != filepath.Join(home, ".local", "share", "b70ctl") ||
		got.PacksRoot != filepath.Join(home, ".local", "share", "b70ctl", "packs") {
		t.Fatalf("ResolvePaths() = %#v", got)
	}

	t.Setenv("XDG_CONFIG_HOME", "/cfg")
	t.Setenv("XDG_DATA_HOME", "/data")
	got, err = ResolvePaths()
	if err != nil {
		t.Fatal(err)
	}
	if got.ConfigFile != "/cfg/b70ctl/config.json" || got.DataRoot != "/data/b70ctl" || got.PacksRoot != "/data/b70ctl/packs" {
		t.Fatalf("ResolvePaths() with XDG = %#v", got)
	}
}
