package hf

import (
	"os"
	"path/filepath"
	"testing"
)

func TestTokenSetGetClear(t *testing.T) {
	root := t.TempDir()
	t.Setenv("XDG_CONFIG_HOME", root)
	t.Setenv("HF_TOKEN", "")

	configured, source, err := Configured()
	if err != nil || configured || source != "" {
		t.Fatalf("Configured() = %v, %q, %v", configured, source, err)
	}
	if err := Set("stored-token"); err != nil {
		t.Fatal(err)
	}
	token, source, err := Get()
	if err != nil || token != "stored-token" || source != SourceFile {
		t.Fatalf("Get() = %q, %q, %v", token, source, err)
	}
	path := filepath.Join(root, "b70ctl", "hf_token")
	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if info.Mode().Perm() != 0o600 {
		t.Fatalf("token mode = %o, want 600", info.Mode().Perm())
	}
	if err := Clear(); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatalf("token file after Clear() error = %v", err)
	}
	if err := Clear(); err != nil {
		t.Fatalf("second Clear() error = %v", err)
	}
}

func TestEnvironmentTokenOverridesStoredToken(t *testing.T) {
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	t.Setenv("HF_TOKEN", "")
	if err := Set("stored-token"); err != nil {
		t.Fatal(err)
	}
	t.Setenv("HF_TOKEN", "environment-token")
	token, source, err := Get()
	if err != nil || token != "environment-token" || source != SourceEnvironment {
		t.Fatalf("Get() = %q, %q, %v", token, source, err)
	}
}

func TestTokenFallbackPath(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("XDG_CONFIG_HOME", "")
	t.Setenv("HF_TOKEN", "")
	if err := Set("stored-token"); err != nil {
		t.Fatal(err)
	}
	info, err := os.Stat(filepath.Join(home, ".config", "b70ctl", "hf_token"))
	if err != nil {
		t.Fatal(err)
	}
	if info.Mode().Perm() != 0o600 {
		t.Fatalf("token mode = %o, want 600", info.Mode().Perm())
	}
}
