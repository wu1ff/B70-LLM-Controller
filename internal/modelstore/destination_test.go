package modelstore

import (
	"os"
	"path/filepath"
	"testing"
)

func TestRemoveDestinationClearsUndiscoverableDirectory(t *testing.T) {
	root := t.TempDir()
	destination := filepath.Join(root, "example__model")
	writeTestFile(t, filepath.Join(destination, "leftover.bin"), "junk")
	other := filepath.Join(root, "unrelated")
	writeTestMarker(t, other, "other/model", "revision-2")

	status, err := RemoveDestination(root, "example/model")
	if err != nil || status != RemovalRemoved {
		t.Fatalf("RemoveDestination() = %q, %v", status, err)
	}
	if _, err := os.Stat(destination); !os.IsNotExist(err) {
		t.Fatalf("destination still exists or returned unexpected error: %v", err)
	}
	if _, err := os.Stat(other); err != nil {
		t.Fatalf("unrelated directory was removed: %v", err)
	}
}

func TestRemoveDestinationMissingIsHarmless(t *testing.T) {
	status, err := RemoveDestination(t.TempDir(), "example/model")
	if err != nil || status != RemovalMissing {
		t.Fatalf("RemoveDestination() = %q, %v", status, err)
	}
}

func TestRemoveDestinationRefusesRecognizedModels(t *testing.T) {
	tests := []struct {
		name  string
		build func(t *testing.T, root string)
	}{
		{name: "marked for the same identity", build: func(t *testing.T, root string) {
			writeTestMarker(t, filepath.Join(root, "example__model"), "example/model", "revision-1")
		}},
		{name: "marked for another identity", build: func(t *testing.T, root string) {
			writeTestMarker(t, filepath.Join(root, "example__model"), "other/model", "revision-9")
		}},
		{name: "Hugging Face local layout", build: func(t *testing.T, root string) {
			writeTestFile(t, filepath.Join(root, "example__model", ".cache", "huggingface", "trees", "revision-1.json"), `{}`)
		}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			root := t.TempDir()
			test.build(t, root)
			status, err := RemoveDestination(root, "example/model")
			if err != nil || status != RemovalRecognizedModel {
				t.Fatalf("RemoveDestination() = %q, %v", status, err)
			}
			if _, err := os.Stat(filepath.Join(root, "example__model")); err != nil {
				t.Fatalf("recognized model was removed: %v", err)
			}
		})
	}
}

func TestRemoveDestinationRefusesUnsafeDestinations(t *testing.T) {
	t.Run("symbolic link", func(t *testing.T) {
		root := t.TempDir()
		target := filepath.Join(root, "real")
		writeTestFile(t, filepath.Join(target, "weights.bin"), "weights")
		if err := os.Symlink(target, filepath.Join(root, "example__model")); err != nil {
			t.Fatal(err)
		}
		if _, err := RemoveDestination(root, "example/model"); err == nil {
			t.Fatal("RemoveDestination() accepted a symbolic link")
		}
		if _, err := os.Stat(target); err != nil {
			t.Fatalf("symlink target was removed: %v", err)
		}
	})
	t.Run("regular file", func(t *testing.T) {
		root := t.TempDir()
		writeTestFile(t, filepath.Join(root, "example__model"), "not a directory")
		if _, err := RemoveDestination(root, "example/model"); err == nil {
			t.Fatal("RemoveDestination() accepted a regular file")
		}
		if _, err := os.Stat(filepath.Join(root, "example__model")); err != nil {
			t.Fatalf("file was removed: %v", err)
		}
	})
	t.Run("symlink escaping the model root", func(t *testing.T) {
		root := t.TempDir()
		external := t.TempDir()
		writeTestFile(t, filepath.Join(external, "weights.bin"), "external weights")
		if err := os.MkdirAll(root, 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.Symlink(external, filepath.Join(root, "example__model")); err != nil {
			t.Fatal(err)
		}
		if _, err := RemoveDestination(root, "example/model"); err == nil {
			t.Fatal("RemoveDestination() accepted an escaping symlink")
		}
		if _, err := os.Stat(filepath.Join(external, "weights.bin")); err != nil {
			t.Fatalf("external file was removed: %v", err)
		}
	})
	t.Run("unusable repository names", func(t *testing.T) {
		root := t.TempDir()
		for _, repo := range []string{"", ".", ".."} {
			if _, err := RemoveDestination(root, repo); err == nil {
				t.Fatalf("RemoveDestination(%q) was accepted", repo)
			}
		}
	})
}

func TestDestinationExistsReportsOccupation(t *testing.T) {
	root := t.TempDir()
	if occupied, err := DestinationExists(root, "example/model"); err != nil || occupied {
		t.Fatalf("DestinationExists() on empty root = %v, %v", occupied, err)
	}
	writeTestFile(t, filepath.Join(root, "example__model", "weights.bin"), "weights")
	if occupied, err := DestinationExists(root, "example/model"); err != nil || !occupied {
		t.Fatalf("DestinationExists() on occupied root = %v, %v", occupied, err)
	}
	if occupied, err := DestinationExists(root, "other/model"); err != nil || occupied {
		t.Fatalf("DestinationExists() on foreign repo = %v, %v", occupied, err)
	}
}

func TestDestinationNameMatchesControllerLayout(t *testing.T) {
	if got := DestinationName("Qwen/Qwen3.8-27B-FP8"); got != "Qwen__Qwen3.8-27B-FP8" {
		t.Fatalf("DestinationName() = %q", got)
	}
}
