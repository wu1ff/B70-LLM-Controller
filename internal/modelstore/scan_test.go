package modelstore

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func TestScanNonexistentRootIsEmpty(t *testing.T) {
	artifacts, err := Scan(filepath.Join(t.TempDir(), "missing"))
	if err != nil {
		t.Fatalf("Scan() error = %v", err)
	}
	if len(artifacts) != 0 {
		t.Fatalf("Scan() = %#v", artifacts)
	}
}

func TestScanDiscoversValidManualMarker(t *testing.T) {
	root := t.TempDir()
	modelPath := filepath.Join(root, "qwen38")
	writeTestFile(t, filepath.Join(modelPath, markerName), `{
  "repo": "Qwen/Qwen3.8-27B-FP8",
  "revision": "abc123"
}`)
	writeTestFile(t, filepath.Join(modelPath, "config.json"), `{}`)

	artifacts, err := Scan(root)
	if err != nil {
		t.Fatalf("Scan() error = %v", err)
	}
	want := []Artifact{{
		Repo:     "Qwen/Qwen3.8-27B-FP8",
		Revision: "abc123",
		Path:     modelPath,
	}}
	if !reflect.DeepEqual(artifacts, want) {
		t.Fatalf("Scan() = %#v, want %#v", artifacts, want)
	}
}

func TestScanIgnoresInvalidMarkersAndUnrelatedDirectories(t *testing.T) {
	root := t.TempDir()
	writeTestFile(t, filepath.Join(root, "malformed", markerName), `{"repo":`)
	writeTestFile(t, filepath.Join(root, "unknown-field", markerName), `{
  "repo": "example/model",
  "revision": "abc123",
  "extra": true
}`)
	writeTestFile(t, filepath.Join(root, "missing-repo", markerName), `{"repo":"","revision":"abc123"}`)
	writeTestFile(t, filepath.Join(root, "missing-revision", markerName), `{"repo":"example/model","revision":""}`)
	writeTestFile(t, filepath.Join(root, "unrelated", "config.json"), `{}`)
	writeTestFile(t, filepath.Join(root, "loose-file"), "junk")

	artifacts, err := Scan(root)
	if err != nil {
		t.Fatalf("Scan() error = %v", err)
	}
	if len(artifacts) != 0 {
		t.Fatalf("Scan() = %#v", artifacts)
	}
}

func TestScanDiscoversHFSnapshot(t *testing.T) {
	root := t.TempDir()
	snapshot := filepath.Join(root, "models--Qwen--Qwen3.8-27B-FP8", "snapshots", "abc123")
	writeTestFile(t, filepath.Join(snapshot, "config.json"), `{}`)

	artifacts, err := Scan(root)
	if err != nil {
		t.Fatalf("Scan() error = %v", err)
	}
	want := []Artifact{{
		Repo:     "Qwen/Qwen3.8-27B-FP8",
		Revision: "abc123",
		Path:     snapshot,
	}}
	if !reflect.DeepEqual(artifacts, want) {
		t.Fatalf("Scan() = %#v, want %#v", artifacts, want)
	}
}

func TestScanDiscoversHFSnapshotUnderHub(t *testing.T) {
	root := t.TempDir()
	snapshot := filepath.Join(root, "hub", "models--example--model", "snapshots", "revision-1")
	if err := os.MkdirAll(snapshot, 0o755); err != nil {
		t.Fatal(err)
	}

	artifacts, err := Scan(root)
	if err != nil {
		t.Fatalf("Scan() error = %v", err)
	}
	want := []Artifact{{Repo: "example/model", Revision: "revision-1", Path: snapshot}}
	if !reflect.DeepEqual(artifacts, want) {
		t.Fatalf("Scan() = %#v, want %#v", artifacts, want)
	}
}

func TestScanDiscoversHFLocalDirectoryRevisionMetadata(t *testing.T) {
	root := t.TempDir()
	modelPath := filepath.Join(root, "example__model")
	writeTestFile(t, filepath.Join(modelPath, ".cache", "huggingface", "trees", "revision-1.json"), `{}`)
	writeTestFile(t, filepath.Join(modelPath, "model.bin"), "model")

	artifacts, err := Scan(root)
	if err != nil {
		t.Fatal(err)
	}
	want := []Artifact{{Repo: "example/model", Revision: "revision-1", Path: modelPath}}
	if !reflect.DeepEqual(artifacts, want) {
		t.Fatalf("Scan() = %#v, want %#v", artifacts, want)
	}
	status, err := Remove(root, "example/model", "revision-1")
	if err != nil || status != RemovalExternalHFCache {
		t.Fatalf("Remove() = %q, %v", status, err)
	}
}

func TestFindRequiresExactIdentity(t *testing.T) {
	artifacts := []Artifact{{Repo: "Qwen/Qwen3.8-27B-FP8", Revision: "abc123", Path: "/models/qwen"}}

	found, ok := Find(artifacts, "Qwen/Qwen3.8-27B-FP8", "abc123")
	if !ok || found != artifacts[0] {
		t.Fatalf("Find() = %#v, %v", found, ok)
	}
	if _, ok := Find(artifacts, "Qwen/Qwen3.8-27B-FP8", "different"); ok {
		t.Fatal("Find() matched the wrong revision")
	}
	if _, ok := Find(artifacts, "other/Qwen3.8-27B-FP8", "abc123"); ok {
		t.Fatal("Find() matched the wrong repo")
	}
}

func TestScanOrderingAndDuplicateLookupAreDeterministic(t *testing.T) {
	root := t.TempDir()
	writeTestMarker(t, filepath.Join(root, "z-copy"), "zeta/model", "revision-2")
	writeTestMarker(t, filepath.Join(root, "b-copy"), "alpha/model", "revision-1")
	writeTestMarker(t, filepath.Join(root, "a-copy"), "alpha/model", "revision-1")
	writeTestMarker(t, filepath.Join(root, "later-revision"), "alpha/model", "revision-2")

	artifacts, err := Scan(root)
	if err != nil {
		t.Fatalf("Scan() error = %v", err)
	}
	want := []Artifact{
		{Repo: "alpha/model", Revision: "revision-1", Path: filepath.Join(root, "a-copy")},
		{Repo: "alpha/model", Revision: "revision-1", Path: filepath.Join(root, "b-copy")},
		{Repo: "alpha/model", Revision: "revision-2", Path: filepath.Join(root, "later-revision")},
		{Repo: "zeta/model", Revision: "revision-2", Path: filepath.Join(root, "z-copy")},
	}
	if !reflect.DeepEqual(artifacts, want) {
		t.Fatalf("Scan() = %#v, want %#v", artifacts, want)
	}

	found, ok := Find([]Artifact{want[1], want[0]}, "alpha/model", "revision-1")
	if !ok || found.Path != filepath.Join(root, "a-copy") {
		t.Fatalf("Find() = %#v, %v", found, ok)
	}
}

func TestRemoveDeletesOnlyMatchingMarkedStandaloneModel(t *testing.T) {
	root := t.TempDir()
	target := filepath.Join(root, "target")
	writeTestMarker(t, target, "example/model", "revision-1")
	writeTestFile(t, filepath.Join(target, "weights.bin"), "weights")
	other := filepath.Join(root, "other")
	writeTestMarker(t, other, "example/other", "revision-2")

	status, err := Remove(root, "example/model", "revision-1")
	if err != nil || status != RemovalRemoved {
		t.Fatalf("Remove() = %q, %v", status, err)
	}
	if _, err := os.Stat(target); !os.IsNotExist(err) {
		t.Fatalf("target still exists or returned unexpected error: %v", err)
	}
	if _, err := os.Stat(other); err != nil {
		t.Fatalf("unrelated model was removed: %v", err)
	}
}

func TestRemoveRetainsExternalHFCacheAndDuplicateCopies(t *testing.T) {
	t.Run("external Hugging Face cache", func(t *testing.T) {
		root := t.TempDir()
		snapshot := filepath.Join(root, "hub", "models--example--model", "snapshots", "revision-1")
		writeTestFile(t, filepath.Join(snapshot, "weights.bin"), "weights")

		status, err := Remove(root, "example/model", "revision-1")
		if err != nil || status != RemovalExternalHFCache {
			t.Fatalf("Remove() = %q, %v", status, err)
		}
		if _, err := os.Stat(snapshot); err != nil {
			t.Fatalf("Hugging Face snapshot was removed: %v", err)
		}
	})

	t.Run("multiple exact copies", func(t *testing.T) {
		root := t.TempDir()
		first := filepath.Join(root, "first")
		second := filepath.Join(root, "second")
		writeTestMarker(t, first, "example/model", "revision-1")
		writeTestMarker(t, second, "example/model", "revision-1")

		status, err := Remove(root, "example/model", "revision-1")
		if err != nil || status != RemovalMultipleCopies {
			t.Fatalf("Remove() = %q, %v", status, err)
		}
		for _, path := range []string{first, second} {
			if _, err := os.Stat(path); err != nil {
				t.Fatalf("duplicate copy %q was removed: %v", path, err)
			}
		}
	})
}

func TestRemoveMissingArtifactIsHarmless(t *testing.T) {
	status, err := Remove(t.TempDir(), "example/model", "revision-1")
	if err != nil || status != RemovalMissing {
		t.Fatalf("Remove() = %q, %v", status, err)
	}
}

func TestSafeRemovalPathsRejectsModelRootAndExternalPath(t *testing.T) {
	root := t.TempDir()
	if _, _, err := safeRemovalPaths(root, root); err == nil {
		t.Fatal("safeRemovalPaths() accepted the model root")
	}
	external := t.TempDir()
	if _, _, err := safeRemovalPaths(root, external); err == nil {
		t.Fatal("safeRemovalPaths() accepted an external path")
	}
}

func writeTestMarker(t *testing.T, path, repo, revision string) {
	t.Helper()
	writeTestFile(t, filepath.Join(path, markerName), `{"repo":"`+repo+`","revision":"`+revision+`"}`)
}

func writeTestFile(t *testing.T, path, contents string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(contents), 0o644); err != nil {
		t.Fatal(err)
	}
}
