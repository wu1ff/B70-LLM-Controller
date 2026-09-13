package modelstore

import (
	"path/filepath"
	"testing"
)

func TestManagedRequiresMatchingModelMarker(t *testing.T) {
	root := t.TempDir()
	managed := filepath.Join(root, "example__model")
	writeTestMarker(t, managed, "example/model", "revision-1")
	writeTestFile(t, filepath.Join(managed, "model.bin"), "weight")
	artifacts, err := Scan(root)
	if err != nil {
		t.Fatalf("Scan() error = %v", err)
	}
	if len(artifacts) != 1 || !artifacts[0].Managed() {
		t.Fatalf("marker-backed artifact is not managed: %#v", artifacts)
	}
}

func TestManagedRejectsExternalAndUnrecognizedArtifacts(t *testing.T) {
	root := t.TempDir()
	snapshot := filepath.Join(root, "hub", "models--example--model", "snapshots", "revision-1")
	writeTestFile(t, filepath.Join(snapshot, "model.bin"), "weight")
	hfLocal := filepath.Join(root, "example__model")
	writeTestFile(t, filepath.Join(hfLocal, ".cache", "huggingface", "trees", "revision-1.json"), `{}`)
	writeTestFile(t, filepath.Join(hfLocal, "model.bin"), "weight")
	// A cache-layout snapshot that carries a marker for a different identity
	// than the directory itself represents.
	foreignMarker := filepath.Join(root, "models--other--model", "snapshots", "revision-2")
	writeTestMarker(t, foreignMarker, "example/model", "revision-1")
	writeTestFile(t, filepath.Join(root, "unrelated", "config.json"), `{}`)

	artifacts, err := Scan(root)
	if err != nil {
		t.Fatalf("Scan() error = %v", err)
	}
	if len(artifacts) == 0 {
		t.Fatal("Scan() discovered nothing")
	}
	for _, artifact := range artifacts {
		if artifact.Managed() {
			t.Fatalf("artifact %q classified as managed: %#v", artifact.Path, artifact)
		}
	}
}

func TestManagedMatchesRemoveAcceptance(t *testing.T) {
	root := t.TempDir()
	managed := filepath.Join(root, "example__model")
	writeTestMarker(t, managed, "example/model", "revision-1")

	artifacts, err := Scan(root)
	if err != nil {
		t.Fatalf("Scan() error = %v", err)
	}
	if len(artifacts) != 1 || !artifacts[0].Managed() {
		t.Fatalf("marker-backed artifact is not managed: %#v", artifacts)
	}
	if status, err := Remove(root, "example/model", "revision-1"); err != nil || status != RemovalRemoved {
		t.Fatalf("Remove() = %q, %v; Managed() agreed with a model Remove refuses", status, err)
	}
}
