package modelstore

import (
	"os"
	"path/filepath"
	"testing"
)

const stagingTestRevision = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"

func TestStagingPathMatchesControllerLayout(t *testing.T) {
	root := t.TempDir()
	path, err := StagingPath(root, "example/model", stagingTestRevision)
	if err != nil {
		t.Fatal(err)
	}
	want := filepath.Join(root, StagingDirName, "example__model-"+stagingTestRevision)
	if path != want {
		t.Fatalf("StagingPath() = %q, want %q", path, want)
	}
	other, err := StagingPath(root, "example/model", "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")
	if err != nil {
		t.Fatal(err)
	}
	if other == path {
		t.Fatal("different revisions share a staging directory")
	}
	another, err := StagingPath(root, "other/model", stagingTestRevision)
	if err != nil {
		t.Fatal(err)
	}
	if another == path {
		t.Fatal("different repos share a staging directory")
	}
}

func TestStagingPathRejectsUnsafeRevisions(t *testing.T) {
	root := t.TempDir()
	for _, revision := range []string{"", ".", "..", "a/b", `a\b`, "/absolute"} {
		if _, err := StagingPath(root, "example/model", revision); err == nil {
			t.Fatalf("StagingPath(revision = %q) succeeded", revision)
		}
	}
}

func TestScanIgnoresStagingDirectory(t *testing.T) {
	root := t.TempDir()
	// Simulate the worst case: a crash between writing the final marker and
	// the promotion rename leaves staging holding a complete marked model.
	staging, err := StagingPath(root, "example/model", stagingTestRevision)
	if err != nil {
		t.Fatal(err)
	}
	writeTestMarker(t, staging, "example/model", stagingTestRevision)
	writeTestFile(t, filepath.Join(staging, "model.bin"), "model")
	final := filepath.Join(root, "final__model")
	writeTestMarker(t, final, "final/model", "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")

	artifacts, err := Scan(root)
	if err != nil {
		t.Fatal(err)
	}
	if len(artifacts) != 1 || artifacts[0].Repo != "final/model" || artifacts[0].Path != final {
		t.Fatalf("Scan() = %#v, want only the final model", artifacts)
	}
}

func TestScanIgnoresStagingSymlink(t *testing.T) {
	root := t.TempDir()
	target := filepath.Join(root, "elsewhere")
	writeTestMarker(t, target, "example/model", stagingTestRevision)
	stagingParent := filepath.Join(root, StagingDirName)
	if err := os.MkdirAll(stagingParent, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(target, filepath.Join(stagingParent, "example__model-"+stagingTestRevision)); err != nil {
		t.Fatal(err)
	}

	artifacts, err := Scan(root)
	if err != nil {
		t.Fatal(err)
	}
	if len(artifacts) != 1 || artifacts[0].Path != target {
		t.Fatalf("Scan() = %#v, want the real directory only", artifacts)
	}
}
