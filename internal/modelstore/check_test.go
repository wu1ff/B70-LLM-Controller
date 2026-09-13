package modelstore

import (
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"b70ctl/internal/modelpack"
)

func TestCheckCompleteInventoryKnownAndUnknownSizesAndExtras(t *testing.T) {
	root := t.TempDir()
	writeTestFile(t, filepath.Join(root, "config.json"), "{}")
	writeTestFile(t, filepath.Join(root, "nested", "model.bin"), "weights")
	writeTestFile(t, filepath.Join(root, "extra.txt"), "harmless")
	two := int64(2)
	result := Check(root, []modelpack.ModelFile{
		{Path: "config.json", Size: &two},
		{Path: "nested/model.bin"},
	})
	if !result.Complete || result.CheckedFiles != 2 || len(result.MissingFiles) != 0 || len(result.SizeMismatches) != 0 {
		t.Fatalf("Check() = %#v", result)
	}
}

func TestCheckCountsMissingAndWrongSize(t *testing.T) {
	root := t.TempDir()
	writeTestFile(t, filepath.Join(root, "wrong.bin"), "123")
	four := int64(4)
	result := Check(root, []modelpack.ModelFile{
		{Path: "missing-one"},
		{Path: "missing-two"},
		{Path: "wrong.bin", Size: &four},
	})
	if result.Complete || result.CheckedFiles != 3 || !reflect.DeepEqual(result.MissingFiles, []string{"missing-one", "missing-two"}) || !reflect.DeepEqual(result.SizeMismatches, []string{"wrong.bin"}) {
		t.Fatalf("Check() = %#v", result)
	}
}

func TestCheckAcceptsZeroByteFile(t *testing.T) {
	root := t.TempDir()
	writeTestFile(t, filepath.Join(root, "empty"), "")
	zero := int64(0)
	if result := Check(root, []modelpack.ModelFile{{Path: "empty", Size: &zero}}); !result.Complete {
		t.Fatalf("Check() = %#v", result)
	}
}

func TestCheckRejectsDirectoryAndNonexistentArtifact(t *testing.T) {
	root := t.TempDir()
	if err := os.Mkdir(filepath.Join(root, "expected"), 0o755); err != nil {
		t.Fatal(err)
	}
	for _, path := range []string{root, filepath.Join(root, "missing-root")} {
		result := Check(path, []modelpack.ModelFile{{Path: "expected"}})
		if result.Complete || !reflect.DeepEqual(result.MissingFiles, []string{"expected"}) {
			t.Fatalf("Check(%q) = %#v", path, result)
		}
	}
}

func TestCheckExpectedPathMustRemainInsideArtifact(t *testing.T) {
	parent := t.TempDir()
	root := filepath.Join(parent, "artifact")
	writeTestFile(t, filepath.Join(parent, "outside"), "model")
	if err := os.MkdirAll(root, 0o755); err != nil {
		t.Fatal(err)
	}
	result := Check(root, []modelpack.ModelFile{{Path: "../outside"}})
	if result.Complete || !reflect.DeepEqual(result.MissingFiles, []string{"../outside"}) {
		t.Fatalf("Check() = %#v", result)
	}
}

func TestCheckFollowsHFSnapshotSymlinkToRegularBlob(t *testing.T) {
	root := t.TempDir()
	blob := filepath.Join(root, "blobs", "abc")
	writeTestFile(t, blob, "blob")
	snapshot := filepath.Join(root, "snapshots", "revision")
	if err := os.MkdirAll(snapshot, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(blob, filepath.Join(snapshot, "model.bin")); err != nil {
		t.Fatal(err)
	}
	four := int64(4)
	result := Check(snapshot, []modelpack.ModelFile{{Path: "model.bin", Size: &four}})
	if !result.Complete {
		t.Fatalf("Check() = %#v", result)
	}
}

func TestAssessDistinguishesMissingPresentAndIncomplete(t *testing.T) {
	root := t.TempDir()
	model := modelpack.Model{Repo: "example/model", Revision: "revision", Files: []modelpack.ModelFile{{Path: "model.bin"}}}
	if result := Assess(nil, model); result.State != Missing {
		t.Fatalf("missing Assess() = %#v", result)
	}
	artifacts := []Artifact{{Repo: model.Repo, Revision: model.Revision, Path: root}}
	if result := Assess(artifacts, model); result.State != Incomplete {
		t.Fatalf("incomplete Assess() = %#v", result)
	}
	writeTestFile(t, filepath.Join(root, "model.bin"), "model")
	if result := Assess(artifacts, model); result.State != Present {
		t.Fatalf("present Assess() = %#v", result)
	}
}

func TestLiveCurrentQwenCompleteness(t *testing.T) {
	root := os.Getenv("B70_LIVE_MODEL_ROOT")
	if root == "" {
		t.Skip("set B70_LIVE_MODEL_ROOT to run the live model completeness check")
	}
	packRoot := filepath.Join("..", "..", "model-packs", "Qwen3.8-27B")
	manifest, err := modelpack.Load(packRoot)
	if err != nil {
		t.Fatal(err)
	}
	artifacts, err := Scan(root)
	if err != nil {
		t.Fatal(err)
	}
	for _, model := range manifest.Models {
		result := Assess(artifacts, model)
		t.Logf("%s expected=%d checked=%d missing=%d size_mismatches=%d result=%s",
			model.Repo, len(model.Files), result.Check.CheckedFiles, len(result.Check.MissingFiles), len(result.Check.SizeMismatches), result.State)
		if result.State != Present {
			t.Error(fmt.Sprintf("%s is %s", model.Repo, result.State))
		}
	}
}
