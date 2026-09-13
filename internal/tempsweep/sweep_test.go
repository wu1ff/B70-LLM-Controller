package tempsweep

import (
	"os"
	"path/filepath"
	"testing"
)

func makeDirectory(t *testing.T, path string) string {
	t.Helper()
	if err := os.MkdirAll(path, 0o755); err != nil {
		t.Fatal(err)
	}
	return path
}

func makeFile(t *testing.T, path, contents string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(contents), 0o644); err != nil {
		t.Fatal(err)
	}
}

func assertGone(t *testing.T, path string) {
	t.Helper()
	if _, err := os.Lstat(path); !os.IsNotExist(err) {
		t.Fatalf("%s still exists: %v", path, err)
	}
}

func assertPresent(t *testing.T, path string) {
	t.Helper()
	if _, err := os.Lstat(path); err != nil {
		t.Fatalf("%s missing: %v", path, err)
	}
}

func TestSweepRemovesLegacyTempDirsAcrossManagedRoots(t *testing.T) {
	models := t.TempDir()
	data := t.TempDir()
	packs := t.TempDir()
	legacyModel := makeDirectory(t, filepath.Join(models, ".b70ctl-download-123456"))
	makeFile(t, filepath.Join(legacyModel, "model.bin"), "partial")
	packTemp := makeDirectory(t, filepath.Join(data, ".pack-download-99"))
	importTemp := makeDirectory(t, filepath.Join(packs, ".import-7"))

	removed, err := Sweep(
		Root{Path: models, Prefix: ".b70ctl-download-"},
		Root{Path: data, Prefix: ".pack-download-"},
		Root{Path: packs, Prefix: ".import-"},
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(removed) != 3 {
		t.Fatalf("Sweep() removed %#v, want 3 directories", removed)
	}
	assertGone(t, legacyModel)
	assertGone(t, packTemp)
	assertGone(t, importTemp)
}

func TestSweepPreservesStagingFinalModelsAndUnrelatedEntries(t *testing.T) {
	models := t.TempDir()
	data := t.TempDir()
	packs := t.TempDir()
	finalModel := makeDirectory(t, filepath.Join(models, "example__model"))
	makeFile(t, filepath.Join(finalModel, "model.bin"), "model")
	stagingRoot := makeDirectory(t, filepath.Join(models, ".b70ctl-staging"))
	stagingKey := makeDirectory(t, filepath.Join(stagingRoot, "example__model-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"))
	makeFile(t, filepath.Join(stagingKey, "model.bin"), "partial")
	makeFile(t, filepath.Join(stagingRoot, "example__model-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.lock"), "{}")
	packDirectory := makeDirectory(t, filepath.Join(packs, "qwen38", "1.0.0"))
	unrelatedDir := makeDirectory(t, filepath.Join(packs, ".important"))
	unrelatedFile := filepath.Join(packs, ".import-not-a-dir")
	makeFile(t, unrelatedFile, "keep")
	noDashName := makeDirectory(t, filepath.Join(models, ".b70ctl-download"))

	removed, err := Sweep(
		Root{Path: models, Prefix: ".b70ctl-download-"},
		Root{Path: data, Prefix: ".pack-download-"},
		Root{Path: packs, Prefix: ".import-"},
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(removed) != 0 {
		t.Fatalf("Sweep() removed %#v, want nothing", removed)
	}
	assertPresent(t, finalModel)
	assertPresent(t, stagingRoot)
	assertPresent(t, stagingKey)
	assertPresent(t, packDirectory)
	assertPresent(t, unrelatedDir)
	assertPresent(t, unrelatedFile)
	assertPresent(t, noDashName)
}

func TestSweepRefusesSymlinkedEntries(t *testing.T) {
	models := t.TempDir()
	outside := t.TempDir()
	outsideFile := filepath.Join(outside, "precious.bin")
	makeFile(t, outsideFile, "keep")
	if err := os.Symlink(outside, filepath.Join(models, ".b70ctl-download-escape")); err != nil {
		t.Fatal(err)
	}

	removed, err := Sweep(Root{Path: models, Prefix: ".b70ctl-download-"})
	if err != nil {
		t.Fatal(err)
	}
	if len(removed) != 0 {
		t.Fatalf("Sweep() removed %#v, want nothing", removed)
	}
	if _, err := os.Lstat(filepath.Join(models, ".b70ctl-download-escape")); err != nil {
		t.Fatalf("symlink was removed: %v", err)
	}
	assertPresent(t, outsideFile)
}

func TestSweepSkipsMissingOrUnsafeRoots(t *testing.T) {
	models := t.TempDir()
	realRoot := t.TempDir()
	makeDirectory(t, filepath.Join(realRoot, "data"))
	symlinked := filepath.Join(models, "linked-root")
	if err := os.Symlink(realRoot, symlinked); err != nil {
		t.Fatal(err)
	}
	fileRoot := filepath.Join(models, "not-a-dir")
	makeFile(t, fileRoot, "file")

	removed, err := Sweep(
		Root{Path: filepath.Join(models, "missing"), Prefix: ".import-"},
		Root{Path: symlinked, Prefix: ".pack-download-"},
		Root{Path: fileRoot, Prefix: ".import-"},
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(removed) != 0 {
		t.Fatalf("Sweep() removed %#v, want nothing", removed)
	}
	assertPresent(t, symlinked)
	assertPresent(t, fileRoot)
}

func TestSweepNeverRemovesRootDirectories(t *testing.T) {
	models := t.TempDir()
	makeDirectory(t, filepath.Join(models, ".b70ctl-download-1"))
	makeDirectory(t, filepath.Join(models, ".b70ctl-download-2"))

	if _, err := Sweep(Root{Path: models, Prefix: ".b70ctl-download-"}); err != nil {
		t.Fatal(err)
	}
	assertPresent(t, models)
	entries, err := os.ReadDir(models)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 0 {
		t.Fatalf("root retained entries %#v", entries)
	}
}

func TestSweepRejectsInvalidPrefix(t *testing.T) {
	models := t.TempDir()
	for _, prefix := range []string{"", ".", "..", "a/b"} {
		if _, err := Sweep(Root{Path: models, Prefix: prefix}); err == nil {
			t.Fatalf("Sweep(prefix = %q) succeeded", prefix)
		}
	}
}

// TestStartupSweepAfterCrashLeftovers is the fixture-only integration
// simulation of a Controller startup after a crash: legacy throwaway temps
// are removed while resumable staging and the final model stay untouched.
func TestStartupSweepAfterCrashLeftovers(t *testing.T) {
	models := t.TempDir()
	data := t.TempDir()
	packs := t.TempDir()

	legacyDownload := makeDirectory(t, filepath.Join(models, ".b70ctl-download-abc"))
	makeFile(t, filepath.Join(legacyDownload, "model-00001-of-00002.safetensors"), "partial")
	packDownload := makeDirectory(t, filepath.Join(data, ".pack-download-def"))
	makeFile(t, filepath.Join(packDownload, "pack.tar.gz"), "partial")
	importTemp := makeDirectory(t, filepath.Join(packs, ".import-ghi"))

	stagingRoot := makeDirectory(t, filepath.Join(models, ".b70ctl-staging"))
	stagingKey := makeDirectory(t, filepath.Join(stagingRoot, "example__model-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"))
	makeFile(t, filepath.Join(stagingKey, ".b70-staging.json"), `{"schema":1,"repo":"example/model","revision":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}`)
	makeFile(t, filepath.Join(stagingKey, "model-00001-of-00002.safetensors"), "completed weights")

	finalModel := makeDirectory(t, filepath.Join(models, "example__model"))
	makeFile(t, filepath.Join(finalModel, ".b70-model.json"), `{"repo":"other/model","revision":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"}`)
	makeFile(t, filepath.Join(finalModel, "model.bin"), "model")

	removed, err := Sweep(
		Root{Path: models, Prefix: ".b70ctl-download-"},
		Root{Path: data, Prefix: ".pack-download-"},
		Root{Path: packs, Prefix: ".import-"},
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(removed) != 3 {
		t.Fatalf("Sweep() removed %#v, want the three legacy leftovers", removed)
	}
	assertGone(t, legacyDownload)
	assertGone(t, packDownload)
	assertGone(t, importTemp)
	assertPresent(t, stagingKey)
	assertPresent(t, filepath.Join(stagingKey, "model-00001-of-00002.safetensors"))
	assertPresent(t, finalModel)
	assertPresent(t, filepath.Join(finalModel, "model.bin"))
}
