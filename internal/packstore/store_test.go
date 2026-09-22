package packstore

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"b70ctl/internal/modelpack"
)

func TestImportCopiesOnlyRequiredFiles(t *testing.T) {
	source := t.TempDir()
	store := t.TempDir()
	manifest := testManifest("qwen-test", "Qwen Test Pack", "1.0.0")
	shared := manifest.Profiles[0]
	shared.ID = "base-2"
	shared.Context = 8192
	manifest.Profiles = append(manifest.Profiles, shared)
	writeTestPack(t, source, manifest)
	writeFile(t, filepath.Join(source, "validate.py"), "unused\n")
	writeFile(t, filepath.Join(source, "random.bin"), "junk\n")

	installed, err := Import(source, store, SourceLocal)
	if err != nil {
		t.Fatalf("Import() error = %v", err)
	}
	if *installed != (InstalledPack{ID: "qwen-test", Name: "Qwen Test Pack", Version: "1.0.0", Source: "local"}) {
		t.Fatalf("Import() = %#v", installed)
	}

	destination := filepath.Join(store, "qwen-test", "1.0.0")
	if err := os.RemoveAll(source); err != nil {
		t.Fatal(err)
	}
	if _, err := modelpack.Load(destination); err != nil {
		t.Fatalf("installed pack does not survive source removal: %v", err)
	}

	wantFiles := []string{"README.md", "pack.json", "source.json"}
	if got := regularFiles(t, destination); !reflect.DeepEqual(got, wantFiles) {
		t.Fatalf("installed files = %v, want %v", got, wantFiles)
	}
	var sourceRecord map[string]string
	data, err := os.ReadFile(filepath.Join(destination, "source.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(data, &sourceRecord); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(sourceRecord, map[string]string{"source": "local"}) {
		t.Fatalf("source.json = %v", sourceRecord)
	}
}

func TestImportPublicQwenPack(t *testing.T) {
	source := filepath.Join("..", "..", "model-packs", "Qwen3.8-27B")
	store := t.TempDir()
	installed, err := Import(source, store, SourceLocal)
	if err != nil {
		t.Fatalf("Import() error = %v", err)
	}
	if *installed != (InstalledPack{ID: "qwen38-27b-b70", Name: "Qwen3.8 27B B70 Pack (MTP1 / dFlash2)", Version: "1.0.3", Source: SourceLocal}) {
		t.Fatalf("Import() = %#v", installed)
	}
	manifest, err := modelpack.Load(filepath.Join(store, installed.ID, installed.Version))
	if err != nil {
		t.Fatal(err)
	}
	if len(manifest.Models) != 5 || len(manifest.Runtimes) != 1 || len(manifest.Modes) != 3 || len(manifest.Profiles) != 96 {
		t.Fatalf("installed pack counts = %d/%d/%d/%d", len(manifest.Models), len(manifest.Runtimes), len(manifest.Modes), len(manifest.Profiles))
	}
}

func TestImportRejectsInvalidSource(t *testing.T) {
	source := t.TempDir()
	store := t.TempDir()
	manifest := testManifest("invalid-pack", "Invalid Pack", "1.0.0")
	writeTestPack(t, source, manifest)
	if err := os.Remove(filepath.Join(source, "README.md")); err != nil {
		t.Fatal(err)
	}

	_, err := Import(source, store, SourceLocal)
	if err == nil || !strings.Contains(err.Error(), "pack README.md is required") {
		t.Fatalf("Import() error = %v", err)
	}
	assertPathMissing(t, filepath.Join(store, manifest.ID, manifest.Version))
}

func TestImportRejectsDuplicate(t *testing.T) {
	source := t.TempDir()
	store := t.TempDir()
	manifest := testManifest("qwen-test", "Qwen Test Pack", "1.0.0")
	writeTestPack(t, source, manifest)
	if _, err := Import(source, store, SourceLocal); err != nil {
		t.Fatal(err)
	}

	_, err := Import(source, store, SourceLocal)
	want := "pack qwen-test 1.0.0 is already installed"
	if err == nil || err.Error() != want {
		t.Fatalf("Import() error = %v, want %q", err, want)
	}
}

func TestImportReplacesPackWithInvalidManifest(t *testing.T) {
	store := t.TempDir()
	destination := filepath.Join(store, "qwen-test", "1.0.0")
	writeFile(t, filepath.Join(destination, "pack.json"), "{\"schema_version\":1}\n")
	writeFile(t, filepath.Join(destination, "recipe.json"), "{}\n")

	source := t.TempDir()
	manifest := testManifest("qwen-test", "Qwen Test Pack", "1.0.0")
	writeTestPack(t, source, manifest)

	installed, err := Import(source, store, SourceLocal)
	if err != nil {
		t.Fatalf("Import() error = %v", err)
	}
	if *installed != (InstalledPack{ID: "qwen-test", Name: "Qwen Test Pack", Version: "1.0.0", Source: "local"}) {
		t.Fatalf("Import() = %#v", installed)
	}
	if _, err := modelpack.Load(destination); err != nil {
		t.Fatalf("replaced pack does not load: %v", err)
	}
	wantFiles := []string{"README.md", "pack.json", "source.json"}
	if got := regularFiles(t, destination); !reflect.DeepEqual(got, wantFiles) {
		t.Fatalf("installed files = %v, want %v", got, wantFiles)
	}
}

func TestImportReplacesPackWithInvalidProvenance(t *testing.T) {
	store := t.TempDir()
	destination := filepath.Join(store, "qwen-test", "1.0.0")
	writeTestPack(t, destination, testManifest("qwen-test", "Qwen Test Pack", "1.0.0"))
	writeFile(t, filepath.Join(destination, "source.json"), "{\"source\":\"unknown\"}\n")

	source := t.TempDir()
	writeTestPack(t, source, testManifest("qwen-test", "Qwen Test Pack", "1.0.0"))

	installed, err := Import(source, store, SourceLocal)
	if err != nil {
		t.Fatalf("Import() error = %v", err)
	}
	if installed.ID != "qwen-test" || installed.Source != SourceLocal {
		t.Fatalf("Import() = %#v", installed)
	}
	listed, err := List(store)
	if err != nil || len(listed) != 1 || listed[0].Source != SourceLocal {
		t.Fatalf("List() = %#v, %v", listed, err)
	}
}

func TestImportRejectsVersionThatEscapesStore(t *testing.T) {
	source := t.TempDir()
	store := t.TempDir()
	manifest := testManifest("qwen-test", "Qwen Test Pack", "../outside")
	writeTestPack(t, source, manifest)

	_, err := Import(source, store, SourceLocal)
	if err == nil || !strings.Contains(err.Error(), "directory name") {
		t.Fatalf("Import() error = %v", err)
	}
	assertPathMissing(t, filepath.Join(store, "outside"))
}

func TestListNonexistentStoreIsEmpty(t *testing.T) {
	installed, err := List(filepath.Join(t.TempDir(), "missing"))
	if err != nil {
		t.Fatalf("List() error = %v", err)
	}
	if len(installed) != 0 {
		t.Fatalf("List() = %v", installed)
	}
}

func TestListReturnsInstalledPacksInLexicalOrderAndSkipsJunk(t *testing.T) {
	store := t.TempDir()
	imports := []modelpack.Manifest{
		testManifest("zeta", "Zeta", "1.0.0"),
		testManifest("alpha", "Alpha Two", "2.0.0"),
		testManifest("alpha", "Alpha One", "1.0.0"),
	}
	for _, manifest := range imports {
		source := t.TempDir()
		writeTestPack(t, source, manifest)
		if _, err := Import(source, store, SourceLocal); err != nil {
			t.Fatalf("Import() error = %v", err)
		}
	}
	invalidSource := testManifest("invalid-source", "Invalid Source", "1.0.0")
	invalidSourceDirectory := t.TempDir()
	writeTestPack(t, invalidSourceDirectory, invalidSource)
	if _, err := Import(invalidSourceDirectory, store, SourceLocal); err != nil {
		t.Fatalf("Import() error = %v", err)
	}
	writeFile(t, filepath.Join(store, "invalid-source", "1.0.0", "source.json"), `{"source":"unknown"}`)
	writeFile(t, filepath.Join(store, "loose-file"), "junk\n")
	writeFile(t, filepath.Join(store, "junk", "not-a-pack", "README.md"), "junk\n")
	writeFile(t, filepath.Join(store, "broken", "1.0.0", "source.json"), `{"source":"local"}`)

	installed, err := List(store)
	if err != nil {
		t.Fatalf("List() error = %v", err)
	}
	want := []InstalledPack{
		{ID: "alpha", Name: "Alpha One", Version: "1.0.0", Source: "local"},
		{ID: "alpha", Name: "Alpha Two", Version: "2.0.0", Source: "local"},
		{ID: "zeta", Name: "Zeta", Version: "1.0.0", Source: "local"},
	}
	if !reflect.DeepEqual(installed, want) {
		t.Fatalf("List() = %#v, want %#v", installed, want)
	}
}

func TestImportRecordsRemoteSourceAndRejectsUnknownSource(t *testing.T) {
	source := t.TempDir()
	writeTestPack(t, source, testManifest("remote-pack", "Remote Pack", "1.0.0"))
	store := t.TempDir()

	installed, err := Import(source, store, SourceRemote)
	if err != nil {
		t.Fatal(err)
	}
	if installed.Source != SourceRemote {
		t.Fatalf("source = %q", installed.Source)
	}
	listed, err := List(store)
	if err != nil || len(listed) != 1 || listed[0].Source != SourceRemote {
		t.Fatalf("List() = %#v, %v", listed, err)
	}
	if _, err := Import(source, t.TempDir(), "unknown"); err == nil {
		t.Fatal("Import accepted an unknown source")
	}
}

func TestRemoveDeletesOnlyExactInstalledPackAndEmptyParent(t *testing.T) {
	store := t.TempDir()
	for _, version := range []string{"1.0.0", "2.0.0"} {
		source := t.TempDir()
		writeTestPack(t, source, testManifest("qwen-test", "Qwen Test Pack", version))
		if _, err := Import(source, store, SourceLocal); err != nil {
			t.Fatal(err)
		}
	}
	otherSource := t.TempDir()
	writeTestPack(t, otherSource, testManifest("other-pack", "Other Pack", "1.0.0"))
	if _, err := Import(otherSource, store, SourceLocal); err != nil {
		t.Fatal(err)
	}

	if err := Remove(store, "qwen-test", "1.0.0"); err != nil {
		t.Fatalf("Remove() error = %v", err)
	}
	assertPathMissing(t, filepath.Join(store, "qwen-test", "1.0.0"))
	if _, err := os.Stat(filepath.Join(store, "qwen-test", "2.0.0")); err != nil {
		t.Fatalf("sibling version was removed: %v", err)
	}
	if _, err := os.Stat(filepath.Join(store, "other-pack", "1.0.0")); err != nil {
		t.Fatalf("other pack was removed: %v", err)
	}

	if err := Remove(store, "qwen-test", "2.0.0"); err != nil {
		t.Fatalf("Remove() error = %v", err)
	}
	assertPathMissing(t, filepath.Join(store, "qwen-test"))
}

func TestRemoveRejectsMissingAndUnsafePackPaths(t *testing.T) {
	store := t.TempDir()
	if err := Remove(store, "missing", "1.0.0"); err == nil || err.Error() != "pack missing 1.0.0 is not installed" {
		t.Fatalf("Remove() error = %v", err)
	}
	if err := Remove(store, "../outside", "1.0.0"); err == nil {
		t.Fatal("Remove() accepted an escaping pack ID")
	}

	external := t.TempDir()
	writeFile(t, filepath.Join(external, "keep.txt"), "keep\n")
	if err := os.Symlink(external, filepath.Join(store, "linked-pack")); err != nil {
		t.Fatal(err)
	}
	if err := Remove(store, "linked-pack", "1.0.0"); err == nil {
		t.Fatal("Remove() accepted a symlinked pack directory")
	}
	if _, err := os.Stat(filepath.Join(external, "keep.txt")); err != nil {
		t.Fatalf("external file was removed: %v", err)
	}
}

func testManifest(id, name, version string) modelpack.Manifest {
	empty := modelpack.LaunchBlock{DockerArgs: []string{}, Environment: map[string]string{}, CommandArgs: []string{}}
	return modelpack.Manifest{
		SchemaVersion: 1,
		ID:            id,
		Name:          name,
		Version:       version,
		Models: []modelpack.Model{
			{ID: "target", Name: "Target", Kind: "target", Repo: "example/target", Revision: "revision-1", Gated: false, Files: []modelpack.ModelFile{{Path: "model.bin"}}, MountPath: "/models/target", Launch: empty},
		},
		Runtimes: []modelpack.Runtime{
			{ID: "runtime", Image: "example/runtime", Digest: "sha256:" + strings.Repeat("a", 64), ContainerPort: 8000, HealthPath: "/health", Launch: modelpack.RuntimeLaunch{DockerArgs: []string{}, Environment: map[string]string{}, Command: []string{"serve"}}},
		},
		Modes: []modelpack.Mode{{ID: "base", DisplayName: "Base", Assistants: []string{}, Launch: empty}},
		Profiles: []modelpack.Profile{
			{
				ID: "base-1", ModelID: "target", RuntimeID: "runtime", Cards: 1,
				TensorParallel: 1, Context: 4096, Mode: "base", Launch: empty,
			},
		},
	}
}

func writeTestPack(t *testing.T, directory string, manifest modelpack.Manifest) {
	t.Helper()
	writeFile(t, filepath.Join(directory, "README.md"), "# Test pack\n")
	data, err := json.MarshalIndent(manifest, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(directory, "pack.json"), data, 0o644); err != nil {
		t.Fatal(err)
	}
}

func writeFile(t *testing.T, path, contents string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(contents), 0o644); err != nil {
		t.Fatal(err)
	}
}

func regularFiles(t *testing.T, root string) []string {
	t.Helper()
	var files []string
	if err := filepath.WalkDir(root, func(path string, entry os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if entry.Type().IsRegular() {
			relative, err := filepath.Rel(root, path)
			if err != nil {
				return err
			}
			files = append(files, relative)
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	return files
}

func assertPathMissing(t *testing.T, path string) {
	t.Helper()
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatalf("path %q exists or returned unexpected error: %v", path, err)
	}
}
