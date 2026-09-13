package runtime

import (
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"b70ctl/internal/modelpack"
	"b70ctl/internal/modelstore"
)

func TestResolveAvailabilityBlockers(t *testing.T) {
	manifest := testManifest()
	present := testInventory(t)
	runtimes := testRuntimeResults()

	t.Run("base profile available with target and runtime", func(t *testing.T) {
		got := Resolve(manifest, present[:1], runtimes, 4)
		assertAvailability(t, got[0], true, nil)
	})

	t.Run("missing target blocks every dependent profile", func(t *testing.T) {
		got := Resolve(manifest, nil, runtimes, 4)
		assertAvailability(t, got[0], false, []string{"target model"})
		assertAvailability(t, got[1], false, []string{"target model", "assistant draft"})
	})

	t.Run("missing assistant blocks only its profile", func(t *testing.T) {
		got := Resolve(manifest, present[:1], runtimes, 4)
		assertAvailability(t, got[0], true, nil)
		assertAvailability(t, got[1], false, []string{"assistant draft"})
	})

	t.Run("missing runtime blocks dependent profile", func(t *testing.T) {
		missingRuntime := testRuntimeResults()
		missingRuntime["draft"] = ImageResult{Status: ImageMissing}
		got := Resolve(manifest, present, missingRuntime, 4)
		assertAvailability(t, got[0], true, nil)
		assertAvailability(t, got[1], false, []string{"runtime draft"})
	})

	t.Run("wrong model revision remains missing", func(t *testing.T) {
		wrong := append([]modelstore.Artifact(nil), present...)
		wrong[0].Revision = "wrong"
		got := Resolve(manifest, wrong, runtimes, 4)
		assertAvailability(t, got[0], false, []string{"target model"})
	})

	t.Run("incomplete target blocks every dependent profile", func(t *testing.T) {
		incomplete := testInventory(t)
		if err := os.Remove(filepath.Join(incomplete[0].Path, "model.bin")); err != nil {
			t.Fatal(err)
		}
		got := Resolve(manifest, incomplete, runtimes, 4)
		assertAvailability(t, got[0], false, []string{"target model incomplete"})
		assertAvailability(t, got[1], false, []string{"target model incomplete"})
	})

	t.Run("incomplete assistant blocks only its profile", func(t *testing.T) {
		incomplete := testInventory(t)
		if err := os.Remove(filepath.Join(incomplete[1].Path, "model.bin")); err != nil {
			t.Fatal(err)
		}
		got := Resolve(manifest, incomplete, runtimes, 4)
		assertAvailability(t, got[0], true, nil)
		assertAvailability(t, got[1], false, []string{"assistant draft incomplete"})
	})
}

func TestResolveCardCapacityAllowsSmallerProfiles(t *testing.T) {
	manifest := testManifest()
	manifest.Profiles = []modelpack.Profile{
		testProfile("one", 1),
		testProfile("two", 2),
		testProfile("four", 4),
	}

	tests := []struct {
		available int
		want      []bool
		missing   []string
	}{
		{available: 4, want: []bool{true, true, true}},
		{available: 2, want: []bool{true, true, false}, missing: []string{"requires 4 cards"}},
		{available: 1, want: []bool{true, false, false}, missing: []string{"requires 2 cards", "requires 4 cards"}},
	}

	for _, test := range tests {
		got := Resolve(manifest, testInventory(t), testRuntimeResults(), test.available)
		for i, want := range test.want {
			if got[i].Available != want {
				t.Fatalf("Resolve(availableCards=%d)[%d].Available = %v, want %v", test.available, i, got[i].Available, want)
			}
		}
		var missing []string
		for _, result := range got {
			missing = append(missing, result.Missing...)
		}
		if !reflect.DeepEqual(missing, test.missing) {
			t.Fatalf("Resolve(availableCards=%d) missing = %v, want %v", test.available, missing, test.missing)
		}
	}
}

func TestResolveSharedRuntimeAndDeterministicResults(t *testing.T) {
	manifest := testManifest()
	manifest.Profiles[1].RuntimeID = "base"

	first := Resolve(manifest, testInventory(t), map[string]ImageResult{
		"base": {Status: ImagePresent},
	}, 4)
	second := Resolve(manifest, testInventory(t), map[string]ImageResult{
		"base": {Status: ImagePresent},
	}, 4)

	if !reflect.DeepEqual(first, second) {
		t.Fatalf("Resolve() is not deterministic:\nfirst  = %#v\nsecond = %#v", first, second)
	}
	for _, result := range first {
		assertAvailability(t, result, true, nil)
	}
	if first[0].ProfileID != "base-1" || first[0].ModelID != "target" || first[0].ModelName != "Target Model" || first[0].Cards != 1 || first[0].Context != 4096 || first[0].Mode != "base" {
		t.Fatalf("Resolve() profile fields = %#v", first[0])
	}
}

func testManifest() *modelpack.Manifest {
	digest := "sha256:" + strings.Repeat("a", 64)
	return &modelpack.Manifest{
		Models: []modelpack.Model{
			{ID: "target", Name: "Target Model", Kind: "target", Repo: "example/target", Revision: "target-rev", Files: []modelpack.ModelFile{{Path: "model.bin"}}},
			{ID: "draft", Name: "Draft Model", Kind: "assistant", Repo: "example/draft", Revision: "draft-rev", Files: []modelpack.ModelFile{{Path: "model.bin"}}},
		},
		Runtimes: []modelpack.Runtime{
			{ID: "base", Image: "example/base:1", Digest: digest},
			{ID: "draft", Image: "example/draft:1", Digest: digest},
		},
		Modes: []modelpack.Mode{
			{ID: "base", DisplayName: "Base", Assistants: []string{}},
			{ID: "dflash2", DisplayName: "dFlash2", Assistants: []string{"draft"}},
		},
		Profiles: []modelpack.Profile{
			testProfile("base-1", 1),
			{
				ID: "draft-1", ModelID: "target", RuntimeID: "draft", Cards: 1,
				Context: 4096, Mode: "dflash2",
			},
		},
	}
}

func testProfile(id string, cards int) modelpack.Profile {
	return modelpack.Profile{
		ID: id, ModelID: "target", RuntimeID: "base", Cards: cards,
		Context: 4096, Mode: "base",
	}
}

func testInventory(t *testing.T) []modelstore.Artifact {
	t.Helper()
	root := t.TempDir()
	target := filepath.Join(root, "target")
	draft := filepath.Join(root, "draft")
	for _, path := range []string{target, draft} {
		if err := os.MkdirAll(path, 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(path, "model.bin"), []byte("model"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return []modelstore.Artifact{
		{Repo: "example/target", Revision: "target-rev", Path: target},
		{Repo: "example/draft", Revision: "draft-rev", Path: draft},
	}
}

func testRuntimeResults() map[string]ImageResult {
	return map[string]ImageResult{
		"base":  {Status: ImagePresent},
		"draft": {Status: ImagePresent},
	}
}

func assertAvailability(t *testing.T, got ProfileAvailability, available bool, missing []string) {
	t.Helper()
	if got.Available != available || !reflect.DeepEqual(got.Missing, missing) {
		t.Fatalf("availability = %v, missing = %v; want %v, %v", got.Available, got.Missing, available, missing)
	}
}
