package tui

import (
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"

	"b70ctl/internal/modelpack"
	"b70ctl/internal/packstore"
	"b70ctl/internal/runtime"
)

func TestQualifiedSelectionFiltering(t *testing.T) {
	manifest := &modelpack.Manifest{Modes: []modelpack.Mode{{ID: "base"}, {ID: "mtp"}, {ID: "dflash2"}}}
	profiles := []modelpack.Profile{
		{ModelID: "one", Cards: 1, Context: 32768, Mode: "base"},
		{ModelID: "one", Cards: 2, Context: 65536, Mode: "base"},
		{ModelID: "one", Cards: 2, Context: 65536, Mode: "dflash2"},
		{ModelID: "one", Cards: 4, Context: 262144, Mode: "dflash2"},
		{ModelID: "two", Cards: 1, Context: 4096, Mode: "mtp"},
	}
	if got := cardOptions(profiles, "one"); !reflect.DeepEqual(got, []int{1, 2, 4}) {
		t.Fatalf("cardOptions() = %v", got)
	}
	if got := contextOptions(profiles, "one", 2); !reflect.DeepEqual(got, []int{65536}) {
		t.Fatalf("contextOptions() = %v", got)
	}
	if got := modeOptions(manifest, profiles, "one", 2, 65536); !reflect.DeepEqual(got, []string{"base", "dflash2"}) {
		t.Fatalf("modeOptions() = %v", got)
	}
	if _, ok := exactProfile(profiles, "one", 2, 65536, "dflash2"); !ok {
		t.Fatal("exactProfile() did not resolve declared combination")
	}
	if _, ok := exactProfile(profiles, "one", 4, 65536, "dflash2"); ok {
		t.Fatal("exactProfile() manufactured an undeclared combination")
	}
}

func TestResolveRuntimeMetadata(t *testing.T) {
	packs := []loadedPack{
		{
			installed: packstore.InstalledPack{ID: "qwen", Version: "1.0.0"},
			manifest: &modelpack.Manifest{
				Models:   []modelpack.Model{{ID: "target", Name: "Qwen", Kind: "target"}},
				Profiles: []modelpack.Profile{{ID: "four-card", ModelID: "target", Cards: 4, Context: 262144, Mode: "dflash2"}},
			},
		},
	}
	status := runtime.StatusResult{
		State:       runtime.StateRunning,
		PackID:      "qwen",
		PackVersion: "1.0.0",
		ProfileID:   "four-card",
		BindAddress: "127.0.0.1",
		HostPort:    18000,
	}
	got := resolveRuntimeMetadata(status, packs)
	if got.pack == nil || got.profile == nil || got.model == nil {
		t.Fatalf("resolveRuntimeMetadata() = %#v", got)
	}
	if got.model.Name != "Qwen" || got.profile.Cards != 4 || got.profile.Context != 262144 || got.profile.Mode != "dflash2" {
		t.Fatalf("resolved metadata = %#v, %#v", got.model, got.profile)
	}
}

func TestFormatContext(t *testing.T) {
	tests := map[int]string{
		32768:  "32K",
		65536:  "64K",
		131072: "128K",
		262144: "256K",
		12345:  "12345",
	}
	for value, want := range tests {
		if got := formatContext(value); got != want {
			t.Fatalf("formatContext(%d) = %q, want %q", value, got, want)
		}
	}
}

func TestPublicQwenPackSelectorsAndModeLabels(t *testing.T) {
	root := filepath.Join("..", "..", "model-packs", "Qwen3.8-27B")
	manifest, err := modelpack.Load(root)
	if err != nil {
		t.Fatal(err)
	}
	if modeName(manifest, "base") != "Base" || modeName(manifest, "mtp") != "MTP1" || modeName(manifest, "dflash2") != "dFlash2" {
		t.Fatalf("mode labels = %q/%q/%q", modeName(manifest, "base"), modeName(manifest, "mtp"), modeName(manifest, "dflash2"))
	}
	if got := modeOptions(manifest, manifest.Profiles, "qwen38-27b-fp8", 2, 32768); !reflect.DeepEqual(got, []string{"base", "mtp", "dflash2"}) {
		t.Fatalf("mode order = %v", got)
	}
	for _, absent := range []struct {
		model      string
		cards, ctx int
		mode       string
	}{
		{"qwen38-27b-fp8", 1, 32768, "base"},
		{"qwen38-27b-fp8", 2, 131072, "base"},
		{"qwen38-27b-fp8", 2, 262144, "base"},
		{"qwen38-27b-uncensored-fp8", 1, 32768, "base"},
		{"qwen38-27b-uncensored-fp8", 2, 131072, "base"},
		{"qwen38-27b-uncensored-fp8", 2, 262144, "base"},
		{"qwen38-27b-int4", 1, 131072, "mtp"},
		{"qwen38-27b-int4", 1, 65536, "dflash2"},
		{"qwen38-27b-uncensored-int4", 1, 131072, "mtp"},
		{"qwen38-27b-uncensored-int4", 1, 65536, "dflash2"},
	} {
		if _, ok := exactProfile(manifest.Profiles, absent.model, absent.cards, absent.ctx, absent.mode); ok {
			t.Fatalf("unsupported selector row present: %#v", absent)
		}
	}
	if len(manifest.Profiles) != 96 {
		t.Fatalf("selector rows = %d", len(manifest.Profiles))
	}
	if _, ok := exactProfile(manifest.Profiles, "qwen38-27b-fp8", 2, 65536, "base"); !ok {
		t.Fatal("FP8 TP2 64K base is not selectable")
	}
	if _, ok := exactProfile(manifest.Profiles, "qwen38-27b-uncensored-fp8", 2, 65536, "base"); !ok {
		t.Fatal("uncensored FP8 TP2 64K base is not selectable")
	}
}

// TestPublicQwenPackRunModelMatrix proves the Run Model selection path against
// the public pack: buildTargets produces the four served targets, and the
// generic card/context/mode selectors expose the qualified matrix for each,
// including the uncensored checkpoints mirroring their standard counterparts.
func TestPublicQwenPackRunModelMatrix(t *testing.T) {
	root := filepath.Join("..", "..", "model-packs", "Qwen3.8-27B")
	manifest, err := modelpack.Load(root)
	if err != nil {
		t.Fatal(err)
	}
	targets := buildTargets([]loadedPack{{installed: packstore.InstalledPack{ID: manifest.ID, Version: manifest.Version}, manifest: manifest}})
	if len(targets) != 4 {
		t.Fatalf("run targets = %d, want 4", len(targets))
	}
	byModel := map[string]targetChoice{}
	labels := make([]string, 0, len(targets))
	for _, target := range targets {
		byModel[target.model.ID] = target
		labels = append(labels, target.label)
	}
	for _, modelID := range []string{"qwen38-27b-fp8", "qwen38-27b-uncensored-fp8", "qwen38-27b-int4", "qwen38-27b-uncensored-int4"} {
		if _, ok := byModel[modelID]; !ok {
			t.Fatalf("run targets missing %s: %v", modelID, labels)
		}
	}

	allModes := []string{"base", "mtp", "dflash2"}
	lane := func(context int, modes ...string) laneChoice { return laneChoice{context: context, modes: modes} }
	fp8Matrix := map[int][]laneChoice{
		2: {lane(32768, allModes...), lane(65536, allModes...)},
		4: {lane(32768, allModes...), lane(65536, allModes...), lane(131072, allModes...), lane(262144, allModes...)},
	}
	int4Matrix := map[int][]laneChoice{
		1: {lane(131072, "base"), lane(32768, allModes...), lane(65536, "base", "mtp")},
		2: {lane(32768, allModes...), lane(65536, allModes...), lane(131072, allModes...), lane(262144, allModes...)},
		4: {lane(32768, allModes...), lane(65536, allModes...), lane(131072, allModes...), lane(262144, allModes...)},
	}
	matrices := map[string]map[int][]laneChoice{
		"qwen38-27b-fp8":             fp8Matrix,
		"qwen38-27b-uncensored-fp8":  fp8Matrix,
		"qwen38-27b-int4":            int4Matrix,
		"qwen38-27b-uncensored-int4": int4Matrix,
	}

	for modelID, wantLanes := range matrices {
		target := byModel[modelID]
		wantCards := make([]int, 0, len(wantLanes))
		for cards := range wantLanes {
			wantCards = append(wantCards, cards)
		}
		sort.Ints(wantCards)
		if got := cardOptions(target.profiles, modelID); !reflect.DeepEqual(got, wantCards) {
			t.Fatalf("%s cardOptions() = %v, want %v", modelID, got, wantCards)
		}
		if strings.HasSuffix(modelID, "fp8") && wantCards[0] == 1 {
			t.Fatalf("%s offers a single-card FP8 choice: %v", modelID, wantCards)
		}
		for cards, lanes := range wantLanes {
			wantContexts := make([]int, 0, len(lanes))
			for _, choice := range lanes {
				wantContexts = append(wantContexts, choice.context)
			}
			sort.Ints(wantContexts)
			if got := contextOptions(target.profiles, modelID, cards); !reflect.DeepEqual(got, wantContexts) {
				t.Fatalf("%s TP%d contextOptions() = %v, want %v", modelID, cards, got, wantContexts)
			}
			for _, choice := range lanes {
				if got := modeOptions(manifest, target.profiles, modelID, cards, choice.context); !reflect.DeepEqual(got, choice.modes) {
					t.Fatalf("%s TP%d %s modeOptions() = %v, want %v", modelID, cards, formatContext(choice.context), got, choice.modes)
				}
				if _, ok := exactProfile(target.profiles, modelID, cards, choice.context, choice.modes[0]); !ok {
					t.Fatalf("%s TP%d %s exactProfile() missing for mode %s", modelID, cards, formatContext(choice.context), choice.modes[0])
				}
			}
		}
	}

	for _, pair := range [][2]string{
		{"qwen38-27b-uncensored-fp8", "qwen38-27b-fp8"},
		{"qwen38-27b-uncensored-int4", "qwen38-27b-int4"},
	} {
		uncensored, standard := byModel[pair[0]], byModel[pair[1]]
		uncensoredCards := cardOptions(uncensored.profiles, pair[0])
		if !reflect.DeepEqual(uncensoredCards, cardOptions(standard.profiles, pair[1])) {
			t.Fatalf("%s cards %v differ from %s %v", pair[0], uncensoredCards, pair[1], cardOptions(standard.profiles, pair[1]))
		}
		for _, cards := range uncensoredCards {
			contexts := contextOptions(uncensored.profiles, pair[0], cards)
			if want := contextOptions(standard.profiles, pair[1], cards); !reflect.DeepEqual(contexts, want) {
				t.Fatalf("%s TP%d contexts %v differ from %s %v", pair[0], cards, contexts, pair[1], want)
			}
			for _, context := range contexts {
				got := modeOptions(manifest, uncensored.profiles, pair[0], cards, context)
				want := modeOptions(manifest, standard.profiles, pair[1], cards, context)
				if !reflect.DeepEqual(got, want) {
					t.Fatalf("%s TP%d %s modes %v differ from %s %v", pair[0], cards, formatContext(context), got, pair[1], want)
				}
			}
		}
	}
}

type laneChoice struct {
	context int
	modes   []string
}
