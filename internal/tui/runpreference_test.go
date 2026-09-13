package tui

import (
	"path/filepath"
	"reflect"
	"testing"

	"b70ctl/internal/config"
	"b70ctl/internal/modelpack"
	"b70ctl/internal/packstore"
)

func preferenceTargets() []targetChoice {
	manifest := &modelpack.Manifest{
		Modes: []modelpack.Mode{{ID: "base"}, {ID: "dflash2"}},
		Models: []modelpack.Model{
			{ID: "model-a", Name: "Model A", Kind: "target"},
			{ID: "model-b", Name: "Model B", Kind: "target"},
		},
		Profiles: []modelpack.Profile{
			{ID: "a-1", ModelID: "model-a", Cards: 1, Context: 32768, Mode: "base"},
			{ID: "a-2", ModelID: "model-a", Cards: 2, Context: 65536, Mode: "base"},
			{ID: "a-3", ModelID: "model-a", Cards: 2, Context: 65536, Mode: "dflash2"},
			{ID: "b-1", ModelID: "model-b", Cards: 2, Context: 32768, Mode: "base"},
			{ID: "b-2", ModelID: "model-b", Cards: 4, Context: 262144, Mode: "dflash2"},
		},
	}
	return buildTargets([]loadedPack{{installed: packstore.InstalledPack{ID: "pack", Version: "1.0.0"}, manifest: manifest}})
}

// defaultSelection mirrors the Run Model entry state before any persisted
// preference is applied: target 0 with its normal cards/context/mode defaults.
func defaultSelection(t *testing.T, targets []targetChoice, availableCards int) runSelection {
	t.Helper()
	selection := runSelection{target: 0, access: config.AccessLocal, port: 8000}
	application := &app{}
	application.resetSelection(&selection, targets[0], availableCards)
	return selection
}

func applyPreference(t *testing.T, preference *config.RunSelection, targets []targetChoice, availableCards int) runSelection {
	t.Helper()
	selection := defaultSelection(t, targets, availableCards)
	application := &app{config: config.Config{LastRun: preference}}
	application.restoreLastRun(&selection, targets, availableCards)
	return selection
}

func TestRestoreLastRunWithoutPreferenceKeepsDefaults(t *testing.T) {
	targets := preferenceTargets()
	selection := applyPreference(t, nil, targets, 4)
	want := defaultSelection(t, targets, 4)
	if !reflect.DeepEqual(selection, want) {
		t.Fatalf("selection without preference = %#v, want untouched defaults %#v", selection, want)
	}
	if want.target != 0 || want.cards != 1 || want.context != 32768 || want.mode != "base" {
		t.Fatalf("unexpected baseline defaults: %#v", want)
	}
}

func TestRestoreLastRunZeroValuesKeepDefaults(t *testing.T) {
	targets := preferenceTargets()
	selection := applyPreference(t, &config.RunSelection{}, targets, 4)
	want := defaultSelection(t, targets, 4)
	if !reflect.DeepEqual(selection, want) {
		t.Fatalf("selection for zero preference = %#v, want defaults %#v", selection, want)
	}
}

func TestRestoreLastRunRestoresValidTuple(t *testing.T) {
	targets := preferenceTargets()
	selection := applyPreference(t, &config.RunSelection{ModelID: "model-b", Cards: 4, Context: 262144, Mode: "dflash2"}, targets, 4)
	if selection.target != 1 {
		t.Fatalf("target = %d, want 1", selection.target)
	}
	if selection.cards != 4 || selection.context != 262144 || selection.mode != "dflash2" {
		t.Fatalf("restored selection = %#v, want 4/262144/dflash2", selection)
	}
	if _, ok := exactProfile(targets[selection.target].profiles, "model-b", selection.cards, selection.context, selection.mode); !ok {
		t.Fatalf("restored selection does not resolve to a declared profile: %#v", selection)
	}
}

func TestRestoreLastRunMissingModelFallsBackToDefaults(t *testing.T) {
	targets := preferenceTargets()
	selection := applyPreference(t, &config.RunSelection{ModelID: "removed", Cards: 4, Context: 262144, Mode: "dflash2"}, targets, 4)
	want := defaultSelection(t, targets, 4)
	if !reflect.DeepEqual(selection, want) {
		t.Fatalf("selection for missing model = %#v, want defaults %#v", selection, want)
	}
}

func TestRestoreLastRunStaleCardsDropsDependentsToDefaults(t *testing.T) {
	targets := preferenceTargets()
	// model-b still exists but its remembered 8-card choice is gone; cards
	// fall back to the normal default for model-b and context/mode resolve
	// from that default rather than the remembered values.
	selection := applyPreference(t, &config.RunSelection{ModelID: "model-b", Cards: 8, Context: 262144, Mode: "dflash2"}, targets, 4)
	if selection.target != 1 {
		t.Fatalf("target = %d, want restored 1", selection.target)
	}
	if selection.cards != 2 {
		t.Fatalf("cards = %d, want default 2", selection.cards)
	}
	if selection.context != 32768 || selection.mode != "base" {
		t.Fatalf("dependent selection = %d/%s, want defaults 32768/base", selection.context, selection.mode)
	}
	if _, ok := exactProfile(targets[selection.target].profiles, "model-b", selection.cards, selection.context, selection.mode); !ok {
		t.Fatalf("fallback selection does not resolve to a declared profile: %#v", selection)
	}
}

func TestRestoreLastRunRestoresCardsAndModeAcrossMissingContext(t *testing.T) {
	targets := preferenceTargets()
	// model-a still offers 2 cards, but not at the remembered context; cards
	// are restored, context falls back to the default for that card choice,
	// and the remembered mode is still honored because model-a declares it
	// for the defaulted context.
	selection := applyPreference(t, &config.RunSelection{ModelID: "model-a", Cards: 2, Context: 131072, Mode: "dflash2"}, targets, 4)
	if selection.target != 0 || selection.cards != 2 {
		t.Fatalf("target/cards = %d/%d, want 0/2", selection.target, selection.cards)
	}
	if selection.context != 65536 {
		t.Fatalf("context = %d, want default 65536", selection.context)
	}
	if selection.mode != "dflash2" {
		t.Fatalf("mode = %q, want restored dflash2", selection.mode)
	}
	if _, ok := exactProfile(targets[selection.target].profiles, "model-a", selection.cards, selection.context, selection.mode); !ok {
		t.Fatalf("partial restore does not resolve to a declared profile: %#v", selection)
	}
}

func TestRestoreLastRunIgnoresRememberedModeMissingForResolvedContext(t *testing.T) {
	targets := preferenceTargets()
	// model-a at 1 card only declares base, so the remembered dflash2 mode
	// is dropped even though cards and context restore cleanly.
	selection := applyPreference(t, &config.RunSelection{ModelID: "model-a", Cards: 1, Context: 32768, Mode: "dflash2"}, targets, 4)
	if selection.cards != 1 || selection.context != 32768 || selection.mode != "base" {
		t.Fatalf("selection = %#v, want 1/32768/base", selection)
	}
}

func TestRestoreLastRunRestoresCardsBeyondDetectedHardware(t *testing.T) {
	targets := preferenceTargets()
	// Pack membership decides existence; detected-card availability keeps
	// being surfaced by the existing profile-unavailable UI.
	selection := applyPreference(t, &config.RunSelection{ModelID: "model-a", Cards: 2, Context: 65536, Mode: "dflash2"}, targets, 1)
	if selection.cards != 2 || selection.context != 65536 || selection.mode != "dflash2" {
		t.Fatalf("selection = %#v, want 2/65536/dflash2", selection)
	}
}

func TestSaveRunSelectionPersistsTupleOnStartPath(t *testing.T) {
	targets := preferenceTargets()
	path := filepath.Join(t.TempDir(), "config.json")
	application := &app{
		config: config.Config{ModelDirectory: "/srv/models", DefaultAccess: config.AccessLAN, DefaultPort: 18000},
		paths:  config.Paths{ConfigFile: path},
	}
	selection := runSelection{target: 1, cards: 4, context: 262144, mode: "dflash2", access: config.AccessLAN, port: 18000}
	application.saveRunSelection(targets[1], selection)

	saved, err := config.Load(path)
	if err != nil {
		t.Fatal(err)
	}
	want := &config.RunSelection{ModelID: "model-b", Cards: 4, Context: 262144, Mode: "dflash2"}
	if !reflect.DeepEqual(saved.LastRun, want) {
		t.Fatalf("persisted last run = %#v, want %#v", saved.LastRun, want)
	}
	if saved.ModelDirectory != "/srv/models" || saved.DefaultAccess != config.AccessLAN || saved.DefaultPort != 18000 {
		t.Fatalf("save clobbered existing settings: %#v", saved)
	}
	if application.message != "" {
		t.Fatalf("successful save reported %q", application.message)
	}

	// The saved tuple restores through the same entry path Run Model uses.
	restored := applyPreference(t, saved.LastRun, targets, 4)
	if restored.target != 1 || restored.cards != 4 || restored.context != 262144 || restored.mode != "dflash2" {
		t.Fatalf("round-tripped selection = %#v, want model-b 4/262144/dflash2", restored)
	}
}
