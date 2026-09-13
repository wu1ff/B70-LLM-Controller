package tui

import (
	"errors"
	"reflect"
	"strings"
	"testing"

	"b70ctl/internal/hf"
	"b70ctl/internal/install"
	"b70ctl/internal/modelpack"
	"b70ctl/internal/runtime"
)

func TestLocalPackTargetSelectionStartsEmptyAndToggles(t *testing.T) {
	selection := localPackSelection{
		targets: []modelpack.Model{
			{ID: "one", Kind: "target"},
			{ID: "two", Kind: "target"},
		},
		selected: map[string]bool{},
	}
	if got := selection.ids(); len(got) != 0 {
		t.Fatalf("initial selection = %v", got)
	}
	selection.toggle(1)
	if got := selection.ids(); !reflect.DeepEqual(got, []string{"two"}) {
		t.Fatalf("selection after toggle = %v", got)
	}
	selection.toggle(1)
	if got := selection.ids(); len(got) != 0 {
		t.Fatalf("selection after second toggle = %v", got)
	}
}

func TestPackInstallConfirmationShowsDerivedCountsAndDefaultsToNo(t *testing.T) {
	application, output := uninstallTestApp(t, []byte{'\n'})
	manifest := &modelpack.Manifest{Name: "Test Pack", Version: "1.0.0"}
	plan := install.Plan{
		SelectedTargetIDs: []string{"target"},
		Artifacts: []install.Artifact{
			{State: install.Present},
			{State: install.Incomplete},
			{State: install.Missing},
			{State: install.Missing, SkipReason: "skip"},
		},
	}
	yes, err := application.confirmPackInstall(manifest, plan)
	if err != nil || yes {
		t.Fatalf("confirmPackInstall() = %v, %v", yes, err)
	}
	got := stripANSI(readTestOutput(t, output))
	for _, want := range []string{"SELECTED TARGETS  1", "EXISTING          1", "INCOMPLETE        1", "DOWNLOADS         1", "SKIPPED           1", "▸ NO"} {
		if !strings.Contains(got, want) {
			t.Fatalf("confirmation missing %q: %q", want, got)
		}
	}
}

func TestPartialFailureSummaryIncludesOneAccessGuidanceBlock(t *testing.T) {
	result := install.Result{Items: []install.Item{
		{Artifact: install.Artifact{Name: "Public Target"}, Outcome: install.Downloaded},
		{Artifact: install.Artifact{Name: "Gated Target"}, Outcome: install.Failed, Reason: "Hugging Face access denied", Err: hf.ErrAccessDenied},
		{Artifact: install.Artifact{Name: "Other Target"}, Outcome: install.Failed, Reason: "Hugging Face access denied", Err: errors.Join(hf.ErrAccessDenied, errors.New("403"))},
	}}
	got := strings.Join(packInstallSummary(result, true), "\n")
	if strings.Count(got, "Hugging Face access issue") != 1 {
		t.Fatalf("access guidance count in summary = %d: %q", strings.Count(got, "Hugging Face access issue"), got)
	}
	for _, want := range []string{"Public Target\nDownloaded", "Gated Target\nFailed — Hugging Face access denied", "2 artifacts could not be acquired.", "- your token is valid", "Missing models can be retried later from Models."} {
		if !strings.Contains(got, want) {
			t.Fatalf("summary missing %q: %q", want, got)
		}
	}
}

func TestAccessFailureSummaryWithoutTokenPointsToSettings(t *testing.T) {
	result := install.Result{Items: []install.Item{
		{Artifact: install.Artifact{Name: "Uncensored Target"}, Outcome: install.Failed, Reason: "Hugging Face token not configured", Err: hf.ErrAccessUncertain},
	}}
	got := strings.Join(packInstallSummary(result, false), "\n")
	for _, want := range []string{
		"Uncensored Target\nFailed — Hugging Face token not configured",
		"No Hugging Face token is configured.",
		"Set one under Settings → Hugging Face Token, then retry.",
		"Missing models can be retried later from Models.",
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("summary missing %q: %q", want, got)
		}
	}
	if strings.Contains(got, "your token is valid") {
		t.Fatalf("no-token summary told the user to check a token: %q", got)
	}
}

func TestPackOnlyInstallSummary(t *testing.T) {
	got := strings.Join(packInstallSummary(install.Result{}, false), "\n")
	if !strings.Contains(got, "No models were downloaded.\nYou can add them later from Models.") {
		t.Fatalf("pack-only summary = %q", got)
	}
}

func TestPackInstallSummaryIncludesRuntimeResultAndRetryGuidance(t *testing.T) {
	result := install.Result{
		Items: []install.Item{{Artifact: install.Artifact{Name: "Target"}, Outcome: install.Downloaded}},
		RuntimeItems: []install.RuntimeItem{{
			Runtime: modelpack.Runtime{ID: "runtime"}, Outcome: runtime.AcquisitionFailed, Reason: "Docker pull failed",
		}},
	}
	got := strings.Join(packInstallSummary(result, false), "\n")
	for _, want := range []string{
		"Target\nDownloaded", "Runtime\nFailed — Docker pull failed",
		"Models can still be managed from Models.", "Runtime can be retried later from Installed Packs.",
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("summary missing %q: %q", want, got)
		}
	}
}
