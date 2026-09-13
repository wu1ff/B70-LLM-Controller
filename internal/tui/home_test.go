package tui

import (
	"reflect"
	"strings"
	"testing"

	"b70ctl/internal/config"
	"b70ctl/internal/modelpack"
	"b70ctl/internal/runtime"
)

func TestB70WordmarkExactLines(t *testing.T) {
	want := []string{
		" ::::::::   ::::::::    :::::::: ",
		":+:    :+: :+:    :+:  :+:    :+:",
		"+:+    +:+      +:+   +:+      +:+",
		"+#++:++#+     +#+     +#+      +#+",
		"+#+    +#+   +#+      +#+      +#+",
		"#+#    #+#  #+#       #+#      #+#",
		"#########   ###         ########",
	}
	if got := strings.Split(b70Wordmark, "\n"); !reflect.DeepEqual(got, want) {
		t.Fatalf("b70Wordmark lines = %#v, want %#v", got, want)
	}
	rendered := renderB70Wordmark()
	for index := range rendered {
		if got := stripANSI(rendered[index]); got != want[index] {
			t.Fatalf("renderB70Wordmark()[%d] = %q, want %q", index, got, want[index])
		}
	}
	colored := strings.Join(rendered, "\n")
	for sequence, character := range map[string]string{dimCyan + ":": ":", blue + "+": "+", icyBlue + "#": "#"} {
		if !strings.Contains(colored, sequence) {
			t.Fatalf("wordmark does not apply the required style to %q", character)
		}
	}
}

func TestB70WordmarkLowerGlyphAlignment(t *testing.T) {
	lines := strings.Split(b70Wordmark, "\n")
	penultimateSeven := strings.Index(lines[5][10:], "#+#") + 10
	baseSeven := strings.Index(lines[6][10:], "###") + 10
	if penultimateSeven != baseSeven {
		t.Fatalf("7 lower stroke starts at %d but base starts at %d", penultimateSeven, baseSeven)
	}
	if zeroOuter := strings.Index(lines[5][20:], "#+#") + 20; zeroOuter != 22 {
		t.Fatalf("0 outer stroke starts at %d, want 22", zeroOuter)
	}
	for row, rightOuter := range []int{
		strings.LastIndex(lines[2], "+:+"),
		strings.LastIndex(lines[3], "+#+"),
		strings.LastIndex(lines[4], "+#+"),
		strings.LastIndex(lines[5], "#+#"),
	} {
		if rightOuter != 31 {
			t.Fatalf("0 right outer stroke on row %d starts at %d, want 31", row+3, rightOuter)
		}
	}
	if bottomZero := strings.Index(lines[6][20:], "########") + 20; bottomZero != 24 {
		t.Fatalf("0 bottom stroke starts at %d, want 24", bottomZero)
	}
}

func TestHomeMenuOrderAndNavigation(t *testing.T) {
	want := []string{"RUN MODEL", "RUNTIME", "MODELS", "MODEL PACKS", "SETTINGS", "EXIT"}
	got := make([]string, len(homeMenuItems))
	for index, item := range homeMenuItems {
		got[index] = item.title
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("home menu = %v, want %v", got, want)
	}
	if next(0, len(homeMenuItems)) != 1 || previous(1, len(homeMenuItems)) != 0 || previous(0, len(homeMenuItems)) != 5 {
		t.Fatal("home up/down navigation semantics changed")
	}
}

func TestLastRunSummaryValidAndReadOnly(t *testing.T) {
	preference := &config.RunSelection{ModelID: "qwen", Cards: 2, Context: 65536, Mode: "dflash2"}
	wantPreference := *preference
	manifest := &modelpack.Manifest{
		Models:   []modelpack.Model{{ID: "qwen", Name: "Qwen3.8 27B Uncensored FP8"}},
		Modes:    []modelpack.Mode{{ID: "dflash2", DisplayName: "dFlash2"}},
		Profiles: []modelpack.Profile{{ModelID: "qwen", Cards: 2, Context: 65536, Mode: "dflash2"}},
	}
	got := lastRunSummary(preference, []loadedPack{{manifest: manifest}})
	want := "Qwen3.8 27B Uncensored FP8 · TP2 · 64K · dFlash2"
	if got != want {
		t.Fatalf("lastRunSummary() = %q, want %q", got, want)
	}
	if !reflect.DeepEqual(*preference, wantPreference) {
		t.Fatalf("lastRunSummary() mutated preference: got %#v, want %#v", *preference, wantPreference)
	}
	if got := lastRunSummary(&config.RunSelection{ModelID: "qwen", Cards: 4, Context: 65536, Mode: "dflash2"}, []loadedPack{{manifest: manifest}}); got != "" {
		t.Fatalf("invalid remembered selection rendered as %q", got)
	}
}

func TestHomeSelectedCardMoves(t *testing.T) {
	view := runtimeView{status: runtime.StatusResult{State: runtime.StateStopped}}
	runSelected := stripANSI(strings.Join(renderHome(100, 36, 0, view, nil, "remembered", "0.1.0", ""), "\n"))
	if !strings.Contains(runSelected, "▸ RUN MODEL") || !strings.Contains(runSelected, "LAST  remembered") {
		t.Fatalf("Run Model card missing from home:\n%s", runSelected)
	}
	runtimeSelected := stripANSI(strings.Join(renderHome(100, 36, 1, view, nil, "remembered", "0.1.0", ""), "\n"))
	if strings.Contains(runtimeSelected, "▸ RUN MODEL") || strings.Contains(runtimeSelected, "LAST  remembered") || !strings.Contains(runtimeSelected, "▸ RUNTIME") {
		t.Fatalf("selected card did not move to Runtime:\n%s", runtimeSelected)
	}
}

func TestHomeNarrowIdentityFallback(t *testing.T) {
	view := runtimeView{status: runtime.StatusResult{State: runtime.StateStopped}}
	lines := renderHome(48, 30, 0, view, nil, "", "0.1.0", "")
	plain := stripANSI(strings.Join(lines, "\n"))
	if !strings.Contains(plain, "B70 // LLM CONTROLLER") {
		t.Fatalf("narrow identity missing:\n%s", plain)
	}
	if strings.Contains(plain, "::::::::") {
		t.Fatalf("full wordmark rendered at narrow width:\n%s", plain)
	}
	for index, line := range lines {
		if got := visibleWidth(line); got > 48 {
			t.Fatalf("narrow line %d width = %d, want <= 48: %q", index, got, stripANSI(line))
		}
	}
}

func TestHomeMinimumSizeFitsWithoutScrolling(t *testing.T) {
	view := runtimeView{status: runtime.StatusResult{State: runtime.StateStopped}}
	lines := renderHome(48, 20, 0, view, nil, "remembered", "0.1.0", "")
	if len(lines) > 20 {
		t.Fatalf("minimum-size home has %d lines, want <= 20", len(lines))
	}
	for index, line := range lines {
		if got := visibleWidth(line); got > 48 {
			t.Fatalf("minimum-size line %d width = %d, want <= 48: %q", index, got, stripANSI(line))
		}
	}
}
