package tui

import (
	"strings"
	"testing"
)

func TestPageIdentityAndFocusedRowRemainReadableWithoutColor(t *testing.T) {
	if got := stripANSI(pageIdentity("Run Model")); got != "B70 // RUN MODEL" {
		t.Fatalf("pageIdentity() = %q", got)
	}
	if got := stripANSI(focusRow("Qwen", true, false, 40)); !strings.Contains(got, "▸ Qwen") {
		t.Fatalf("focusRow() = %q", got)
	}
	if got := stripANSI(statusText("Incomplete")); got != "● INCOMPLETE" {
		t.Fatalf("statusText() = %q", got)
	}
}

func TestSubpagePanelAndLongFieldFitWidth(t *testing.T) {
	width := 48
	line := choiceFieldRow("Model Directory", strings.Repeat("x", 100), "", true, false, width-2)
	for index, rendered := range renderTitledBox("CONTROLLER", width, []string{line}) {
		if got := visibleWidth(rendered); got > width {
			t.Fatalf("line %d width = %d, want <= %d", index, got, width)
		}
	}
}

func TestDisplayWindowKeepsUnderlyingValueAndCursorVisible(t *testing.T) {
	value := "abcdefghijklmnopqrstuvwxyz"
	visible, cursor := displayWindow(value, len([]rune(value)), 8)
	if visible != "tuvwxyz" || cursor != 7 || value != "abcdefghijklmnopqrstuvwxyz" {
		t.Fatalf("displayWindow() = %q, %d", visible, cursor)
	}
}

func TestDestructiveDefaultsRemainSafe(t *testing.T) {
	if confirmDefaultSelection != 0 || uninstallDefaultSelection != 2 {
		t.Fatalf("unsafe defaults: confirm=%d uninstall=%d", confirmDefaultSelection, uninstallDefaultSelection)
	}
}
