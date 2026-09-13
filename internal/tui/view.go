package tui

import (
	"fmt"
	"strings"
	"unicode/utf8"
)

const reverse = "\x1b[7m"

const (
	confirmDefaultSelection   = 0
	uninstallDefaultSelection = 2
)

type actionStyle uint8

const (
	actionSecondary actionStyle = iota
	actionPrimary
	actionDestructive
)

type subpagePanel struct {
	title   string
	content []string
}

func pageIdentity(title string) string {
	return blue + "B70" + reset + dimCyan + " // " + reset + icyBlue + strings.ToUpper(title) + reset
}

func subpageWidth(width int) int {
	return max(1, min(width-4, 104))
}

func (a *app) drawSubpage(page, panel string, content []string, footer string, cursor bool) {
	a.drawSubpagePanels(page, []subpagePanel{{title: panel, content: content}}, footer, cursor)
}

func (a *app) drawSubpagePanels(page string, panels []subpagePanel, footer string, cursor bool) {
	width, height := a.dimensions()
	contentWidth := subpageWidth(width)
	margin := strings.Repeat(" ", max(0, (width-contentWidth)/2))
	lines := []string{margin + pageIdentity(page), ""}
	for panelIndex, panel := range panels {
		if panelIndex > 0 {
			lines = append(lines, "")
		}
		for _, line := range renderTitledBox(panel.title, contentWidth, panel.content) {
			lines = append(lines, margin+line)
		}
	}
	if footer != "" {
		footerLine := margin + muted + truncate(footer, contentWidth) + reset
		for len(lines) < height-1 {
			lines = append(lines, "")
		}
		lines = append(lines, footerLine)
	}
	a.draw(lines, cursor)
}

func subpageFooter(parts ...string) string {
	var visible []string
	for _, part := range parts {
		if part != "" {
			visible = append(visible, part)
		}
	}
	return strings.Join(visible, "   ")
}

func focusRow(text string, selected, disabled bool, width int) string {
	marker := "  "
	style := ""
	if selected {
		marker = "▸ "
		style = icyBlue
	}
	if disabled {
		style = muted
	}
	return "  " + style + marker + truncate(text, max(1, width-4)) + reset
}

func choiceFieldRow(label, value, hint string, selected, disabled bool, width int) string {
	marker := "  "
	labelStyle := muted
	valueStyle := ""
	if selected {
		marker = "▸ "
		labelStyle = blue
		valueStyle = icyBlue
	}
	if disabled {
		labelStyle = muted
		valueStyle = muted
	}
	labelWidth := 18
	if width < 58 {
		labelWidth = 16
	}
	prefixWidth := 2 + 2 + labelWidth
	hintWidth := utf8.RuneCountInString(hint)
	available := max(1, width-prefixWidth-hintWidth-4)
	line := "  " + labelStyle + marker + fmt.Sprintf("%-*s", labelWidth, strings.ToUpper(label)) + reset
	line += valueStyle + truncate(value, available) + reset
	if hint != "" && visibleWidth(line)+hintWidth+2 <= width {
		line += strings.Repeat(" ", max(2, width-visibleWidth(line)-hintWidth-2)) + muted + hint + reset
	}
	return line
}

func fieldRow(label, value string, width int) string {
	labelWidth := 18
	if width < 58 {
		labelWidth = 14
	}
	labelWidth = max(labelWidth, utf8.RuneCountInString(label)+2)
	available := max(1, width-labelWidth-4)
	if visibleWidth(value) > available {
		value = truncate(stripANSI(value), available)
	}
	return "  " + muted + fmt.Sprintf("%-*s", labelWidth, strings.ToUpper(label)) + reset + value
}

func actionRow(label string, selected, disabled bool, style actionStyle, width int) string {
	marker := "  "
	color := muted
	if style == actionPrimary {
		color = blue
	}
	if style == actionDestructive {
		color = red
	}
	if selected {
		marker = "▸ "
		if !disabled && style != actionDestructive {
			color = icyBlue
		}
	}
	if disabled {
		color = muted
	}
	return "  " + color + marker + truncate(strings.ToUpper(label), max(1, width-4)) + reset
}

func statusText(status string) string {
	upper := strings.ToUpper(status)
	style := muted
	switch upper {
	case "PRESENT", "AVAILABLE", "RUNNING", "COMPLETE", "DOWNLOADED", "REUSED":
		style = green
	case "STARTING", "INCOMPLETE", "PULLING", "ORPHANED":
		style = amber
	case "MISSING", "UNAVAILABLE", "FAILED", "ERROR":
		style = red
	}
	return style + "● " + upper + reset
}

func messageLine(value string, width int, style string) string {
	value = strings.ReplaceAll(value, "\r\n", " · ")
	value = strings.ReplaceAll(value, "\n", " · ")
	value = strings.ReplaceAll(value, "\r", " · ")
	value = strings.ReplaceAll(value, "\t", " ")
	return "  " + style + truncate(value, max(1, width-4)) + reset
}

func listPosition(start, end, total int) []string {
	var lines []string
	if start > 0 {
		lines = append(lines, muted+fmt.Sprintf("  ↑ %d above", start)+reset)
	}
	if end < total {
		lines = append(lines, muted+fmt.Sprintf("  ↓ %d more", total-end)+reset)
	}
	return lines
}

func displayWindow(value string, cursor, width int) (string, int) {
	characters := []rune(value)
	width = max(1, width)
	start := 0
	if cursor >= width {
		start = cursor - width + 1
	}
	if start > len(characters) {
		start = len(characters)
	}
	end := min(len(characters), start+width)
	visible := string(characters[start:end])
	return visible, cursor - start
}

func editorValueLine(value []rune, cursor, width int, secret bool) string {
	display := string(value)
	if secret {
		display = strings.Repeat("*", len(value))
	}
	visible, visibleCursor := displayWindow(display, cursor, max(1, width-7))
	characters := []rune(visible)
	if visibleCursor > len(characters) {
		visibleCursor = len(characters)
	}
	before := string(characters[:visibleCursor])
	cursorCharacter := " "
	after := ""
	if visibleCursor < len(characters) {
		cursorCharacter = string(characters[visibleCursor])
		after = string(characters[visibleCursor+1:])
	}
	return "  " + blue + "▸ " + reset + before + reverse + cursorCharacter + reset + after
}
