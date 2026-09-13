package tui

import (
	"fmt"
	"strings"
	"unicode/utf8"

	"b70ctl/internal/config"
	"b70ctl/internal/runtime"
)

const b70Wordmark = ` ::::::::   ::::::::    :::::::: 
:+:    :+: :+:    :+:  :+:    :+:
+:+    +:+      +:+   +:+      +:+
+#++:++#+     +#+     +#+      +#+
+#+    +#+   +#+      +#+      +#+
#+#    #+#  #+#       #+#      #+#
#########   ###         ########`

const (
	cyan       = "\x1b[38;2;83;177;214m"
	dimCyan    = "\x1b[2;38;2;83;177;214m"
	icyBlue    = "\x1b[38;2;214;239;255m"
	muted      = "\x1b[38;2;116;130;146m"
	borderBlue = "\x1b[38;2;64;91;119m"
	amber      = "\x1b[38;2;218;165;32m"
)

type homeMenuItem struct {
	title       string
	description string
}

var homeMenuItems = []homeMenuItem{
	{title: "RUN MODEL", description: "Select and launch a qualified model profile"},
	{title: "RUNTIME", description: "Active runtime information and controls"},
	{title: "MODELS", description: "Local model inventory"},
	{title: "MODEL PACKS", description: "Qualified model-pack management"},
	{title: "SETTINGS", description: "Controller configuration"},
	{title: "EXIT"},
}

func renderHome(width, height, selected int, view runtimeView, b70Count *int, lastRun, version, message string) []string {
	contentWidth := min(width-4, 104)
	if contentWidth < 1 {
		contentWidth = 1
	}
	margin := strings.Repeat(" ", max(0, (width-contentWidth)/2))
	withMargin := func(lines []string) []string {
		result := make([]string, len(lines))
		for index, line := range lines {
			result[index] = margin + line
		}
		return result
	}

	var lines []string
	if width >= 68 && height >= 30 {
		wordmark := renderB70Wordmark()
		wordmarkWidth := 0
		for _, line := range wordmark {
			wordmarkWidth = max(wordmarkWidth, visibleWidth(line))
		}
		for _, line := range wordmark {
			lines = append(lines, margin+centerANSI(padANSI(line, wordmarkWidth), contentWidth))
		}
		lines = append(lines, margin+centerANSI(muted+"LLM CONTROLLER"+reset, contentWidth), "")
	} else {
		identity := blue + "B70" + reset + dimCyan + " // " + reset + icyBlue + "LLM CONTROLLER" + reset
		lines = append(lines, margin+centerANSI(identity, contentWidth), "")
	}

	dense := height < 26
	lines = append(lines, withMargin(renderTitledBox("RUNTIME", contentWidth, renderRuntimeSummary(view, b70Count, contentWidth-2, dense)))...)
	lines = append(lines, "")
	lines = append(lines, withMargin(renderControlPanel(contentWidth, selected, lastRun, dense))...)
	if message != "" {
		lines = append(lines, margin+muted+truncate(message, contentWidth)+reset)
	}

	footer := margin + renderHomeFooter(contentWidth, version)
	for len(lines) < height-1 {
		lines = append(lines, "")
	}
	lines = append(lines, footer)
	return lines
}

func renderB70Wordmark() []string {
	plain := strings.Split(b70Wordmark, "\n")
	lines := make([]string, len(plain))
	for index, line := range plain {
		var rendered strings.Builder
		for _, character := range line {
			switch character {
			case ':':
				rendered.WriteString(dimCyan)
			case '+':
				rendered.WriteString(blue)
			case '#':
				rendered.WriteString(icyBlue)
			default:
				rendered.WriteString(reset)
			}
			rendered.WriteRune(character)
		}
		rendered.WriteString(reset)
		lines[index] = rendered.String()
	}
	return lines
}

func renderTitledBox(title string, width int, content []string) []string {
	width = max(width, utf8.RuneCountInString(title)+7)
	innerWidth := width - 2
	topRule := strings.Repeat("─", max(1, width-utf8.RuneCountInString(title)-5))
	lines := []string{borderBlue + "╭─ " + blue + title + borderBlue + " " + topRule + "╮" + reset}
	for _, line := range content {
		lines = append(lines, borderBlue+"│"+reset+padANSI(line, innerWidth)+borderBlue+"│"+reset)
	}
	lines = append(lines, borderBlue+"╰"+strings.Repeat("─", innerWidth)+"╯"+reset)
	return lines
}

func renderRuntimeSummary(view runtimeView, b70Count *int, width int, forceTwoColumns bool) []string {
	unknown := "—"
	status := styledRuntimeStatus(view)
	model, mode, cards, context := unknown, unknown, unknown, unknown
	if view.err == nil && view.model != nil && view.profile != nil && view.status.State != runtime.StateStopped {
		model = view.model.Name
		mode = modeName(view.pack.manifest, view.profile.Mode)
		cards = fmt.Sprintf("TP%d", view.profile.Cards)
		context = formatContext(view.profile.Context)
	}
	count := unknown
	if b70Count != nil {
		count = fmt.Sprintf("%d", *b70Count)
	}

	left := []homeField{{"STATUS", status}, {"MODEL", model}, {"MODE", mode}}
	right := []homeField{{"B70", count}, {"CARDS", cards}, {"CONTEXT", context}}
	if width < 72 && !forceTwoColumns {
		fields := append(left, right...)
		lines := make([]string, 0, len(fields))
		for _, field := range fields {
			lines = append(lines, renderHomeField(field, width))
		}
		return lines
	}

	gap := 3
	leftWidth := (width - gap) / 2
	rightWidth := width - gap - leftWidth
	lines := make([]string, len(left))
	for index := range left {
		lines[index] = padANSI(renderHomeField(left[index], leftWidth), leftWidth) + strings.Repeat(" ", gap) + renderHomeField(right[index], rightWidth)
	}
	return lines
}

type homeField struct {
	label string
	value string
}

func renderHomeField(field homeField, width int) string {
	prefix := "  " + muted + fmt.Sprintf("%-10s", field.label) + reset
	available := max(1, width-12)
	value := field.value
	if !strings.Contains(value, "\x1b[") {
		value = truncate(value, available)
	}
	return prefix + value
}

func styledRuntimeStatus(view runtimeView) string {
	if view.err != nil {
		return red + "● ERROR" + reset
	}
	label := strings.ToUpper(string(view.status.State))
	style := muted
	switch view.status.State {
	case runtime.StateRunning:
		style = green
	case runtime.StateStarting:
		style = amber
	case runtime.StateFailed:
		style = red
	}
	if label == "" {
		label = "UNKNOWN"
	}
	return style + "● " + label + reset
}

func renderControlPanel(width, selected int, lastRun string, dense bool) []string {
	innerWidth := width - 2
	var content []string
	for index, item := range homeMenuItems {
		if index == selected {
			selectedItem := item
			cardLastRun := ""
			if index == 0 && !dense {
				cardLastRun = lastRun
			}
			if dense {
				selectedItem.description = ""
			}
			content = append(content, renderSelectedHomeCard(selectedItem, innerWidth-4, cardLastRun)...)
			continue
		}
		unselectedItem := item
		if dense {
			unselectedItem.description = ""
		}
		content = append(content, renderUnselectedHomeItem(unselectedItem, innerWidth))
	}
	return renderTitledBox("CONTROL", width, content)
}

func renderSelectedHomeCard(item homeMenuItem, width int, lastRun string) []string {
	width = max(width, 16)
	innerWidth := width - 2
	lines := []string{"  " + blue + "╭" + strings.Repeat("─", innerWidth) + "╮" + reset}
	title := icyBlue + " ▸ " + truncate(item.title, max(1, innerWidth-3)) + reset
	lines = append(lines, "  "+blue+"│"+reset+padANSI(title, innerWidth)+blue+"│"+reset)
	if item.description != "" {
		description := muted + "   " + truncate(item.description, max(1, innerWidth-3)) + reset
		lines = append(lines, "  "+blue+"│"+reset+padANSI(description, innerWidth)+blue+"│"+reset)
	}
	if lastRun != "" {
		metadata := "   " + blue + "LAST  " + reset + muted + truncate(lastRun, max(1, innerWidth-9)) + reset
		lines = append(lines, "  "+blue+"│"+reset+padANSI(metadata, innerWidth)+blue+"│"+reset)
	}
	lines = append(lines, "  "+blue+"╰"+strings.Repeat("─", innerWidth)+"╯"+reset)
	return lines
}

func renderUnselectedHomeItem(item homeMenuItem, width int) string {
	prefix := "    " + muted + fmt.Sprintf("%-16s", item.title) + reset
	if item.description == "" || width <= 20 {
		return truncateANSI(prefix, width)
	}
	descriptionWidth := max(1, width-20)
	return prefix + muted + truncate(item.description, descriptionWidth) + reset
}

func lastRunSummary(preference *config.RunSelection, packs []loadedPack) string {
	if preference == nil {
		return ""
	}
	for _, pack := range packs {
		for _, model := range pack.manifest.Models {
			if model.ID != preference.ModelID {
				continue
			}
			if _, ok := exactProfile(pack.manifest.Profiles, model.ID, preference.Cards, preference.Context, preference.Mode); !ok {
				return ""
			}
			return fmt.Sprintf("%s · TP%d · %s · %s", model.Name, preference.Cards, formatContext(preference.Context), modeName(pack.manifest, preference.Mode))
		}
	}
	return ""
}

func renderHomeFooter(width int, version string) string {
	left := "↑↓ Navigate   Enter Select   Ctrl-C Exit"
	right := "B70CTL"
	if version != "" {
		right += " " + version
	}
	if utf8.RuneCountInString(left)+utf8.RuneCountInString(right)+2 > width {
		left = truncate(left, max(1, width-utf8.RuneCountInString(right)-2))
	}
	gap := max(1, width-utf8.RuneCountInString(left)-utf8.RuneCountInString(right))
	return muted + left + strings.Repeat(" ", gap) + right + reset
}

func centerANSI(value string, width int) string {
	padding := max(0, width-visibleWidth(value))
	return strings.Repeat(" ", padding/2) + value
}

func padANSI(value string, width int) string {
	if visibleWidth(value) > width {
		value = truncateANSI(value, width)
	}
	return value + strings.Repeat(" ", max(0, width-visibleWidth(value)))
}

func truncateANSI(value string, width int) string {
	if width <= 0 {
		return ""
	}
	plain := stripANSI(value)
	return truncate(plain, width)
}

func visibleWidth(value string) int {
	return utf8.RuneCountInString(stripANSI(value))
}

func stripANSI(value string) string {
	var plain strings.Builder
	inEscape := false
	for _, character := range value {
		if inEscape {
			if character == 'm' {
				inEscape = false
			}
			continue
		}
		if character == '\x1b' {
			inEscape = true
			continue
		}
		plain.WriteRune(character)
	}
	return plain.String()
}
