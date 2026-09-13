package tui

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"golang.org/x/term"

	"b70ctl/internal/catalog"
	"b70ctl/internal/config"
	"b70ctl/internal/hardware"
	"b70ctl/internal/hf"
	"b70ctl/internal/install"
	"b70ctl/internal/modelpack"
	"b70ctl/internal/modelstore"
	"b70ctl/internal/packstore"
	"b70ctl/internal/packupdate"
	"b70ctl/internal/runtime"
	"b70ctl/internal/uninstall"
)

const (
	blue       = "\x1b[38;2;0;104;181m"
	red        = "\x1b[31m"
	green      = "\x1b[32m"
	reset      = "\x1b[0m"
	clear      = "\x1b[2J\x1b[H"
	hideCursor = "\x1b[?25l"
	showCursor = "\x1b[?25h"
	minimumW   = 48
	minimumH   = 20
)

type app struct {
	in            *input
	out           *os.File
	config        config.Config
	paths         config.Paths
	message       string
	version       string
	detectedB70   int
	b70CountKnown bool
}

type loadedPack struct {
	installed packstore.InstalledPack
	manifest  *modelpack.Manifest
	path      string
}

type runtimeView struct {
	status  runtime.StatusResult
	pack    *loadedPack
	profile *modelpack.Profile
	model   *modelpack.Model
	err     error
}

type option struct {
	label    string
	disabled bool
}

type targetChoice struct {
	pack     loadedPack
	model    modelpack.Model
	profiles []modelpack.Profile
	label    string
}

type runSelection struct {
	target  int
	cards   int
	context int
	mode    string
	access  string
	port    int
}

type evaluation struct {
	availability runtime.ProfileAvailability
	models       []modelstore.Artifact
	detected     int
	err          error
}

type modelDownloadState struct {
	repository string
	status     string
	progress   hf.Progress
	started    time.Time
	lastSample time.Time
	lastBytes  int64
	speed      float64
	elapsed    time.Duration
	activity   int
	terminal   bool
}

type downloadOutcome struct {
	result hf.Result
	err    error
}

type preparationOutcome struct {
	result install.Result
	err    error
}

type remotePackOutcome struct {
	pack *catalog.AcquiredPack
	err  error
}

type runtimeDownloadState struct {
	declared modelpack.Runtime
	status   string
	activity string
	latest   string
	started  time.Time
	elapsed  time.Duration
	frame    int
}

type runtimeDownloadOutcome struct {
	result runtime.AcquisitionResult
}

type localPackSelection struct {
	targets  []modelpack.Model
	selected map[string]bool
}

var acquireRuntime = runtime.AcquireOwned

var newCatalogClient = catalog.NewClient

var runtimeStatus = runtime.Status

type hfClient interface {
	Inspect(context.Context, string, string) (hf.Repository, error)
	Install(context.Context, string, hf.Repository, func(hf.Progress)) (hf.Result, error)
}

var newHFClient = func(token string) hfClient { return hf.NewClient(token) }

func Run(stdin, stdout *os.File, value config.Config, paths config.Paths, versions ...string) error {
	application := &app{
		in:     newInput(stdin),
		out:    stdout,
		config: value,
		paths:  paths,
	}
	if len(versions) > 0 {
		application.version = versions[0]
	}
	if devices, err := hardware.DetectB70(); err == nil {
		application.detectedB70 = len(devices)
		application.b70CountKnown = true
	}
	defer application.in.close()
	defer fmt.Fprint(stdout, reset+showCursor+"\r\n")
	err := application.mainScreen()
	if err == io.EOF {
		return nil
	}
	return err
}

func (a *app) mainScreen() error {
	selected := 0
	for {
		if !a.sizeOK() {
			exit, err := a.tooSmall()
			if err != nil || exit {
				return err
			}
			continue
		}
		view := a.runtimeView()
		var count *int
		if a.b70CountKnown {
			count = &a.detectedB70
		}
		var packs []loadedPack
		if a.config.LastRun != nil {
			packs, _ = a.loadPacks()
		}
		width, height := a.dimensions()
		lines := renderHome(width, height, selected, view, count, lastRunSummary(a.config.LastRun, packs), a.version, a.message)
		a.draw(lines, false)
		a.message = ""

		event, err := a.in.next()
		if err != nil {
			if err == io.EOF {
				return nil
			}
			return err
		}
		switch event.key {
		case keyCtrlC:
			return nil
		case keyUp:
			selected = previous(selected, len(homeMenuItems))
		case keyDown:
			selected = next(selected, len(homeMenuItems))
		case keyEnter:
			switch selected {
			case 0:
				if err := a.runModelScreen(); err != nil {
					return err
				}
			case 1:
				if err := a.runtimeScreen(); err != nil {
					return err
				}
			case 2:
				if err := a.modelsScreen(); err != nil {
					return err
				}
			case 3:
				if err := a.modelPacksScreen(); err != nil {
					return err
				}
			case 4:
				if err := a.settingsScreen(); err != nil {
					return err
				}
			case 5:
				return nil
			}
		}
	}
}

func (a *app) runtimeScreen() error {
	selected := 0
	for {
		if !a.sizeOK() {
			exit, err := a.tooSmall()
			if err != nil || exit {
				return err
			}
			continue
		}
		view := a.runtimeView()
		contentWidth := subpageWidth(a.width()) - 2
		var statusLines []string
		var actions []string
		if view.err != nil {
			statusLines = append(statusLines,
				fieldRow("Status", statusText("Error"), contentWidth),
				"",
				messageLine(view.err.Error(), contentWidth, muted),
			)
			actions = []string{"Refresh", "Back"}
		} else if view.status.State == runtime.StateStopped {
			statusLines = append(statusLines, fieldRow("Status", statusText("Stopped"), contentWidth), "", messageLine("No model is currently running.", contentWidth, muted))
			if view.status.PackID != "" {
				actions = []string{"View Logs", "Back"}
			} else {
				actions = []string{"Back"}
			}
		} else {
			for _, detail := range runtimeDetails(view) {
				label, value, found := strings.Cut(detail, "        ")
				if !found {
					statusLines = append(statusLines, messageLine(detail, contentWidth, ""))
					continue
				}
				if strings.TrimSpace(label) == "Status" {
					value = statusText(strings.TrimSpace(value))
				}
				if strings.TrimSpace(label) == "Cards" {
					value = "TP" + strings.TrimSpace(value)
				}
				statusLines = append(statusLines, fieldRow(strings.TrimSpace(label), strings.TrimSpace(value), contentWidth))
			}
			if view.status.State == runtime.StateFailed {
				actions = []string{"View Logs", "Back"}
			} else {
				actions = []string{"Refresh", "View Logs", "Stop", "Back"}
			}
		}
		if selected >= len(actions) {
			selected = len(actions) - 1
		}
		actionLines := make([]string, 0, len(actions))
		for index, action := range actions {
			style := actionSecondary
			if action == "Stop" {
				style = actionDestructive
			}
			actionLines = append(actionLines, actionRow(action, index == selected, false, style, contentWidth))
		}
		if a.message != "" {
			statusLines = append(statusLines, "", messageLine(a.message, contentWidth, muted))
		}
		a.drawSubpagePanels("Runtime", []subpagePanel{{title: "STATUS", content: statusLines}, {title: "ACTIONS", content: actionLines}}, subpageFooter("↑↓ Navigate", "Enter Select", "Esc Back"), false)
		a.message = ""

		event, err := a.in.next()
		if err != nil {
			return err
		}
		if event.key == keyCtrlC {
			return io.EOF
		}
		if event.key == keyEscape {
			return nil
		}
		if event.key == keyUp {
			selected = previous(selected, len(actions))
		}
		if event.key == keyDown {
			selected = next(selected, len(actions))
		}
		if event.key != keyEnter {
			continue
		}
		action := actions[selected]
		switch action {
		case "Back":
			return nil
		case "View Logs":
			if err := a.logsScreen(); err != nil {
				return err
			}
		case "Stop":
			name := "the current model"
			if view.model != nil {
				name = view.model.Name
			}
			yes, err := a.confirm("Stop " + name + "?")
			if err != nil {
				return err
			}
			if yes {
				if err := runtime.Stop(); err != nil {
					a.message = err.Error()
				}
			}
		}
	}
}

func (a *app) logsScreen() error {
	logs, logErr := runtime.Logs(100)
	selected := 0
	for {
		if !a.sizeOK() {
			exit, err := a.tooSmall()
			if err != nil || exit {
				return err
			}
			continue
		}
		width, height := a.dimensions()
		contentWidth := subpageWidth(width) - 2
		lines := []string{}
		if logErr != nil {
			lines = append(lines, messageLine(logErr.Error(), contentWidth, red))
		} else {
			logLines := newestLogLines(logs, max(1, height-11))
			for _, line := range logLines {
				lines = append(lines, " "+truncate(sanitize(line), max(1, contentWidth-2)))
			}
		}
		actionLines := []string{}
		actions := []string{"Refresh", "Back"}
		for index, action := range actions {
			actionLines = append(actionLines, actionRow(action, index == selected, false, actionSecondary, contentWidth))
		}
		a.drawSubpagePanels("Runtime Log", []subpagePanel{{title: "LATEST 100 LINES", content: lines}, {title: "ACTIONS", content: actionLines}}, subpageFooter("↑↓ Navigate", "Enter Select", "Esc Back", "Latest tail"), false)

		event, err := a.in.next()
		if err != nil {
			return err
		}
		switch event.key {
		case keyCtrlC:
			return io.EOF
		case keyEscape:
			return nil
		case keyUp, keyDown:
			selected = 1 - selected
		case keyEnter:
			if selected == 1 {
				return nil
			}
			logs, logErr = runtime.Logs(100)
		}
	}
}

func (a *app) runModelScreen() error {
	packs, err := a.loadPacks()
	if err != nil {
		return a.messageScreen("Run Model", err.Error())
	}
	targets := buildTargets(packs)
	if len(targets) == 0 {
		return a.messageScreen("Run Model", "no qualified profiles are installed")
	}
	selection := runSelection{target: 0, access: a.config.DefaultAccess, port: a.config.DefaultPort}
	detected := detectedCards()
	a.resetSelection(&selection, targets[0], detected)
	a.restoreLastRun(&selection, targets, detected)
	selected := 0
	for {
		if !a.sizeOK() {
			exit, err := a.tooSmall()
			if err != nil || exit {
				return err
			}
			continue
		}
		target := targets[selection.target]
		profile, ok := exactProfile(target.profiles, target.model.ID, selection.cards, selection.context, selection.mode)
		eval := evaluation{err: fmt.Errorf("profile is unavailable")}
		if ok {
			eval = evaluate(target, profile, a.config.ModelDirectory)
		}
		contentWidth := subpageWidth(a.width()) - 2
		values := []string{
			target.label,
			strconv.Itoa(selection.cards),
			formatContext(selection.context),
			modeName(target.pack.manifest, selection.mode),
			accessName(selection.access),
			strconv.Itoa(selection.port),
			"Start",
			"Back",
		}
		labels := []string{"Model", "Cards", "Context", "Mode", "Access", "Port"}
		profileLines := make([]string, 0, 10)
		for index, label := range labels {
			hint := ""
			if index == 1 {
				hint = "TP" + values[index]
			} else if index == 4 {
				hint = "← →"
			}
			profileLines = append(profileLines, choiceFieldRow(label, values[index], hint, index == selected, false, contentWidth))
		}
		profileLines = append(profileLines, "")
		profileLines = append(profileLines,
			actionRow("Start Model", selected == 6, !eval.availability.Available, actionPrimary, contentWidth),
			actionRow("Back", selected == 7, false, actionSecondary, contentWidth),
		)
		availabilityLines := []string{}
		if eval.availability.Available {
			availabilityLines = append(availabilityLines, messageLine("● PROFILE AVAILABLE", contentWidth, green))
		} else {
			availabilityLines = append(availabilityLines, messageLine("● PROFILE UNAVAILABLE", contentWidth, red))
			for _, blocker := range eval.availability.Missing {
				availabilityLines = append(availabilityLines, messageLine("• "+blocker, contentWidth, muted))
			}
			if eval.err != nil {
				availabilityLines = append(availabilityLines, "", messageLine(eval.err.Error(), contentWidth, red))
			}
		}
		if selection.access == config.AccessLAN {
			availabilityLines = append(availabilityLines, "", messageLine("LAN exposes the model API to other devices on this network.", contentWidth, muted))
		}
		if a.message != "" {
			availabilityLines = append(availabilityLines, "", messageLine(a.message, contentWidth, red))
		}
		a.drawSubpagePanels("Run Model", []subpagePanel{{title: "PROFILE", content: profileLines}, {title: "AVAILABILITY", content: availabilityLines}}, subpageFooter("↑↓ Navigate", "Enter Select", "←→ Change", "Esc Back"), false)
		a.message = ""

		event, err := a.in.next()
		if err != nil {
			return err
		}
		switch event.key {
		case keyCtrlC:
			return io.EOF
		case keyEscape:
			return nil
		case keyUp:
			selected = previous(selected, len(values))
		case keyDown:
			selected = next(selected, len(values))
		case keyLeft, keyRight:
			if selected == 4 {
				selection.access = otherAccess(selection.access)
			}
		case keyEnter:
			switch selected {
			case 0:
				options := make([]option, len(targets))
				for index := range targets {
					options[index].label = targets[index].label
				}
				choice, accepted, err := a.choose("Model", options, selection.target)
				if err != nil {
					return err
				}
				if accepted {
					selection.target = choice
					a.resetSelection(&selection, targets[choice], eval.detected)
				}
			case 1:
				cards := cardOptions(target.profiles, target.model.ID)
				options := make([]option, len(cards))
				current := 0
				for index, count := range cards {
					options[index].label = strconv.Itoa(count)
					options[index].disabled = count > eval.detected
					if options[index].disabled {
						options[index].label += fmt.Sprintf("  (unavailable; %d detected)", eval.detected)
					}
					if count == selection.cards {
						current = index
					}
				}
				choice, accepted, err := a.choose("Cards", options, current)
				if err != nil {
					return err
				}
				if accepted {
					selection.cards = cards[choice]
					resetContextMode(&selection, target)
				}
			case 2:
				contexts := contextOptions(target.profiles, target.model.ID, selection.cards)
				options := make([]option, len(contexts))
				current := 0
				for index, context := range contexts {
					options[index].label = formatContext(context)
					if context == selection.context {
						current = index
					}
				}
				choice, accepted, err := a.choose("Context", options, current)
				if err != nil {
					return err
				}
				if accepted {
					selection.context = contexts[choice]
					selection.mode = modeOptions(target.pack.manifest, target.profiles, target.model.ID, selection.cards, selection.context)[0]
				}
			case 3:
				modes := modeOptions(target.pack.manifest, target.profiles, target.model.ID, selection.cards, selection.context)
				options := make([]option, len(modes))
				current := 0
				for index, mode := range modes {
					options[index].label = modeName(target.pack.manifest, mode)
					if mode == selection.mode {
						current = index
					}
				}
				choice, accepted, err := a.choose("Mode", options, current)
				if err != nil {
					return err
				}
				if accepted {
					selection.mode = modes[choice]
				}
			case 4:
				selection.access = otherAccess(selection.access)
			case 5:
				text, accepted, err := a.editText("Port:", strconv.Itoa(selection.port))
				if err != nil {
					return err
				}
				if accepted {
					port, err := parsePort(text)
					if err != nil {
						a.message = err.Error()
					} else {
						selection.port = port
					}
				}
			case 6:
				if !eval.availability.Available {
					a.message = "profile is unavailable"
					continue
				}
				a.saveRunSelection(target, selection)
				launch, err := runtime.BuildLaunch(target.pack.manifest, profile, eval.models, runtime.Access(selection.access), selection.port)
				if err == nil {
					err = runtime.Start(launch)
				}
				if err != nil {
					a.message = err.Error()
					continue
				}
				return a.runtimeScreen()
			case 7:
				return nil
			}
		}
	}
}

func (a *app) modelsScreen() error {
	selected := 0
	for {
		if !a.sizeOK() {
			exit, err := a.tooSmall()
			if err != nil || exit {
				return err
			}
			continue
		}
		packs, err := a.loadPacks()
		var artifacts []modelstore.Artifact
		if err == nil {
			artifacts, err = modelstore.Scan(a.config.ModelDirectory)
		}
		entries := modelsInventory(packs, artifacts)
		width, height := a.dimensions()
		contentWidth := subpageWidth(width) - 2
		lines := []string{}
		totalItems := len(entries) + 2
		if selected >= totalItems {
			selected = totalItems - 1
		}
		if err != nil {
			lines = append(lines, messageLine("● ERROR", contentWidth, red), messageLine(err.Error(), contentWidth, muted))
		} else if len(entries) == 0 {
			lines = append(lines, messageLine("No installed packs require models.", contentWidth, muted))
		} else {
			visible := max(1, (height-9)/5)
			start := 0
			if len(entries) > visible {
				if selected < len(entries) && selected >= visible {
					start = selected - visible + 1
				} else if selected >= len(entries) {
					start = len(entries) - visible
				}
			}
			end := min(len(entries), start+visible)
			for index := start; index < end; index++ {
				entry := entries[index]
				readiness := entry.readiness
				detail := "    " + statusText(string(readiness.State)) + muted + "     rev " + truncateRevision(entry.model.Revision) + reset
				if entry.orphan {
					detail += "  " + statusText("Orphaned")
				}
				lines = append(lines, focusRow(entry.model.Repo, index == selected, false, contentWidth), detail)
				if readiness.State == modelstore.Incomplete {
					for _, summary := range completenessSummary(readiness.Check) {
						lines = append(lines, messageLine(strings.TrimSpace(summary), contentWidth, amber))
					}
				}
				lines = append(lines, "")
			}
			lines = append(lines, listPosition(start, end, len(entries))...)
		}
		actions := []string{"Rescan", "Back"}
		actionLines := []string{}
		for index, action := range actions {
			actionLines = append(actionLines, actionRow(action, len(entries)+index == selected, false, actionSecondary, contentWidth))
		}
		if a.message != "" {
			lines = append(lines, "", messageLine(a.message, contentWidth, muted))
		}
		a.drawSubpagePanels("Models", []subpagePanel{{title: "INVENTORY", content: lines}, {title: "ACTIONS", content: actionLines}}, subpageFooter("↑↓ Navigate", "Enter Select", "Esc Back"), false)
		a.message = ""
		event, err := a.in.next()
		if err != nil {
			return err
		}
		switch event.key {
		case keyCtrlC:
			return io.EOF
		case keyEscape:
			return nil
		case keyUp:
			selected = previous(selected, totalItems)
		case keyDown:
			selected = next(selected, totalItems)
		case keyEnter:
			if selected < len(entries) {
				entry := entries[selected]
				if err := a.modelScreen(entry.model, entry.readiness); err != nil {
					return err
				}
			} else if selected == len(entries) {
				a.message = "Models rescanned."
			} else {
				return nil
			}
		}
	}
}

func (a *app) modelScreen(model modelpack.Model, readiness modelstore.ReadinessResult) error {
	selected := 0
	refresh := func() error {
		artifacts, err := modelstore.Scan(a.config.ModelDirectory)
		if err != nil {
			return err
		}
		readiness = modelstore.Assess(artifacts, model)
		selected = 0
		return nil
	}
	for {
		if !a.sizeOK() {
			exit, err := a.tooSmall()
			if err != nil || exit {
				return err
			}
			continue
		}
		// An orphan is a local model no installed pack references and no
		// pack supplies an expected file inventory for: Delete is its only
		// action. Models carrying a pack inventory keep their normal
		// Download/Replace actions even while unreferenced.
		orphan := false
		if packs, packsErr := a.loadPacks(); packsErr == nil {
			orphan = len(model.Files) == 0 && len(packsRequiring(packs, model.Repo, model.Revision)) == 0
		}
		actions := []string{"Download", "Back"}
		occupied := false
		switch {
		case orphan && readiness.State == modelstore.Present:
			actions = []string{"Delete", "Back"}
		case orphan:
			actions = []string{"Back"}
		case readiness.State == modelstore.Present:
			actions = []string{"Delete", "Back"}
		case readiness.State == modelstore.Incomplete:
			actions = []string{"Replace", "Back"}
		default:
			occupied, _ = modelstore.DestinationExists(a.config.ModelDirectory, model.Repo)
			if occupied {
				actions = []string{"Replace", "Back"}
			}
		}
		if selected >= len(actions) {
			selected = len(actions) - 1
		}
		contentWidth := subpageWidth(a.width()) - 2
		artifactLines := []string{
			fieldRow("Repository", model.Repo, contentWidth),
			fieldRow("Status", statusText(string(readiness.State)), contentWidth),
			fieldRow("Revision", model.Revision, contentWidth),
		}
		if orphan {
			artifactLines = append(artifactLines, fieldRow("Required by", statusText("Orphaned"), contentWidth))
		}
		var issueLines []string
		if readiness.State == modelstore.Incomplete {
			issueLines = append(issueLines, messageLine("This model does not match the complete file inventory required by the installed pack.", contentWidth, muted))
			for _, line := range completenessSummary(readiness.Check) {
				issueLines = append(issueLines, messageLine(strings.TrimSpace(line), contentWidth, amber))
			}
			for _, line := range affectedPaths(readiness.Check, 3) {
				issueLines = append(issueLines, messageLine(strings.TrimSpace(line), contentWidth, muted))
			}
		} else if readiness.State == modelstore.Missing && occupied {
			issueLines = append(issueLines, messageLine("The download destination is occupied by an unrecognized directory.", contentWidth, muted))
		} else if orphan {
			issueLines = append(issueLines, messageLine("No installed pack requires this model, so its expected file inventory is not known.", contentWidth, muted))
		}
		actionLines := []string{}
		for index, action := range actions {
			style := actionSecondary
			switch action {
			case "Download":
				style = actionPrimary
			case "Delete", "Replace":
				style = actionDestructive
			}
			actionLines = append(actionLines, actionRow(action, index == selected, false, style, contentWidth))
		}
		if a.message != "" {
			artifactLines = append(artifactLines, "", messageLine(a.message, contentWidth, muted))
		}
		panels := []subpagePanel{{title: "ARTIFACT", content: artifactLines}}
		if len(issueLines) > 0 {
			panels = append(panels, subpagePanel{title: "ISSUES", content: issueLines})
		}
		panels = append(panels, subpagePanel{title: "ACTIONS", content: actionLines})
		a.drawSubpagePanels("Model", panels, subpageFooter("↑↓ Navigate", "Enter Select", "Esc Back"), false)
		a.message = ""
		event, err := a.in.next()
		if err != nil {
			return err
		}
		switch event.key {
		case keyCtrlC:
			return io.EOF
		case keyEscape:
			return nil
		case keyUp:
			selected = previous(selected, len(actions))
		case keyDown:
			selected = next(selected, len(actions))
		case keyEnter:
			switch actions[selected] {
			case "Back":
				return nil
			case "Replace":
				if reason := a.replaceBlocked(model); reason != "" {
					a.message = reason
					continue
				}
				yes, err := a.confirmModelRemoval(model, "Replace this model? The local files are removed and the model is downloaded again.")
				if err != nil {
					return err
				}
				if !yes {
					continue
				}
				if reason := a.clearModelForReplacement(model); reason != "" {
					a.message = reason
					continue
				}
				if err := refresh(); err != nil {
					return err
				}
				fallthrough
			case "Download":
				installed, err := a.downloadModel(model)
				if err != nil {
					return err
				}
				if installed {
					if err := refresh(); err != nil {
						return err
					}
				}
			case "Delete":
				if reason := a.deleteBlocked(model); reason != "" {
					a.message = reason
					continue
				}
				yes, err := a.confirmModelRemoval(model, "Delete this model? The local files are removed permanently.")
				if err != nil {
					return err
				}
				if !yes {
					continue
				}
				removed, message := a.deleteModelArtifact(model)
				a.message = message
				if orphan && removed {
					// No installed pack requires the model, so the Models
					// list drops the row entirely on its next rebuild
					// instead of keeping a Missing requirement row.
					return nil
				}
				if err := refresh(); err != nil {
					return err
				}
			}
		}
	}
}

// deleteBlocked reports why the model must not be deleted, or "" when a
// standalone delete may proceed: the model may not be mounted by a Starting
// or Running runtime and may not be required by more than one installed pack.
func (a *app) deleteBlocked(model modelpack.Model) string {
	if reason := a.inUseBlocked(model, "deleted"); reason != "" {
		return reason
	}
	packs, err := a.loadPacks()
	if err != nil {
		return err.Error()
	}
	if count := len(packsRequiring(packs, model.Repo, model.Revision)); count > 1 {
		return "Not deleted — the checkpoint is shared by " + strconv.Itoa(count) + " installed packs."
	}
	return ""
}

// replaceBlocked reports why the model must not be replaced, or "" when the
// replacement may proceed. Sharing is not a blocker: replacement ends with
// the checkpoint Present again for every installed pack.
func (a *app) replaceBlocked(model modelpack.Model) string {
	return a.inUseBlocked(model, "replaced")
}

// inUseBlocked refuses destructive model actions while a Starting or Running
// Controller runtime mounts the model. A runtime state that cannot be
// determined safely also refuses.
func (a *app) inUseBlocked(model modelpack.Model, verb string) string {
	status, err := runtimeStatus()
	if err != nil {
		return "Runtime state is unavailable — the model was not " + verb + ": " + err.Error()
	}
	if status.State != runtime.StateStarting && status.State != runtime.StateRunning {
		return ""
	}
	packs, err := a.loadPacks()
	if err != nil {
		return err.Error()
	}
	view := resolveRuntimeMetadata(status, packs)
	if view.pack == nil || view.profile == nil {
		return "Runtime state is unavailable — the model was not " + verb + ": the running profile could not be resolved"
	}
	mounted := mountedModels(view.pack.manifest, *view.profile)
	if mounted == nil {
		return "Runtime state is unavailable — the model was not " + verb + ": the running profile could not be resolved"
	}
	for _, entry := range mounted {
		if entry.Repo == model.Repo && entry.Revision == model.Revision {
			return "Not " + verb + " — mounted by the running pack (" + view.pack.installed.Name + "). Stop it first."
		}
	}
	return ""
}

// deleteModelArtifact removes the model via the shared removal engine and
// reports whether the artifact was removed, alongside the user-facing status
// message.
func (a *app) deleteModelArtifact(model modelpack.Model) (bool, string) {
	status, err := modelstore.Remove(a.config.ModelDirectory, model.Repo, model.Revision)
	if err != nil {
		return false, "Not deleted — " + err.Error()
	}
	switch status {
	case modelstore.RemovalRemoved:
		return true, "Model deleted."
	case modelstore.RemovalMissing:
		return false, "Model was not found on disk."
	case modelstore.RemovalExternalHFCache:
		return false, "Not deleted — external Hugging Face cache."
	default:
		return false, "Not deleted — multiple local copies found."
	}
}

// clearModelForReplacement removes the local artifact so the normal
// exact-revision download can start from scratch: the marked model via the
// shared removal engine, then any undiscoverable directory occupying the
// canonical destination. It returns "" when the path is clear.
func (a *app) clearModelForReplacement(model modelpack.Model) string {
	status, err := modelstore.Remove(a.config.ModelDirectory, model.Repo, model.Revision)
	if err != nil {
		return "Not replaced — " + err.Error()
	}
	switch status {
	case modelstore.RemovalExternalHFCache:
		return "Not replaced — external Hugging Face cache."
	case modelstore.RemovalMultipleCopies:
		return "Not replaced — multiple local copies found."
	}
	status, err = modelstore.RemoveDestination(a.config.ModelDirectory, model.Repo)
	if err != nil {
		return "Not replaced — " + err.Error()
	}
	if status == modelstore.RemovalRecognizedModel {
		return "Not replaced — the destination holds another recognized model."
	}
	return ""
}

func (a *app) confirmModelRemoval(model modelpack.Model, question string) (bool, error) {
	selected := confirmDefaultSelection
	for {
		if !a.sizeOK() {
			exit, err := a.tooSmall()
			if err != nil || exit {
				return false, err
			}
			continue
		}
		contentWidth := subpageWidth(a.width()) - 2
		lines := []string{
			fieldRow("Repository", model.Repo, contentWidth),
			fieldRow("Revision", model.Revision, contentWidth),
			"",
			messageLine(question, contentWidth, muted),
			"",
			actionRow("No", selected == 0, false, actionSecondary, contentWidth),
			actionRow("Yes", selected == 1, false, actionDestructive, contentWidth),
		}
		a.drawSubpage("Confirm", "CONFIRM", lines, subpageFooter("↑↓ Navigate", "Enter Select", "Esc Cancel"), false)
		event, err := a.in.next()
		if err != nil {
			return false, err
		}
		switch event.key {
		case keyCtrlC:
			return false, io.EOF
		case keyEscape:
			return false, nil
		case keyUp, keyDown:
			selected = 1 - selected
		case keyEnter:
			return selected == 1, nil
		}
	}
}

func (a *app) downloadModel(model modelpack.Model) (bool, error) {
	token, _, err := hf.Get()
	if err != nil {
		a.message = err.Error()
		return false, nil
	}
	if model.Gated && token == "" {
		return false, a.hfAccessRequired()
	}
	contentWidth := subpageWidth(a.width()) - 2
	a.drawSubpage("Model Download", "CHECKING", []string{fieldRow("Model", model.Repo, contentWidth), "", messageLine("Retrieving exact revision metadata…", contentWidth, muted)}, "", false)
	client := newHFClient(token)
	repository, err := client.Inspect(context.Background(), model.Repo, model.Revision)
	if err != nil {
		return false, a.hfDownloadError(model.Repo, err)
	}
	yes, err := a.confirmDownload(repository)
	if err != nil || !yes {
		return false, err
	}
	state := newModelDownloadState(repository, time.Now())
	progresses := make(chan hf.Progress, 1)
	finished := make(chan downloadOutcome, 1)
	go func() {
		result, installErr := client.Install(context.Background(), a.config.ModelDirectory, repository, func(progress hf.Progress) {
			sendLatestProgress(progresses, progress)
		})
		finished <- downloadOutcome{result: result, err: installErr}
	}()

	a.downloadProgress(state)
	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()
	for {
		select {
		case progress := <-progresses:
			state.update(progress)
		case now := <-ticker.C:
			drainProgress(progresses, &state)
			state.refresh(now)
			a.downloadProgress(state)
		case outcome := <-finished:
			drainProgress(progresses, &state)
			if outcome.err != nil {
				state.finish("Failed", time.Now())
				return false, a.hfDownloadError(model.Repo, outcome.err)
			}
			state.finish("Complete", time.Now())
			if outcome.result.AlreadyPresent {
				if modelstore.Check(outcome.result.Path, model.Files).Complete {
					a.message = "Model already present."
					return true, nil
				}
				return false, a.messageScreen("Download Model", "local model is incomplete")
			}
			if !modelstore.Check(outcome.result.Path, model.Files).Complete {
				return false, a.messageScreen("Download Model", "downloaded model is incomplete")
			}
			if err := a.downloadComplete(repository.Repo, outcome.result.BytesDownloaded); err != nil {
				return false, err
			}
			return true, nil
		}
	}
}

func (a *app) confirmDownload(repository hf.Repository) (bool, error) {
	selected := 0
	for {
		if !a.sizeOK() {
			exit, err := a.tooSmall()
			if err != nil {
				return false, err
			}
			if exit {
				return false, nil
			}
			continue
		}
		contentWidth := subpageWidth(a.width()) - 2
		lines := []string{fieldRow("Model", repository.Repo, contentWidth), fieldRow("Revision", repository.Revision, contentWidth)}
		if total, known := repository.TotalSize(); known {
			lines = append(lines, fieldRow("Size", formatBytes(total), contentWidth))
		}
		lines = append(lines, "", messageLine("Download this model?", contentWidth, muted), "", actionRow("No", selected == 0, false, actionSecondary, contentWidth), actionRow("Yes", selected == 1, false, actionPrimary, contentWidth))
		a.drawSubpage("Model Download", "DOWNLOAD", lines, subpageFooter("↑↓ Navigate", "Enter Select", "Esc Cancel"), false)
		event, err := a.in.next()
		if err != nil {
			return false, err
		}
		switch event.key {
		case keyCtrlC:
			return false, io.EOF
		case keyEscape:
			return false, nil
		case keyUp, keyDown:
			selected = 1 - selected
		case keyEnter:
			return selected == 1, nil
		}
	}
}

func (a *app) downloadProgress(state modelDownloadState) {
	contentWidth := subpageWidth(a.width()) - 2
	lines := []string{
		fieldRow("Model", state.repository, contentWidth),
		fieldRow("Status", statusText(state.status), contentWidth),
		fieldRow("File", state.progress.CurrentFile, contentWidth),
		fieldRow("Files", fmt.Sprintf("%d / %d", state.progress.FileNumber, state.progress.TotalFiles), contentWidth),
		fieldRow("Progress", formatDownloadProgress(state.progress), contentWidth),
		fieldRow("Speed", formatRate(state.speed), contentWidth),
		fieldRow("Elapsed", formatElapsed(state.elapsed), contentWidth),
	}
	if state.progress.ReusedFiles > 0 {
		lines = append(lines, fieldRow("Reused", fmt.Sprintf("%d completed file(s)", state.progress.ReusedFiles), contentWidth))
	}
	lines = append(lines, "", messageLine(string(`|/-\`[state.activity])+" Receiving data...", contentWidth, cyan))
	a.drawSubpage("Model Download", "DOWNLOAD", lines, "", false)
}

func (a *app) downloadComplete(repo string, downloaded int64) error {
	for {
		if !a.sizeOK() {
			exit, err := a.tooSmall()
			if err != nil || exit {
				return err
			}
			continue
		}
		contentWidth := subpageWidth(a.width()) - 2
		a.drawSubpage("Model Download", "COMPLETE", []string{messageLine("● DOWNLOAD COMPLETE", contentWidth, green), "", fieldRow("Model", repo, contentWidth), fieldRow("Received", formatBytes(downloaded), contentWidth)}, subpageFooter("Enter Continue", "Esc Back"), false)
		event, err := a.in.next()
		if err != nil {
			return err
		}
		if event.key == keyCtrlC {
			return io.EOF
		}
		if event.key == keyEscape || event.key == keyEnter {
			return nil
		}
	}
}

func (a *app) hfDownloadError(repo string, err error) error {
	switch {
	case hf.IsAccessFailure(err):
		return a.hfAccessFailureNotice(err)
	case errors.Is(err, hf.ErrDestinationExists):
		return a.noticeScreen("Model download failed", []string{
			"Model directory already exists but does not match",
			"the required revision.",
		})
	case errors.Is(err, hf.ErrRepositoryNotFound):
		return a.messageScreen("Model download failed", "repository not found")
	case errors.Is(err, hf.ErrRevisionNotFound):
		return a.messageScreen("Model download failed", "revision not found")
	case errors.Is(err, hf.ErrNetworkUnavailable):
		return a.messageScreen("Model download failed", "network unavailable")
	case errors.Is(err, hf.ErrDownloadInProgress):
		return a.messageScreen("Download already in progress", "another download of this model is active")
	default:
		return a.messageScreen("Model download failed", err.Error())
	}
}

// hfAccessFailureNotice renders guidance for a Hugging Face access
// failure. A rejected configured token is never answered with "add a
// token", and a not-found-shaped response to an unauthenticated request
// is never claimed to be a missing revision: Hugging Face hides gated,
// private, and hidden repositories behind such answers.
func (a *app) hfAccessFailureNotice(err error) error {
	switch {
	case errors.Is(err, hf.ErrAccessDenied):
		return a.noticeScreen("Hugging Face access denied", []string{
			"The configured token could not access this model.",
			"Check the token and make sure your account has access",
			"to the gated repository.",
		})
	case errors.Is(err, hf.ErrAuthenticationRequired):
		return a.noticeScreen("Hugging Face access required", []string{
			"No Hugging Face token is configured for this model.",
			"Set a token under Settings → Hugging Face Token, then retry.",
			"",
			"If the repository is gated, make sure your Hugging Face",
			"account has been granted access / accepted its terms.",
		})
	default:
		return a.noticeScreen("Could not access model revision", []string{
			"No Hugging Face token is configured. If this model is",
			"gated or private, set a token under",
			"Settings → Hugging Face Token and retry.",
		})
	}
}

func (a *app) hfAccessRequired() error {
	selected := 0
	for {
		if !a.sizeOK() {
			exit, err := a.tooSmall()
			if err != nil || exit {
				return err
			}
			continue
		}
		contentWidth := subpageWidth(a.width()) - 2
		lines := []string{messageLine("This model requires authenticated Hugging Face access.", contentWidth, muted), "", messageLine("• Set a token under Settings → Hugging Face Token", contentWidth, muted), messageLine("• Have access to the model repository", contentWidth, muted), "", actionRow("Set Token", selected == 0, false, actionPrimary, contentWidth), actionRow("Back", selected == 1, false, actionSecondary, contentWidth)}
		if a.message != "" {
			lines = append(lines, "", messageLine(a.message, contentWidth, red))
		}
		a.drawSubpage("Hugging Face Access", "CREDENTIAL", lines, subpageFooter("↑↓ Navigate", "Enter Select", "Esc Back"), false)
		a.message = ""
		event, err := a.in.next()
		if err != nil {
			return err
		}
		switch event.key {
		case keyCtrlC:
			return io.EOF
		case keyEscape:
			return nil
		case keyUp, keyDown:
			selected = 1 - selected
		case keyEnter:
			if selected == 1 {
				return nil
			}
			saved, err := a.editSecret()
			if err != nil {
				return err
			}
			if saved {
				return nil
			}
		}
	}
}

func (a *app) modelPacksScreen() error {
	selected := 0
	items := []string{"Browse Available Packs", "Installed Packs", "Unused Runtimes", "Import Local Pack", "Back"}
	descriptions := []string{
		"Discover distributable model packs",
		"Inspect installed packs and runtimes",
		"Remove b70ctl-acquired runtimes no pack uses",
		"Install a pack from a local manifest directory",
		"Return to Home",
	}
	for {
		if !a.sizeOK() {
			exit, err := a.tooSmall()
			if err != nil || exit {
				return err
			}
			continue
		}
		contentWidth := subpageWidth(a.width()) - 2
		lines := []string{}
		for index, item := range items {
			lines = append(lines, focusRow(strings.ToUpper(item), index == selected, false, contentWidth))
			if a.height() >= 24 {
				lines = append(lines, "  "+messageLine(descriptions[index], contentWidth-2, muted))
			}
			if index < len(items)-1 {
				lines = append(lines, "")
			}
		}
		if a.message != "" {
			lines = append(lines, "", messageLine(a.message, contentWidth, red))
		}
		a.drawSubpage("Model Packs", "PACKS", lines, subpageFooter("↑↓ Navigate", "Enter Select", "Esc Back"), false)
		a.message = ""
		event, err := a.in.next()
		if err != nil {
			return err
		}
		switch event.key {
		case keyCtrlC:
			return io.EOF
		case keyEscape:
			return nil
		case keyUp:
			selected = previous(selected, len(items))
		case keyDown:
			selected = next(selected, len(items))
		case keyEnter:
			switch selected {
			case 0:
				if err := a.browseAvailablePacksScreen(); err != nil {
					return err
				}
			case 1:
				if err := a.installedPacksScreen(); err != nil {
					return err
				}
			case 2:
				if err := a.unusedRuntimesScreen(); err != nil {
					return err
				}
			case 3:
				path, accepted, err := a.editText("Local pack path:", "")
				if err != nil {
					return err
				}
				if accepted {
					path, err = absolutePath(path)
					if err == nil {
						var manifest *modelpack.Manifest
						manifest, err = modelpack.Load(path)
						if err != nil {
							err = fmt.Errorf("load model pack: %w", err)
						} else if installedPackExists(a.paths.PacksRoot, manifest.ID, manifest.Version) {
							err = fmt.Errorf("pack %s %s is already installed", manifest.ID, manifest.Version)
						} else {
							var inventory []modelstore.Artifact
							inventory, err = modelstore.Scan(a.config.ModelDirectory)
							if err == nil {
								err = a.installPack(path, manifest, inventory, packstore.SourceLocal, "")
							}
						}
					}
					if err != nil {
						a.message = err.Error()
					}
				}
			case 4:
				return nil
			}
		}
	}
}

func (a *app) browseAvailablePacksScreen() error {
	client := newCatalogClient()
	result, err := client.Refresh(context.Background(), a.paths.DataRoot)
	if err != nil {
		return a.noticeScreen("Available Packs", []string{"Model pack catalog is unavailable."})
	}
	selected := 0
	for {
		if !a.sizeOK() {
			exit, err := a.tooSmall()
			if err != nil || exit {
				return err
			}
			continue
		}
		installed, err := packstore.List(a.paths.PacksRoot)
		if err != nil {
			return a.noticeScreen("Available Packs", []string{err.Error()})
		}
		width, height := a.dimensions()
		totalItems := len(result.Catalog.Packs) + 1
		if selected >= totalItems {
			selected = totalItems - 1
		}
		contentWidth := subpageWidth(width) - 2
		lines := []string{}
		if result.Cached {
			lines = append(lines, messageLine("Showing cached catalog.", contentWidth, muted), "")
		}
		if len(result.Catalog.Packs) == 0 {
			lines = append(lines, messageLine("No model packs are available.", contentWidth, muted), "")
		} else {
			visible := max(1, (height-9)/4)
			start := 0
			if len(result.Catalog.Packs) > visible {
				if selected < len(result.Catalog.Packs) && selected >= visible {
					start = selected - visible + 1
				} else if selected >= len(result.Catalog.Packs) {
					start = len(result.Catalog.Packs) - visible
				}
			}
			end := min(len(result.Catalog.Packs), start+visible)
			for index := start; index < end; index++ {
				entry := result.Catalog.Packs[index]
				lines = append(lines,
					focusRow(entry.Name, index == selected, false, contentWidth),
					catalogEntryDetail(entry, packupdate.Classify(installed, entry)),
					"",
				)
			}
			lines = append(lines, listPosition(start, end, len(result.Catalog.Packs))...)
		}
		if a.message != "" {
			lines = append(lines, "", messageLine(a.message, contentWidth, muted))
		}
		actionLines := []string{actionRow("Back", selected == len(result.Catalog.Packs), false, actionSecondary, contentWidth)}
		a.drawSubpagePanels("Available Packs", []subpagePanel{{title: "CATALOG", content: lines}, {title: "ACTIONS", content: actionLines}}, subpageFooter("↑↓ Navigate", "Enter Select", "Esc Back"), false)
		a.message = ""
		event, err := a.in.next()
		if err != nil {
			return err
		}
		switch event.key {
		case keyCtrlC:
			return io.EOF
		case keyEscape:
			return nil
		case keyUp:
			selected = previous(selected, totalItems)
		case keyDown:
			selected = next(selected, totalItems)
		case keyEnter:
			if selected == len(result.Catalog.Packs) {
				return nil
			}
			entry := result.Catalog.Packs[selected]
			state := packupdate.Classify(installed, entry)
			switch state.Status {
			case packupdate.StatusInstalled:
				if err := a.noticeScreen("Model Pack", []string{"Pack is already installed."}); err != nil {
					return err
				}
				continue
			case packupdate.StatusOlderCatalog:
				if err := a.noticeScreen("Model Pack", []string{
					fmt.Sprintf("Version %s is already installed.", state.Current),
					"The catalog entry is older, so a downgrade is not offered.",
				}); err != nil {
					return err
				}
				continue
			case packupdate.StatusFinishUpdate:
				if err := a.packFinishUpdateScreen(entry, state); err != nil {
					return err
				}
				continue
			case packupdate.StatusUpdateAvailable:
				if err := a.packUpdateScreen(client, entry, state); err != nil {
					return err
				}
				continue
			case packupdate.StatusIncomparable:
				if err := a.noticeScreen("Model Pack", []string{
					state.Reason,
					"Falling back to an exact-version install.",
				}); err != nil {
					return err
				}
			}
			acquired, err := a.downloadRemotePack(client, entry, "Install Pack")
			if err != nil {
				if err := a.noticeScreen("Download Model Pack", []string{err.Error()}); err != nil {
					return err
				}
				continue
			}
			inventory, scanErr := modelstore.Scan(a.config.ModelDirectory)
			if scanErr == nil {
				scanErr = a.installPack(acquired.Path, acquired.Manifest, inventory, packstore.SourceRemote, "")
			}
			closeErr := acquired.Close()
			if scanErr != nil {
				return scanErr
			}
			if closeErr != nil {
				return closeErr
			}
		}
	}
}

func (a *app) downloadRemotePack(client *catalog.Client, entry catalog.Entry, title string) (*catalog.AcquiredPack, error) {
	started := time.Now()
	frame := 0
	draw := func(now time.Time) {
		contentWidth := subpageWidth(a.width()) - 2
		a.drawSubpage(title, "DOWNLOAD", []string{fieldRow("Pack", entry.Name, contentWidth), fieldRow("Version", entry.Version, contentWidth), fieldRow("Status", statusText("Downloading"), contentWidth), fieldRow("Elapsed", formatElapsed(now.Sub(started)), contentWidth), "", messageLine(string(`|/-\`[frame])+" Working...", contentWidth, cyan)}, "", false)
	}
	finished := make(chan remotePackOutcome, 1)
	go func() {
		pack, err := client.Acquire(context.Background(), a.paths.DataRoot, entry)
		finished <- remotePackOutcome{pack: pack, err: err}
	}()
	draw(started)
	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()
	for {
		select {
		case now := <-ticker.C:
			frame = (frame + 1) % len(`|/-\`)
			draw(now)
		case outcome := <-finished:
			return outcome.pack, outcome.err
		}
	}
}

func catalogEntryInstalled(installed []packstore.InstalledPack, entry catalog.Entry) bool {
	for _, pack := range installed {
		if pack.ID == entry.ID && pack.Version == entry.Version {
			return true
		}
	}
	return false
}

// catalogEntryDetail renders the version/status line for one Available Packs
// row: updates show the installed → available transition, and a newest
// version with old versions still present asks the user to finish the
// update.
func catalogEntryDetail(entry catalog.Entry, state packupdate.State) string {
	plain := "    " + muted + "v" + entry.Version + "     " + reset
	switch state.Status {
	case packupdate.StatusUpdateAvailable:
		return "    " + muted + "v" + state.Current + " → v" + entry.Version + "     " + reset + statusText("Update Available")
	case packupdate.StatusFinishUpdate:
		return "    " + muted + "v" + entry.Version + " (old " + strings.Join(state.OldVersions, ", ") + ")" + "     " + reset + statusText("Finish Update")
	case packupdate.StatusOlderCatalog:
		return plain + statusText("Newer Installed")
	default:
		status := "Available"
		if state.Status == packupdate.StatusInstalled {
			status = "Installed"
		}
		return plain + statusText(status)
	}
}

// errUpdatePrepareShown marks an update whose preparation failure was already
// displayed by the normal install flow, so the update wrapper only has to
// report that the previous version stays installed.
var errUpdatePrepareShown = errors.New("update preparation failed")

// packUpdateScreen identifies the installed and catalog versions for one
// update and offers to run it. The update itself acquires and validates the
// new pack, prepares it through the normal install flow, and only then
// retires the old version.
func (a *app) packUpdateScreen(client *catalog.Client, entry catalog.Entry, state packupdate.State) error {
	selected := 0
	for {
		if !a.sizeOK() {
			exit, err := a.tooSmall()
			if err != nil || exit {
				return err
			}
			continue
		}
		contentWidth := subpageWidth(a.width()) - 2
		packLines := []string{
			fieldRow("Pack", entry.Name, contentWidth),
			fieldRow("Current", "v"+state.Current, contentWidth),
			fieldRow("Available", "v"+entry.Version, contentWidth),
		}
		actionLines := []string{
			actionRow("Update", selected == 0, false, actionPrimary, contentWidth),
			actionRow("Back", selected == 1, false, actionSecondary, contentWidth),
		}
		a.drawSubpagePanels("Update Pack", []subpagePanel{{title: "PACK", content: packLines}, {title: "ACTIONS", content: actionLines}}, subpageFooter("↑↓ Navigate", "Enter Select", "Esc Back"), false)
		event, err := a.in.next()
		if err != nil {
			return err
		}
		switch event.key {
		case keyCtrlC:
			return io.EOF
		case keyEscape:
			return nil
		case keyUp, keyDown:
			selected = 1 - selected
		case keyEnter:
			if selected == 1 {
				return nil
			}
			acquired, err := a.downloadRemotePack(client, entry, "Update Pack")
			if err != nil {
				if err := a.noticeScreen("Update Pack", []string{err.Error()}); err != nil {
					return err
				}
				continue
			}
			inventory, scanErr := modelstore.Scan(a.config.ModelDirectory)
			if scanErr == nil {
				scanErr = a.installPack(acquired.Path, acquired.Manifest, inventory, packstore.SourceRemote, state.Current)
			}
			closeErr := acquired.Close()
			if scanErr != nil {
				return scanErr
			}
			if closeErr != nil {
				return closeErr
			}
			return nil
		}
	}
}

func (a *app) confirmPackUpdate(manifest *modelpack.Manifest, plan install.Plan, current string) (bool, error) {
	selected := 0
	counts := plan.Counts()
	for {
		if !a.sizeOK() {
			exit, err := a.tooSmall()
			if err != nil {
				return false, err
			}
			if exit {
				return false, nil
			}
			continue
		}
		contentWidth := subpageWidth(a.width()) - 2
		lines := []string{
			messageLine(fmt.Sprintf("Update %s to %s?", manifest.Name, manifest.Version), contentWidth, muted), "",
			fieldRow("Current", "v"+current, contentWidth),
			fieldRow("Available", "v"+manifest.Version, contentWidth),
			fieldRow("Selected targets", strconv.Itoa(counts.SelectedTargets), contentWidth),
			fieldRow("Existing", strconv.Itoa(counts.Existing), contentWidth),
			fieldRow("Incomplete", strconv.Itoa(counts.Incomplete), contentWidth),
			fieldRow("Downloads", strconv.Itoa(counts.Downloads), contentWidth),
			fieldRow("Skipped", strconv.Itoa(counts.Skipped), contentWidth),
			"",
			actionRow("No", selected == 0, false, actionSecondary, contentWidth),
			actionRow("Yes", selected == 1, false, actionPrimary, contentWidth),
		}
		a.drawSubpage("Update Pack", "REVIEW", lines, subpageFooter("↑↓ Navigate", "Enter Select", "Esc Cancel"), false)
		event, err := a.in.next()
		if err != nil {
			return false, err
		}
		switch event.key {
		case keyCtrlC:
			return false, io.EOF
		case keyEscape:
			return false, nil
		case keyUp, keyDown:
			selected = 1 - selected
		case keyEnter:
			return selected == 1, nil
		}
	}
}

// updatePack completes an update after target selection: prepare the new
// version (which installs it into the pack store and verifies its selected
// artifacts), roll the new version back unless that fully succeeded, and only
// then retire the old version through the shared uninstall engine.
func (a *app) updatePack(sourcePath string, manifest *modelpack.Manifest, plan install.Plan, current string) error {
	yes, err := a.confirmPackUpdate(manifest, plan, current)
	if err != nil || !yes {
		return err
	}
	contentWidth := subpageWidth(a.width()) - 2
	a.drawSubpage("Update Pack", "UPDATING", []string{
		fieldRow("Pack", manifest.Name, contentWidth),
		fieldRow("From", "v"+current, contentWidth),
		fieldRow("To", "v"+manifest.Version, contentWidth),
		"",
		messageLine("Preparing the new version…", contentWidth, cyan),
	}, "", false)
	outcome := packupdate.Apply(a.paths.PacksRoot, a.paths.DataRoot, a.config.ModelDirectory, manifest.ID, manifest.Version, func() (install.Result, error) {
		result, prepared, err := a.preparePack("Update Pack", sourcePath, plan, packstore.SourceRemote)
		if err != nil {
			return install.Result{}, err
		}
		if !prepared {
			return install.Result{}, errUpdatePrepareShown
		}
		return result, nil
	})
	switch {
	case errors.Is(outcome.Err, io.EOF):
		return outcome.Err
	case outcome.Outcome == packupdate.OutcomeUpdated, outcome.Outcome == packupdate.OutcomeRetireIncomplete:
		return a.packUpdateResultScreen(manifest, current, outcome)
	case outcome.Outcome == packupdate.OutcomePrepareIncomplete:
		return a.packUpdateResultScreen(manifest, current, outcome)
	case errors.Is(outcome.Err, errUpdatePrepareShown):
		a.message = "Update did not complete — version " + current + " is still installed."
		return nil
	default:
		return a.noticeScreen("Update Pack", []string{
			"Update did not complete — version " + current + " is still installed.",
			outcome.Err.Error(),
		})
	}
}

func (a *app) packUpdateResultScreen(manifest *modelpack.Manifest, current string, outcome packupdate.Result) error {
	token, _, err := hf.Get()
	if err != nil {
		return err
	}
	for {
		if !a.sizeOK() {
			exit, err := a.tooSmall()
			if err != nil || exit {
				return err
			}
			continue
		}
		contentWidth := subpageWidth(a.width()) - 2
		styled := []string{}
		for _, line := range packUpdateSummary(manifest, current, outcome, token != "") {
			styled = append(styled, messageLine(line, contentWidth, muted))
		}
		a.drawSubpage("Update Pack", "RESULT", styled, subpageFooter("Enter Continue", "Esc Back"), false)
		event, err := a.in.next()
		if err != nil {
			return err
		}
		if event.key == keyCtrlC {
			return io.EOF
		}
		if event.key == keyEscape || event.key == keyEnter {
			return nil
		}
	}
}

// packUpdateSummary renders the update result screen contents: the completed
// update with what retiring the old version cleaned up, or — when the new
// version could not be prepared completely — the kept previous version with
// the per-artifact outcomes.
func packUpdateSummary(manifest *modelpack.Manifest, current string, outcome packupdate.Result, tokenConfigured bool) []string {
	if outcome.Outcome == packupdate.OutcomeUpdated || outcome.Outcome == packupdate.OutcomeRetireIncomplete {
		lines := []string{green + "Pack updated to " + manifest.Version + "." + reset, ""}
		for _, retired := range outcome.Retired {
			for _, item := range retired.Items {
				lines = append(lines, item.Artifact, item.Outcome, "")
			}
		}
		if outcome.Outcome == packupdate.OutcomeRetireIncomplete {
			lines = append(lines,
				"An older pack version could not be removed:",
				outcome.Err.Error(),
				"",
				"Finish the update later from Available Packs.",
				"",
			)
		}
		return append(lines, "Press Enter to continue.")
	}
	lines := []string{
		red + "Update did not complete." + reset,
		"Version " + current + " is still installed.",
		"",
	}
	if outcome.Err != nil && !errors.Is(outcome.Err, errUpdatePrepareShown) {
		lines = append(lines, outcome.Err.Error(), "")
	}
	return append(lines, packArtifactSummary(outcome.Prepared, tokenConfigured)...)
}

// packFinishUpdateScreen resolves the dual-version state left by an
// interrupted update (or by installing a newer pack version alongside an
// older one before updates existed): the newest version is already installed
// and only the old versions remain to retire. No new download happens here.
func (a *app) packFinishUpdateScreen(entry catalog.Entry, state packupdate.State) error {
	selected := 0
	for {
		if !a.sizeOK() {
			exit, err := a.tooSmall()
			if err != nil || exit {
				return err
			}
			continue
		}
		contentWidth := subpageWidth(a.width()) - 2
		packLines := []string{
			fieldRow("Pack", entry.Name, contentWidth),
			fieldRow("Installed", "v"+entry.Version, contentWidth),
			fieldRow("Old version", "v"+strings.Join(state.OldVersions, ", v"), contentWidth),
		}
		detailLines := []string{messageLine("Version "+entry.Version+" is installed; an older version is still present.", contentWidth, muted)}
		actionLines := []string{
			actionRow("Remove Old Version", selected == 0, false, actionDestructive, contentWidth),
			actionRow("Back", selected == 1, false, actionSecondary, contentWidth),
		}
		a.drawSubpagePanels("Update Pack", []subpagePanel{{title: "PACK", content: packLines}, {title: "DETAILS", content: detailLines}, {title: "ACTIONS", content: actionLines}}, subpageFooter("↑↓ Navigate", "Enter Select", "Esc Back"), false)
		event, err := a.in.next()
		if err != nil {
			return err
		}
		switch event.key {
		case keyCtrlC:
			return io.EOF
		case keyEscape:
			return nil
		case keyUp, keyDown:
			selected = 1 - selected
		case keyEnter:
			if selected == 1 {
				return nil
			}
			yes, err := a.confirm("Remove old pack version " + strings.Join(state.OldVersions, ", ") + "?")
			if err != nil {
				return err
			}
			if !yes {
				continue
			}
			contentWidth := subpageWidth(a.width()) - 2
			a.drawSubpage("Update Pack", "REMOVING", []string{fieldRow("Pack", entry.Name, contentWidth), "", messageLine("Removing the old pack version…", contentWidth, red)}, "", false)
			retired, err := packupdate.RetireOlder(a.paths.PacksRoot, a.paths.DataRoot, a.config.ModelDirectory, entry.ID, entry.Version)
			if err != nil {
				if err := a.messageScreen("Update Pack", err.Error()); err != nil {
					return err
				}
				continue
			}
			return a.packRetireResultScreen(entry, retired)
		}
	}
}

func (a *app) packRetireResultScreen(entry catalog.Entry, retired []uninstall.Result) error {
	for {
		if !a.sizeOK() {
			exit, err := a.tooSmall()
			if err != nil || exit {
				return err
			}
			continue
		}
		contentWidth := subpageWidth(a.width()) - 2
		lines := []string{
			messageLine("● OLD VERSION REMOVED", contentWidth, green),
			"",
			messageLine("Old pack version removed.", contentWidth, muted),
			messageLine("Current version: "+entry.Version, contentWidth, muted),
			"",
		}
		for _, result := range retired {
			for _, item := range result.Items {
				lines = append(lines, messageLine(item.Artifact, contentWidth, muted), messageLine(item.Outcome, contentWidth, muted))
			}
		}
		lines = append(lines, "", messageLine("Press Enter to continue.", contentWidth, muted))
		a.drawSubpage("Update Pack", "RESULT", lines, subpageFooter("Enter Continue", "Esc Back"), false)
		event, err := a.in.next()
		if err != nil {
			return err
		}
		if event.key == keyCtrlC {
			return io.EOF
		}
		if event.key == keyEscape || event.key == keyEnter {
			return nil
		}
	}
}

// installPack runs the pack installation flow: target selection, gated-model
// preflight, confirmation, and artifact preparation. updateCurrent is the
// currently installed version when this is an update ("" for a plain
// install); updates confirm with update wording and finish by retiring the
// old version only after the new one is fully prepared.
func (a *app) installPack(sourcePath string, manifest *modelpack.Manifest, inventory []modelstore.Artifact, source string, updateCurrent string) error {
	selection := localPackSelection{targets: install.Targets(manifest), selected: map[string]bool{}}
	focused := 0
	for {
		if !a.sizeOK() {
			exit, err := a.tooSmall()
			if err != nil || exit {
				return err
			}
			continue
		}
		totalItems := len(selection.targets) + 2
		if len(selection.targets) == 0 {
			focused = 0
		}
		contentWidth := subpageWidth(a.width()) - 2
		header := manifest.Name + "  v" + manifest.Version
		if updateCurrent != "" {
			header = manifest.Name + "  v" + updateCurrent + " → v" + manifest.Version
		}
		lines := []string{messageLine(header, contentWidth, muted), ""}
		for index, target := range selection.targets {
			state := string(modelstore.Assess(inventory, target).State)
			mark := "[ ]"
			if selection.selected[target.ID] {
				mark = "[x]"
			}
			nameWidth := max(1, contentWidth-24)
			label := fmt.Sprintf("%s %-*s", mark, nameWidth, truncate(target.Name, nameWidth))
			lines = append(lines, focusRow(label, index == focused, false, contentWidth), "    "+statusText(state))
		}
		lines = append(lines, "", actionRow("Continue", focused == len(selection.targets), false, actionPrimary, contentWidth), actionRow("Back", focused == len(selection.targets)+1, false, actionSecondary, contentWidth))
		title := "Install Pack"
		if updateCurrent != "" {
			title = "Update Pack"
		}
		a.drawSubpage(title, "TARGETS", lines, subpageFooter("↑↓ Navigate", "Enter Toggle/Select", "Esc Back"), false)
		event, err := a.in.next()
		if err != nil {
			return err
		}
		switch event.key {
		case keyCtrlC:
			return io.EOF
		case keyEscape:
			return nil
		case keyUp:
			focused = previous(focused, totalItems)
		case keyDown:
			focused = next(focused, totalItems)
		case keyEnter:
			switch {
			case focused < len(selection.targets):
				selection.toggle(focused)
			case focused == len(selection.targets):
				plan, err := install.BuildPlan(manifest, selection.ids(), inventory)
				if err != nil {
					return err
				}
				continueInstall, err := a.packGatedPreflight(&plan)
				if err != nil {
					return err
				}
				if !continueInstall {
					return nil
				}
				if updateCurrent != "" {
					return a.updatePack(sourcePath, manifest, plan, updateCurrent)
				}
				yes, err := a.confirmPackInstall(manifest, plan)
				if err != nil || !yes {
					return err
				}
				result, prepared, err := a.preparePack(title, sourcePath, plan, source)
				if err != nil {
					return err
				}
				if !prepared {
					return nil
				}
				return a.packInstallResult(result)
			default:
				return nil
			}
		}
	}
}

func (a *app) packGatedPreflight(plan *install.Plan) (bool, error) {
	for plan.HasMissingGated() {
		token, _, err := hf.Get()
		if err != nil {
			return false, err
		}
		if token != "" {
			return true, nil
		}
		selected := 0
		for {
			if !a.sizeOK() {
				exit, err := a.tooSmall()
				if err != nil || exit {
					return false, err
				}
				continue
			}
			contentWidth := subpageWidth(a.width()) - 2
			lines := []string{messageLine("Some selected models require authenticated Hugging Face access.", contentWidth, muted), "", messageLine("• Set a token under Settings → Hugging Face Token", contentWidth, muted), messageLine("• Have access to the model repository", contentWidth, muted), "", actionRow("Set Token", selected == 0, false, actionPrimary, contentWidth), actionRow("Continue Without Gated Models", selected == 1, false, actionSecondary, contentWidth), actionRow("Back", selected == 2, false, actionSecondary, contentWidth)}
			if a.message != "" {
				lines = append(lines, "", messageLine(a.message, contentWidth, red))
			}
			a.drawSubpage("Install Pack", "PREFLIGHT", lines, subpageFooter("↑↓ Navigate", "Enter Select", "Esc Back"), false)
			a.message = ""
			event, err := a.in.next()
			if err != nil {
				return false, err
			}
			switch event.key {
			case keyCtrlC:
				return false, io.EOF
			case keyEscape:
				return false, nil
			case keyUp:
				selected = previous(selected, 3)
			case keyDown:
				selected = next(selected, 3)
			case keyEnter:
				switch selected {
				case 0:
					saved, err := a.editSecret()
					if err != nil {
						return false, err
					}
					if saved {
						return true, nil
					}
				case 1:
					plan.SkipMissingGated("Hugging Face token not configured")
					return true, nil
				case 2:
					return false, nil
				}
			}
		}
	}
	return true, nil
}

func (a *app) confirmPackInstall(manifest *modelpack.Manifest, plan install.Plan) (bool, error) {
	selected := 0
	counts := plan.Counts()
	for {
		if !a.sizeOK() {
			exit, err := a.tooSmall()
			if err != nil {
				return false, err
			}
			if exit {
				return false, nil
			}
			continue
		}
		contentWidth := subpageWidth(a.width()) - 2
		lines := []string{messageLine(fmt.Sprintf("Install %s %s?", manifest.Name, manifest.Version), contentWidth, muted), "", fieldRow("Selected targets", strconv.Itoa(counts.SelectedTargets), contentWidth), fieldRow("Existing", strconv.Itoa(counts.Existing), contentWidth), fieldRow("Incomplete", strconv.Itoa(counts.Incomplete), contentWidth), fieldRow("Downloads", strconv.Itoa(counts.Downloads), contentWidth), fieldRow("Skipped", strconv.Itoa(counts.Skipped), contentWidth), "", actionRow("No", selected == 0, false, actionSecondary, contentWidth), actionRow("Yes", selected == 1, false, actionPrimary, contentWidth)}
		a.drawSubpage("Install Pack", "REVIEW", lines, subpageFooter("↑↓ Navigate", "Enter Select", "Esc Cancel"), false)
		event, err := a.in.next()
		if err != nil {
			return false, err
		}
		switch event.key {
		case keyCtrlC:
			return false, io.EOF
		case keyEscape:
			return false, nil
		case keyUp, keyDown:
			selected = 1 - selected
		case keyEnter:
			return selected == 1, nil
		}
	}
}

// preparePack runs the artifact preparation for a pack and reports whether
// preparation ran to completion. Failures the flow displays itself (access
// notices, error screens) return prepared=false with a nil error so callers
// can react without double-reporting; only terminal errors are returned.
func (a *app) preparePack(title, sourcePath string, plan install.Plan, source string) (install.Result, bool, error) {
	token, _, err := hf.Get()
	if err != nil {
		return install.Result{}, false, err
	}
	a.drawSubpage(title, "PREPARING", []string{messageLine("Preparing pack…", subpageWidth(a.width())-2, cyan)}, "", false)
	events := make(chan install.Event, 1)
	finished := make(chan preparationOutcome, 1)
	client := hf.NewClient(token)
	go func() {
		result, prepareErr := install.Prepare(context.Background(), sourcePath, a.paths.PacksRoot, a.paths.DataRoot, a.config.ModelDirectory, source, plan, client, func(event install.Event) {
			sendLatestInstallEvent(events, event)
		})
		finished <- preparationOutcome{result: result, err: prepareErr}
	}()

	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()
	var modelState *modelDownloadState
	var runtimeState *runtimeDownloadState
	for {
		select {
		case event := <-events:
			if event.Runtime != nil {
				if runtimeState == nil || runtimeState.declared.Digest != event.Runtime.Digest {
					value := newRuntimeDownloadState(*event.Runtime, time.Now())
					runtimeState = &value
				}
				runtimeState.update(event.RuntimeProgress)
				a.runtimeDownloadProgress(*runtimeState)
				modelState = nil
			} else {
				modelState = a.showInstallEvent(title, modelState, event)
				runtimeState = nil
			}
		case now := <-ticker.C:
			if runtimeState != nil {
				runtimeState.refresh(now)
				a.runtimeDownloadProgress(*runtimeState)
			} else if modelState != nil {
				modelState.refresh(now)
				a.downloadProgress(*modelState)
			}
		case outcome := <-finished:
			select {
			case event := <-events:
				if event.Runtime != nil {
					if runtimeState == nil || runtimeState.declared.Digest != event.Runtime.Digest {
						value := newRuntimeDownloadState(*event.Runtime, time.Now())
						runtimeState = &value
					}
					runtimeState.update(event.RuntimeProgress)
				} else {
					modelState = a.showInstallEvent(title, modelState, event)
				}
			default:
			}
			if outcome.err != nil {
				if hf.IsAccessFailure(outcome.err) {
					if err := a.hfAccessFailureNotice(outcome.err); err != nil {
						return install.Result{}, false, err
					}
					return install.Result{}, false, nil
				}
				if err := a.messageScreen(title, outcome.err.Error()); err != nil {
					return install.Result{}, false, err
				}
				return install.Result{}, false, nil
			}
			return outcome.result, true, nil
		}
	}
}

func (a *app) showInstallEvent(title string, state *modelDownloadState, event install.Event) *modelDownloadState {
	if event.Repository == nil {
		contentWidth := subpageWidth(a.width()) - 2
		a.drawSubpage(title, "PREPARING", []string{fieldRow("Model", event.Artifact.Repo, contentWidth), "", messageLine("Retrieving exact revision metadata…", contentWidth, muted)}, "", false)
		return nil
	}
	if state == nil || state.repository != event.Repository.Repo {
		value := newModelDownloadState(*event.Repository, time.Now())
		state = &value
		state.update(event.Progress)
		a.downloadProgress(*state)
		return state
	}
	state.update(event.Progress)
	return state
}

func (a *app) packInstallResult(result install.Result) error {
	token, _, err := hf.Get()
	if err != nil {
		return err
	}
	for {
		if !a.sizeOK() {
			exit, err := a.tooSmall()
			if err != nil || exit {
				return err
			}
			continue
		}
		contentWidth := subpageWidth(a.width()) - 2
		styled := []string{}
		for _, line := range packInstallSummary(result, token != "") {
			styled = append(styled, messageLine(line, contentWidth, muted))
		}
		a.drawSubpage("Install Pack", "RESULT", styled, subpageFooter("Enter Continue", "Esc Back"), false)
		event, err := a.in.next()
		if err != nil {
			return err
		}
		if event.key == keyCtrlC {
			return io.EOF
		}
		if event.key == keyEscape || event.key == keyEnter {
			return nil
		}
	}
}

func packInstallSummary(result install.Result, tokenConfigured bool) []string {
	lines := []string{blue + "Pack installed." + reset, ""}
	return append(lines, packArtifactSummary(result, tokenConfigured)...)
}

// packArtifactSummary renders the per-artifact outcomes and retry guidance
// shared by the install and update result screens.
func packArtifactSummary(result install.Result, tokenConfigured bool) []string {
	lines := []string{}
	if len(result.Items) == 0 && len(result.RuntimeItems) == 0 {
		lines = append(lines,
			"No models were downloaded.",
			"You can add them later from Models.",
			"",
			"Press Enter to continue.",
		)
		return lines
	}
	failed := 0
	accessIssue := result.HasAccessIssue()
	for _, item := range result.Items {
		outcome := string(item.Outcome)
		if item.Reason != "" {
			outcome += " — " + item.Reason
		}
		lines = append(lines, item.Artifact.Name, outcome, "")
		if item.Outcome == install.Failed {
			failed++
		}
	}
	for _, item := range result.RuntimeItems {
		label := "Runtime"
		if len(result.RuntimeItems) > 1 {
			label += " " + item.Runtime.ID
		}
		outcome := string(item.Outcome)
		if item.Reason != "" {
			outcome += " — " + item.Reason
		}
		lines = append(lines, label, outcome, "")
	}
	if failed > 0 {
		label := "artifacts"
		if failed == 1 {
			label = "artifact"
		}
		lines = append(lines,
			fmt.Sprintf("%d %s could not be acquired.", failed, label),
		)
		if !accessIssue {
			lines = append(lines, "Retry missing models later from Models.")
		}
		lines = append(lines, "")
	}
	if accessIssue {
		lines = append(lines,
			"Hugging Face access issue",
			"",
		)
		if tokenConfigured {
			lines = append(lines,
				"Check that:",
				"- your token is valid",
				"- your account has access to the required repository",
			)
		} else {
			lines = append(lines,
				"No Hugging Face token is configured.",
				"Set one under Settings → Hugging Face Token, then retry.",
			)
		}
		lines = append(lines,
			"",
			"Missing models can be retried later from Models.",
			"",
		)
	}
	if result.HasRuntimeFailure() {
		lines = append(lines,
			"Models can still be managed from Models.",
			"Runtime can be retried later from Installed Packs.",
			"",
		)
	}
	lines = append(lines, "Press Enter to continue.")
	return lines
}

func (selection *localPackSelection) toggle(index int) {
	id := selection.targets[index].ID
	selection.selected[id] = !selection.selected[id]
}

func (selection localPackSelection) ids() []string {
	var ids []string
	for _, target := range selection.targets {
		if selection.selected[target.ID] {
			ids = append(ids, target.ID)
		}
	}
	return ids
}

func installedPackExists(storeRoot, id, version string) bool {
	packs, err := packstore.List(storeRoot)
	if err != nil {
		return false
	}
	for _, pack := range packs {
		if pack.ID == id && pack.Version == version {
			return true
		}
	}
	return false
}

func sendLatestInstallEvent(events chan install.Event, event install.Event) {
	select {
	case events <- event:
	default:
		select {
		case <-events:
		default:
		}
		select {
		case events <- event:
		default:
		}
	}
}

func (a *app) installedPacksScreen() error {
	selected := 0
	for {
		if !a.sizeOK() {
			exit, err := a.tooSmall()
			if err != nil || exit {
				return err
			}
			continue
		}
		packs, err := a.loadPacks()
		width, height := a.dimensions()
		contentWidth := subpageWidth(width) - 2
		lines := []string{}
		totalItems := len(packs) + 1
		if selected >= totalItems {
			selected = totalItems - 1
		}
		if err != nil {
			lines = append(lines, messageLine("● ERROR", contentWidth, red), messageLine(err.Error(), contentWidth, muted))
		} else if len(packs) == 0 {
			lines = append(lines, messageLine("No model packs are installed.", contentWidth, muted))
		} else {
			visible := max(1, (height-7)/4)
			start := 0
			if len(packs) > visible {
				if selected < len(packs) && selected >= visible {
					start = selected - visible + 1
				} else if selected >= len(packs) {
					start = len(packs) - visible
				}
			}
			end := min(len(packs), start+visible)
			for index := start; index < end; index++ {
				pack := packs[index]
				lines = append(lines,
					focusRow(pack.installed.Name, index == selected, false, contentWidth),
					"    "+muted+"v"+pack.installed.Version+"     "+strings.ToUpper(sourceName(pack.installed.Source))+reset,
					"",
				)
			}
			lines = append(lines, listPosition(start, end, len(packs))...)
		}
		actionLines := []string{actionRow("Back", selected == len(packs), false, actionSecondary, contentWidth)}
		a.drawSubpagePanels("Installed Packs", []subpagePanel{{title: "INVENTORY", content: lines}, {title: "ACTIONS", content: actionLines}}, subpageFooter("↑↓ Navigate", "Enter Select", "Esc Back"), false)
		event, err := a.in.next()
		if err != nil {
			return err
		}
		switch event.key {
		case keyCtrlC:
			return io.EOF
		case keyEscape:
			return nil
		case keyUp:
			selected = previous(selected, totalItems)
		case keyDown:
			selected = next(selected, totalItems)
		case keyEnter:
			if selected == len(packs) {
				return nil
			}
			removed, err := a.installedPackScreen(packs[selected])
			if err != nil {
				return err
			}
			if removed && selected >= len(packs)-1 && selected > 0 {
				selected--
			}
		}
	}
}

func (a *app) installedPackScreen(pack loadedPack) (bool, error) {
	selected := 0
	for {
		if !a.sizeOK() {
			exit, err := a.tooSmall()
			if err != nil || exit {
				return false, err
			}
			continue
		}
		runtimeState, downloadable := packRuntimeStatus(pack.manifest)
		actions := []string{"Uninstall", "Back"}
		if len(downloadable) > 0 {
			actions = []string{"Download Runtime", "Uninstall", "Back"}
		}
		if selected >= len(actions) {
			selected = len(actions) - 1
		}
		contentWidth := subpageWidth(a.width()) - 2
		packLines := []string{
			fieldRow("Name", pack.installed.Name, contentWidth),
			fieldRow("Version", pack.installed.Version, contentWidth),
			fieldRow("Source", sourceName(pack.installed.Source), contentWidth),
			fieldRow("Runtime", statusText(runtimeState), contentWidth),
		}
		actionLines := []string{}
		for index, action := range actions {
			style := actionSecondary
			if action == "Download Runtime" {
				style = actionPrimary
			} else if action == "Uninstall" {
				style = actionDestructive
			}
			actionLines = append(actionLines, actionRow(action, index == selected, false, style, contentWidth))
		}
		a.drawSubpagePanels("Model Pack", []subpagePanel{{title: "PACK", content: packLines}, {title: "ACTIONS", content: actionLines}}, subpageFooter("↑↓ Navigate", "Enter Select", "Esc Back"), false)
		event, err := a.in.next()
		if err != nil {
			return false, err
		}
		switch event.key {
		case keyCtrlC:
			return false, io.EOF
		case keyEscape:
			return false, nil
		case keyUp:
			selected = previous(selected, len(actions))
		case keyDown:
			selected = next(selected, len(actions))
		case keyEnter:
			switch actions[selected] {
			case "Back":
				return false, nil
			case "Download Runtime":
				if err := a.downloadPackRuntimes(downloadable); err != nil {
					return false, err
				}
				selected = 0
			case "Uninstall":
				return a.uninstallPack(pack)
			}
		}
	}
}

func packRuntimeStatus(manifest *modelpack.Manifest) (string, []modelpack.Runtime) {
	return packRuntimeStatusWithInspect(manifest, runtime.InspectRuntime)
}

func packRuntimeStatusWithInspect(manifest *modelpack.Manifest, inspect func(modelpack.Runtime) runtime.ImageResult) (string, []modelpack.Runtime) {
	seen := map[string]struct{}{}
	var missing []modelpack.Runtime
	anyMissing := false
	for _, declared := range manifest.Runtimes {
		if _, exists := seen[declared.Digest]; exists {
			continue
		}
		seen[declared.Digest] = struct{}{}
		if inspect(declared).Status == runtime.ImagePresent {
			continue
		}
		anyMissing = true
		if declared.Registry != "" {
			missing = append(missing, declared)
		}
	}
	if anyMissing {
		return "Missing", missing
	}
	return "Present", nil
}

func (a *app) downloadPackRuntimes(runtimes []modelpack.Runtime) error {
	for _, declared := range runtimes {
		result, err := a.downloadRuntime(declared)
		if err != nil {
			return err
		}
		if err := a.runtimeDownloadResult(declared, result); err != nil {
			return err
		}
	}
	return nil
}

func (a *app) downloadRuntime(declared modelpack.Runtime) (runtime.AcquisitionResult, error) {
	state := newRuntimeDownloadState(declared, time.Now())
	progresses := make(chan runtime.PullProgress, 1)
	finished := make(chan runtimeDownloadOutcome, 1)
	go func() {
		result := acquireRuntime(context.Background(), a.paths.DataRoot, declared, func(progress runtime.PullProgress) {
			sendLatestRuntimeProgress(progresses, progress)
		})
		finished <- runtimeDownloadOutcome{result: result}
	}()

	a.runtimeDownloadProgress(state)
	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()
	for {
		select {
		case progress := <-progresses:
			state.update(progress)
			a.runtimeDownloadProgress(state)
		case now := <-ticker.C:
			drainRuntimeProgress(progresses, &state)
			state.refresh(now)
			a.runtimeDownloadProgress(state)
		case outcome := <-finished:
			drainRuntimeProgress(progresses, &state)
			return outcome.result, nil
		}
	}
}

func (a *app) runtimeDownloadProgress(state runtimeDownloadState) {
	contentWidth := subpageWidth(a.width()) - 2
	lines := []string{fieldRow("Image", runtimeImageName(state.declared), contentWidth), fieldRow("Status", statusText(state.status), contentWidth), fieldRow("Activity", state.activity, contentWidth), fieldRow("Elapsed", formatElapsed(state.elapsed), contentWidth)}
	if state.latest != "" {
		lines = append(lines, "", fieldRow("Latest", state.latest, contentWidth))
	}
	lines = append(lines, "", messageLine(string(`|/-\`[state.frame])+" Working...", contentWidth, cyan))
	a.drawSubpage("Install Pack", "RUNTIME", lines, "", false)
}

func (a *app) runtimeDownloadResult(declared modelpack.Runtime, result runtime.AcquisitionResult) error {
	message := string(result.Outcome)
	if result.Reason != "" {
		message += " — " + result.Reason
	}
	return a.noticeScreen("Runtime", []string{runtimeImageName(declared), "", message})
}

func newRuntimeDownloadState(declared modelpack.Runtime, now time.Time) runtimeDownloadState {
	return runtimeDownloadState{
		declared: declared,
		status:   "Pulling",
		activity: "Checking local image",
		started:  now,
	}
}

func (state *runtimeDownloadState) update(progress runtime.PullProgress) {
	if progress.Status != "" {
		state.status = progress.Status
	}
	if progress.Activity != "" {
		state.activity = progress.Activity
	}
	if progress.Latest != "" {
		state.latest = progress.Latest
	}
}

func (state *runtimeDownloadState) refresh(now time.Time) {
	state.elapsed = now.Sub(state.started)
	if state.elapsed < 0 {
		state.elapsed = 0
	}
	state.frame = (state.frame + 1) % len(`|/-\`)
}

func runtimeImageName(declared modelpack.Runtime) string {
	value := declared.Registry
	if value == "" {
		value = declared.Image
	}
	value, _, _ = strings.Cut(value, "@")
	if slash := strings.LastIndex(value, "/"); slash >= 0 {
		value = value[slash+1:]
	}
	return value
}

func sendLatestRuntimeProgress(progresses chan runtime.PullProgress, progress runtime.PullProgress) {
	select {
	case progresses <- progress:
	default:
		select {
		case <-progresses:
		default:
		}
		select {
		case progresses <- progress:
		default:
		}
	}
}

func drainRuntimeProgress(progresses chan runtime.PullProgress, state *runtimeDownloadState) {
	for {
		select {
		case progress := <-progresses:
			state.update(progress)
		default:
			return
		}
	}
}

// unusedOwnedRuntimes lists b70ctl-owned runtime images with zero references
// from installed packs. Runtimes without an ownership record — pre-existing,
// user-pulled, or unknown — never appear here no matter what they look like.
func (a *app) unusedOwnedRuntimes() ([]runtime.OwnershipRecord, error) {
	records, err := runtime.LoadOwnership(a.paths.DataRoot)
	if err != nil {
		return nil, err
	}
	packs, err := a.loadPacks()
	if err != nil {
		return nil, err
	}
	manifests := make([]*modelpack.Manifest, 0, len(packs))
	for index := range packs {
		manifests = append(manifests, packs[index].manifest)
	}
	referenced := runtime.ReferencedDigests(manifests)
	entries := make([]runtime.OwnershipRecord, 0, len(records))
	for _, record := range records {
		if _, used := referenced[record.Digest]; used {
			continue
		}
		entries = append(entries, record)
	}
	sort.Slice(entries, func(i, j int) bool { return entries[i].Digest < entries[j].Digest })
	return entries, nil
}

func (a *app) unusedRuntimesScreen() error {
	selected := 0
	for {
		if !a.sizeOK() {
			exit, err := a.tooSmall()
			if err != nil || exit {
				return err
			}
			continue
		}
		entries, err := a.unusedOwnedRuntimes()
		width, height := a.dimensions()
		contentWidth := subpageWidth(width) - 2
		lines := []string{}
		totalItems := len(entries) + 1
		if selected >= totalItems {
			selected = totalItems - 1
		}
		if err != nil {
			lines = append(lines, messageLine("● ERROR", contentWidth, red), messageLine(err.Error(), contentWidth, muted))
		} else if len(entries) == 0 {
			lines = append(lines, messageLine("No b70ctl-acquired runtime images are unused.", contentWidth, muted))
		} else {
			visible := max(1, (height-7)/3)
			start := 0
			if len(entries) > visible {
				if selected < len(entries) && selected >= visible {
					start = selected - visible + 1
				} else if selected >= len(entries) {
					start = len(entries) - visible
				}
			}
			end := min(len(entries), start+visible)
			for index := start; index < end; index++ {
				record := entries[index]
				state, reason := unusedRuntimeState(record)
				detail := "    " + statusText(state) + muted + "  " + shortDigest(record.Digest) + reset
				if reason != "" {
					detail += muted + "  " + reason + reset
				}
				lines = append(lines, focusRow(ownedRuntimeName(record), index == selected, false, contentWidth), detail, "")
			}
			lines = append(lines, listPosition(start, end, len(entries))...)
		}
		actionLines := []string{actionRow("Back", selected == len(entries), false, actionSecondary, contentWidth)}
		a.drawSubpagePanels("Unused Runtimes", []subpagePanel{{title: "B70CTL-ACQUIRED RUNTIMES", content: lines}, {title: "ACTIONS", content: actionLines}}, subpageFooter("↑↓ Navigate", "Enter Select", "Esc Back"), false)
		event, err := a.in.next()
		if err != nil {
			return err
		}
		switch event.key {
		case keyCtrlC:
			return io.EOF
		case keyEscape:
			return nil
		case keyUp:
			selected = previous(selected, totalItems)
		case keyDown:
			selected = next(selected, totalItems)
		case keyEnter:
			if selected >= len(entries) {
				return nil
			}
			if err := a.unusedRuntimeScreen(entries[selected]); err != nil {
				return err
			}
			if selected >= len(entries)-1 && selected > 0 {
				selected--
			}
		}
	}
}

func (a *app) unusedRuntimeScreen(record runtime.OwnershipRecord) error {
	selected := 0
	for {
		if !a.sizeOK() {
			exit, err := a.tooSmall()
			if err != nil || exit {
				return err
			}
			continue
		}
		state, reason := unusedRuntimeState(record)
		actions := []string{"Remove", "Back"}
		if selected >= len(actions) {
			selected = len(actions) - 1
		}
		contentWidth := subpageWidth(a.width()) - 2
		runtimeLines := []string{
			fieldRow("Runtime", ownedRuntimeName(record), contentWidth),
			fieldRow("Digest", shortDigest(record.Digest), contentWidth),
			fieldRow("Status", statusText(state), contentWidth),
			fieldRow("Acquired", record.Acquired.Format("2006-01-02"), contentWidth),
		}
		notice := "This runtime was pulled by b70ctl and no installed pack uses it. Controller never removes images it cannot prove it acquired."
		detailLines := []string{messageLine(notice, contentWidth, muted)}
		if reason != "" {
			detailLines = append(detailLines, messageLine(reason, contentWidth, amber))
		}
		if a.message != "" {
			runtimeLines = append(runtimeLines, "", messageLine(a.message, contentWidth, muted))
		}
		actionLines := []string{
			actionRow("Remove", selected == 0, false, actionDestructive, contentWidth),
			actionRow("Back", selected == 1, false, actionSecondary, contentWidth),
		}
		a.drawSubpagePanels("Unused Runtime", []subpagePanel{{title: "RUNTIME", content: runtimeLines}, {title: "DETAILS", content: detailLines}, {title: "ACTIONS", content: actionLines}}, subpageFooter("↑↓ Navigate", "Enter Select", "Esc Back"), false)
		a.message = ""
		event, err := a.in.next()
		if err != nil {
			return err
		}
		switch event.key {
		case keyCtrlC:
			return io.EOF
		case keyEscape:
			return nil
		case keyUp:
			selected = previous(selected, len(actions))
		case keyDown:
			selected = next(selected, len(actions))
		case keyEnter:
			switch actions[selected] {
			case "Back":
				return nil
			case "Remove":
				yes, err := a.confirm("Remove runtime image " + shortDigest(record.Digest) + "?")
				if err != nil {
					return err
				}
				if !yes {
					continue
				}
				a.message = a.removeUnusedRuntime(record)
			}
		}
	}
}

// removeUnusedRuntime rechecks every removal precondition immediately before
// removal: ownership for the exact digest, fresh installed-pack references,
// active runtime safety, and Docker's exact image identity.
func (a *app) removeUnusedRuntime(record runtime.OwnershipRecord) string {
	packs, err := a.loadPacks()
	if err != nil {
		return "Not removed — installed packs could not be read: " + err.Error()
	}
	manifests := make([]*modelpack.Manifest, 0, len(packs))
	for index := range packs {
		manifests = append(manifests, packs[index].manifest)
	}
	result := runtime.RemoveOwnedRuntime(a.paths.DataRoot, record.Digest, runtime.ReferencedDigests(manifests))
	switch result.Status {
	case runtime.ImageRemovalRemoved:
		return "Runtime image removed."
	case runtime.ImageRemovalMissing:
		return "Image already gone — ownership record cleared."
	default:
		return "Not removed — " + result.Reason + "."
	}
}

// unusedRuntimeState derives the row status for an owned runtime record and
// an optional human explanation when the image is missing or unresolvable.
func unusedRuntimeState(record runtime.OwnershipRecord) (string, string) {
	inspection := runtime.InspectRuntime(modelpack.Runtime{Image: record.Image, Digest: record.Digest})
	switch inspection.Status {
	case runtime.ImagePresent:
		return "UNUSED", ""
	case runtime.ImageMissing:
		return "MISSING", "image absent from Docker — removing clears the stale record"
	default:
		return "UNKNOWN", "Docker image state is unavailable: " + inspection.Err.Error()
	}
}

func ownedRuntimeName(record runtime.OwnershipRecord) string {
	if record.Image == "" {
		return shortDigest(record.Digest)
	}
	return runtimeImageName(modelpack.Runtime{Image: record.Image})
}

func shortDigest(digest string) string {
	if len(digest) <= 7+12 {
		return digest
	}
	return digest[:7+12] + "…"
}

func (a *app) uninstallPack(pack loadedPack) (bool, error) {
	selected := uninstallDefaultSelection
	choices := []string{"Delete everything", "Keep models", "Cancel"}
	for {
		if !a.sizeOK() {
			exit, err := a.tooSmall()
			if err != nil || exit {
				return false, err
			}
			continue
		}
		contentWidth := subpageWidth(a.width()) - 2
		lines := []string{messageLine(pack.installed.Name, contentWidth, muted), ""}
		for index, choice := range choices {
			style := actionDestructive
			if choice == "Cancel" {
				style = actionSecondary
			}
			lines = append(lines, actionRow(choice, index == selected, false, style, contentWidth))
		}
		if selected == 0 {
			lines = append(lines, "", messageLine("Removes the pack, unused pack runtime, and unused standalone model artifacts.", contentWidth, muted))
		} else if selected == 1 {
			lines = append(lines, "", messageLine("Removes the pack and unused pack runtime. Downloaded model files are kept.", contentWidth, muted))
		}
		a.drawSubpage("Uninstall Pack", "REMOVE PACK", lines, subpageFooter("↑↓ Navigate", "Enter Select", "Esc Cancel"), false)
		event, err := a.in.next()
		if err != nil {
			return false, err
		}
		switch event.key {
		case keyCtrlC:
			return false, io.EOF
		case keyEscape:
			return false, nil
		case keyUp:
			selected = previous(selected, len(choices))
		case keyDown:
			selected = next(selected, len(choices))
		case keyEnter:
			if selected == 2 {
				return false, nil
			}
			yes, err := a.confirmUninstall(pack.installed.ID, pack.installed.Version)
			if err != nil || !yes {
				return false, err
			}
			mode := uninstall.DeleteEverything
			if selected == 1 {
				mode = uninstall.KeepModels
			}
			contentWidth := subpageWidth(a.width()) - 2
			a.drawSubpage("Uninstall Pack", "REMOVING", []string{fieldRow("Pack", pack.installed.Name, contentWidth), "", messageLine("Removing selected artifacts…", contentWidth, red)}, "", false)
			result, err := uninstall.Run(a.paths.PacksRoot, a.paths.DataRoot, a.config.ModelDirectory, pack.installed.ID, pack.installed.Version, mode)
			if err != nil {
				if err := a.messageScreen("Uninstall Pack", err.Error()); err != nil {
					return false, err
				}
				return false, nil
			}
			if err := a.uninstallResult(result); err != nil {
				return false, err
			}
			return true, nil
		}
	}
}

func (a *app) confirmUninstall(packID, version string) (bool, error) {
	selected := 0
	for {
		if !a.sizeOK() {
			exit, err := a.tooSmall()
			if err != nil {
				return false, err
			}
			if exit {
				return false, nil
			}
			continue
		}
		contentWidth := subpageWidth(a.width()) - 2
		lines := []string{
			messageLine(fmt.Sprintf("Uninstall %s %s?", packID, version), contentWidth, muted),
			"",
			actionRow("No", selected == 0, false, actionSecondary, contentWidth),
			actionRow("Yes", selected == 1, false, actionDestructive, contentWidth),
		}
		a.drawSubpage("Confirm", "CONFIRM", lines, subpageFooter("↑↓ Navigate", "Enter Select", "Esc Cancel"), false)
		event, err := a.in.next()
		if err != nil {
			return false, err
		}
		switch event.key {
		case keyCtrlC:
			return false, io.EOF
		case keyEscape:
			return false, nil
		case keyUp, keyDown:
			selected = 1 - selected
		case keyEnter:
			return selected == 1, nil
		}
	}
}

func (a *app) uninstallResult(result uninstall.Result) error {
	for {
		if !a.sizeOK() {
			exit, err := a.tooSmall()
			if err != nil || exit {
				return err
			}
			continue
		}
		contentWidth := subpageWidth(a.width()) - 2
		lines := []string{messageLine("● PACK REMOVED", contentWidth, green), ""}
		for _, item := range result.Items {
			lines = append(lines, fieldRow("Artifact", item.Artifact, contentWidth), fieldRow("Outcome", item.Outcome, contentWidth), "")
		}
		a.drawSubpage("Uninstall Pack", "RESULT", lines, subpageFooter("Enter Continue", "Esc Back"), false)
		event, err := a.in.next()
		if err != nil {
			return err
		}
		if event.key == keyCtrlC {
			return io.EOF
		}
		if event.key == keyEscape || event.key == keyEnter {
			return nil
		}
	}
}

func (a *app) settingsScreen() error {
	selected := 0
	for {
		if !a.sizeOK() {
			exit, err := a.tooSmall()
			if err != nil || exit {
				return err
			}
			continue
		}
		configured, _, _ := hf.Configured()
		hfStatus := "Not configured"
		if configured {
			hfStatus = "Configured"
		}
		contentWidth := subpageWidth(a.width()) - 2
		lines := []string{
			choiceFieldRow("Model Directory", a.config.ModelDirectory, "", selected == 0, false, contentWidth),
			choiceFieldRow("Hugging Face", hfStatus, "", selected == 1, false, contentWidth),
			choiceFieldRow("Default Access", accessName(a.config.DefaultAccess), "← →", selected == 2, false, contentWidth),
			choiceFieldRow("Default Port", strconv.Itoa(a.config.DefaultPort), "", selected == 3, false, contentWidth),
			"",
			actionRow("Back", selected == 4, false, actionSecondary, contentWidth),
		}
		if a.message != "" {
			lines = append(lines, "", messageLine(a.message, contentWidth, muted))
		}
		a.drawSubpage("Settings", "CONTROLLER", lines, subpageFooter("↑↓ Navigate", "Enter Edit", "←→ Change", "Esc Back", "Changes save immediately"), false)
		a.message = ""
		event, err := a.in.next()
		if err != nil {
			return err
		}
		switch event.key {
		case keyCtrlC:
			return io.EOF
		case keyEscape:
			return nil
		case keyUp:
			selected = previous(selected, 5)
		case keyDown:
			selected = next(selected, 5)
		case keyLeft, keyRight:
			if selected == 2 {
				a.config.DefaultAccess = otherAccess(a.config.DefaultAccess)
				a.saveConfig()
			}
		case keyEnter:
			switch selected {
			case 0:
				path, accepted, err := a.editText("Model Directory:", a.config.ModelDirectory)
				if err != nil {
					return err
				}
				if accepted {
					path, err = absolutePath(path)
					if err != nil {
						a.message = err.Error()
					} else {
						a.config.ModelDirectory = path
						a.saveConfig()
					}
				}
			case 1:
				if err := a.hfAccessScreen(); err != nil {
					return err
				}
			case 2:
				a.config.DefaultAccess = otherAccess(a.config.DefaultAccess)
				a.saveConfig()
			case 3:
				text, accepted, err := a.editText("Default Port:", strconv.Itoa(a.config.DefaultPort))
				if err != nil {
					return err
				}
				if accepted {
					port, err := parsePort(text)
					if err != nil {
						a.message = err.Error()
					} else {
						a.config.DefaultPort = port
						a.saveConfig()
					}
				}
			case 4:
				return nil
			}
		}
	}
}

func (a *app) hfAccessScreen() error {
	selected := 0
	for {
		if !a.sizeOK() {
			exit, err := a.tooSmall()
			if err != nil || exit {
				return err
			}
			continue
		}
		configured, source, statusErr := hf.Configured()
		status := "Not configured"
		if configured {
			status = "Configured"
		}
		actions := []string{"Set Token", "Back"}
		if configured && source == hf.SourceFile {
			actions = []string{"Set Token", "Clear Token", "Back"}
		}
		if selected >= len(actions) {
			selected = len(actions) - 1
		}
		contentWidth := subpageWidth(a.width()) - 2
		credentialLines := []string{fieldRow("Status", statusText(status), contentWidth)}
		if configured {
			credentialLines = append(credentialLines, fieldRow("Source", source, contentWidth))
		} else {
			credentialLines = append(credentialLines, fieldRow("Source", "—", contentWidth))
		}
		actionLines := []string{}
		for index, action := range actions {
			style := actionSecondary
			if action == "Set Token" {
				style = actionPrimary
			} else if action == "Clear Token" {
				style = actionDestructive
			}
			actionLines = append(actionLines, actionRow(action, index == selected, false, style, contentWidth))
		}
		if statusErr != nil {
			credentialLines = append(credentialLines, "", messageLine(statusErr.Error(), contentWidth, red))
		} else if a.message != "" {
			credentialLines = append(credentialLines, "", messageLine(a.message, contentWidth, muted))
		}
		a.drawSubpagePanels("Hugging Face Access", []subpagePanel{{title: "CREDENTIAL", content: credentialLines}, {title: "ACTIONS", content: actionLines}}, subpageFooter("↑↓ Navigate", "Enter Select", "Esc Back"), false)
		a.message = ""
		event, err := a.in.next()
		if err != nil {
			return err
		}
		switch event.key {
		case keyCtrlC:
			return io.EOF
		case keyEscape:
			return nil
		case keyUp:
			selected = previous(selected, len(actions))
		case keyDown:
			selected = next(selected, len(actions))
		case keyEnter:
			switch actions[selected] {
			case "Set Token":
				if _, err := a.editSecret(); err != nil {
					return err
				}
			case "Clear Token":
				if err := hf.Clear(); err != nil {
					a.message = err.Error()
				} else {
					a.message = "Hugging Face token cleared."
				}
			case "Back":
				return nil
			}
		}
	}
}

func (a *app) choose(title string, options []option, selected int) (int, bool, error) {
	if len(options) == 0 {
		return 0, false, nil
	}
	for {
		if !a.sizeOK() {
			exit, err := a.tooSmall()
			if err != nil {
				return 0, false, err
			}
			if exit {
				return selected, false, nil
			}
			continue
		}
		contentWidth := subpageWidth(a.width()) - 2
		lines := []string{}
		start := 0
		visible := max(1, a.height()-6)
		if selected >= visible {
			start = selected - visible + 1
		}
		end := min(len(options), start+visible)
		for index := start; index < end; index++ {
			choice := options[index]
			lines = append(lines, focusRow(choice.label, index == selected, choice.disabled, contentWidth))
		}
		lines = append(lines, listPosition(start, end, len(options))...)
		a.drawSubpage("Select "+title, strings.ToUpper(title), lines, subpageFooter("↑↓ Navigate", "Enter Select", "Esc Cancel"), false)
		event, err := a.in.next()
		if err != nil {
			return 0, false, err
		}
		switch event.key {
		case keyCtrlC:
			return 0, false, io.EOF
		case keyEscape:
			return selected, false, nil
		case keyUp:
			selected = previous(selected, len(options))
		case keyDown:
			selected = next(selected, len(options))
		case keyEnter:
			if !options[selected].disabled {
				return selected, true, nil
			}
		}
	}
}

func (a *app) editText(prompt, initial string) (string, bool, error) {
	value := []rune(initial)
	cursor := len(value)
	for {
		if !a.sizeOK() {
			exit, err := a.tooSmall()
			if err != nil {
				return "", false, err
			}
			if exit {
				return initial, false, nil
			}
			continue
		}
		contentWidth := subpageWidth(a.width()) - 2
		label := strings.TrimSpace(strings.TrimSuffix(prompt, ":"))
		lines := []string{messageLine(strings.ToUpper(label), contentWidth, muted), "", editorValueLine(value, cursor, contentWidth, false)}
		a.drawSubpage(label, "EDIT VALUE", lines, subpageFooter("←→ Cursor", "Enter Save", "Esc Cancel"), false)
		event, err := a.in.next()
		if err != nil {
			return "", false, err
		}
		switch event.key {
		case keyCtrlC:
			return "", false, io.EOF
		case keyEscape:
			return initial, false, nil
		case keyEnter:
			return string(value), true, nil
		case keyLeft:
			if cursor > 0 {
				cursor--
			}
		case keyRight:
			if cursor < len(value) {
				cursor++
			}
		case keyBackspace:
			if cursor > 0 {
				copy(value[cursor-1:], value[cursor:])
				value[len(value)-1] = 0
				value = value[:len(value)-1]
				cursor--
			}
		case keyCharacter:
			value = append(value, 0)
			copy(value[cursor+1:], value[cursor:])
			value[cursor] = rune(event.char)
			cursor++
		}
	}
}

func (a *app) editSecret() (bool, error) {
	value := []rune{}
	cursor := 0
	for {
		if !a.sizeOK() {
			exit, err := a.tooSmall()
			if err != nil {
				return false, err
			}
			if exit {
				return false, nil
			}
			continue
		}
		contentWidth := subpageWidth(a.width()) - 2
		lines := []string{messageLine("TOKEN", contentWidth, muted), "", editorValueLine(value, cursor, contentWidth, true)}
		a.drawSubpage("Hugging Face Token", "CREDENTIAL", lines, subpageFooter("←→ Cursor", "Enter Save", "Esc Cancel"), false)
		event, err := a.in.next()
		if err != nil {
			return false, err
		}
		switch event.key {
		case keyCtrlC:
			return false, io.EOF
		case keyEscape:
			for index := range value {
				value[index] = 0
			}
			return false, nil
		case keyEnter:
			err := hf.Set(string(value))
			for index := range value {
				value[index] = 0
			}
			if err != nil {
				a.message = err.Error()
				return false, nil
			}
			a.message = "Hugging Face token saved."
			return true, nil
		case keyLeft:
			if cursor > 0 {
				cursor--
			}
		case keyRight:
			if cursor < len(value) {
				cursor++
			}
		case keyBackspace:
			if cursor > 0 {
				copy(value[cursor-1:], value[cursor:])
				value[len(value)-1] = 0
				value = value[:len(value)-1]
				cursor--
			}
		case keyCharacter:
			value = append(value, 0)
			copy(value[cursor+1:], value[cursor:])
			value[cursor] = rune(event.char)
			cursor++
		}
	}
}

func (a *app) confirm(question string) (bool, error) {
	selected := confirmDefaultSelection
	for {
		if !a.sizeOK() {
			exit, err := a.tooSmall()
			if err != nil || exit {
				return false, err
			}
			continue
		}
		contentWidth := subpageWidth(a.width()) - 2
		lines := []string{
			messageLine(question, contentWidth, muted),
			"",
			actionRow("No", selected == 0, false, actionSecondary, contentWidth),
			actionRow("Yes", selected == 1, false, actionDestructive, contentWidth),
		}
		a.drawSubpage("Confirm", "CONFIRM", lines, subpageFooter("↑↓ Navigate", "Enter Select", "Esc Cancel"), false)
		event, err := a.in.next()
		if err != nil {
			return false, err
		}
		switch event.key {
		case keyCtrlC:
			return false, io.EOF
		case keyEscape:
			return false, nil
		case keyUp, keyDown:
			selected = 1 - selected
		case keyEnter:
			return selected == 1, nil
		}
	}
}

func (a *app) messageScreen(title, message string) error {
	return a.noticeScreen(title, []string{message})
}

func (a *app) noticeScreen(title string, message []string) error {
	for {
		if !a.sizeOK() {
			exit, err := a.tooSmall()
			if err != nil || exit {
				return err
			}
			continue
		}
		contentWidth := subpageWidth(a.width()) - 2
		lines := []string{}
		for _, line := range message {
			lines = append(lines, messageLine(line, contentWidth, muted))
		}
		lines = append(lines, "", actionRow("Back", true, false, actionSecondary, contentWidth))
		panel := "NOTICE"
		if strings.Contains(strings.ToLower(title), "fail") || strings.Contains(strings.ToLower(title), "denied") || strings.Contains(strings.ToLower(title), "error") {
			panel = "ERROR"
		}
		a.drawSubpage(title, panel, lines, subpageFooter("Enter Continue", "Esc Back"), false)
		event, err := a.in.next()
		if err != nil {
			return err
		}
		if event.key == keyCtrlC {
			return io.EOF
		}
		if event.key == keyEscape || event.key == keyEnter {
			return nil
		}
	}
}

func (a *app) runtimeView() runtimeView {
	status, err := runtime.Status()
	view := runtimeView{status: status, err: err}
	if err != nil || status.PackID == "" {
		return view
	}
	packs, err := a.loadPacks()
	if err != nil {
		view.err = err
		return view
	}
	return resolveRuntimeMetadata(status, packs)
}

func resolveRuntimeMetadata(status runtime.StatusResult, packs []loadedPack) runtimeView {
	view := runtimeView{status: status}
	for packIndex := range packs {
		pack := &packs[packIndex]
		if pack.installed.ID != status.PackID || pack.installed.Version != status.PackVersion {
			continue
		}
		view.pack = pack
		for profileIndex := range pack.manifest.Profiles {
			profile := &pack.manifest.Profiles[profileIndex]
			if profile.ID != status.ProfileID {
				continue
			}
			view.profile = profile
			for modelIndex := range pack.manifest.Models {
				model := &pack.manifest.Models[modelIndex]
				if model.ID == profile.ModelID {
					view.model = model
					return view
				}
			}
			return view
		}
		return view
	}
	return view
}

func (a *app) loadPacks() ([]loadedPack, error) {
	installed, err := packstore.List(a.paths.PacksRoot)
	if err != nil {
		return nil, err
	}
	result := make([]loadedPack, 0, len(installed))
	for _, pack := range installed {
		path := filepath.Join(a.paths.PacksRoot, pack.ID, pack.Version)
		manifest, err := modelpack.Load(path)
		if err != nil {
			continue
		}
		result = append(result, loadedPack{installed: pack, manifest: manifest, path: path})
	}
	return result, nil
}

func (a *app) resetSelection(selection *runSelection, target targetChoice, availableCards int) {
	cards := cardOptions(target.profiles, target.model.ID)
	selection.cards = cards[0]
	for _, count := range cards {
		if count <= availableCards {
			selection.cards = count
			break
		}
	}
	resetContextMode(selection, target)
}

// restoreLastRun reapplies the persisted run selection in dependency order.
// Restoring the model first re-runs the normal default resolution for that
// target, then each remembered value is applied only if the current packs
// still offer it; the first invalid value keeps the normal default and
// dependent selectors continue resolving from it, so a stale preference can
// never produce a combination the pack does not declare.
func (a *app) restoreLastRun(selection *runSelection, targets []targetChoice, availableCards int) {
	preference := a.config.LastRun
	if preference == nil {
		return
	}
	for index := range targets {
		if targets[index].model.ID == preference.ModelID {
			if index != selection.target {
				selection.target = index
				a.resetSelection(selection, targets[index], availableCards)
			}
			break
		}
	}
	target := targets[selection.target]
	for _, count := range cardOptions(target.profiles, target.model.ID) {
		if count == preference.Cards {
			selection.cards = count
			resetContextMode(selection, target)
			break
		}
	}
	for _, context := range contextOptions(target.profiles, target.model.ID, selection.cards) {
		if context == preference.Context {
			selection.context = context
			selection.mode = modeOptions(target.pack.manifest, target.profiles, target.model.ID, selection.cards, selection.context)[0]
			break
		}
	}
	for _, mode := range modeOptions(target.pack.manifest, target.profiles, target.model.ID, selection.cards, selection.context) {
		if mode == preference.Mode {
			selection.mode = mode
			break
		}
	}
}

func (a *app) saveRunSelection(target targetChoice, selection runSelection) {
	a.config.LastRun = &config.RunSelection{
		ModelID: target.model.ID,
		Cards:   selection.cards,
		Context: selection.context,
		Mode:    selection.mode,
	}
	if err := config.Save(a.paths.ConfigFile, a.config); err != nil {
		a.message = err.Error()
	}
}

func (a *app) saveConfig() {
	if err := config.Save(a.paths.ConfigFile, a.config); err != nil {
		a.message = err.Error()
	} else {
		a.message = "Settings saved."
	}
}

func (a *app) draw(lines []string, cursor bool) {
	cursorCode := hideCursor
	if cursor {
		cursorCode = showCursor
	}
	fmt.Fprint(a.out, clear+cursorCode+strings.Join(lines, "\r\n")+reset)
}

func (a *app) dimensions() (int, int) {
	width, height, err := term.GetSize(int(a.out.Fd()))
	if err != nil {
		return 80, 24
	}
	return width, height
}

func (a *app) width() int {
	width, _ := a.dimensions()
	return width
}

func (a *app) height() int {
	_, height := a.dimensions()
	return height
}

func (a *app) sizeOK() bool {
	width, height := a.dimensions()
	return width >= minimumW && height >= minimumH
}

func (a *app) tooSmall() (bool, error) {
	a.draw([]string{"Terminal too small.", "", "Resize to at least 48 × 20, or press Esc to exit."}, false)
	event, err := a.in.next()
	if err != nil {
		return false, err
	}
	return event.key == keyEscape || event.key == keyCtrlC, nil
}

func runtimePane(view runtimeView) []string {
	if view.err != nil {
		return []string{"Status       Error", "Model        —"}
	}
	lines := []string{"Status       " + string(view.status.State)}
	if view.status.State == runtime.StateStopped || view.model == nil || view.profile == nil {
		return append(lines, "Model        —")
	}
	return append(lines,
		"Model        "+view.model.Name,
		fmt.Sprintf("Cards        %d", view.profile.Cards),
		"Context      "+formatContext(view.profile.Context),
		"Mode         "+modeName(view.pack.manifest, view.profile.Mode),
		"Access       "+accessFromBind(view.status.BindAddress),
		fmt.Sprintf("Port         %d", view.status.HostPort),
	)
}

func runtimeDetails(view runtimeView) []string {
	lines := []string{}
	if view.model != nil {
		lines = append(lines, "Model        "+view.model.Name)
	}
	lines = append(lines, "Status       "+string(view.status.State))
	if view.profile != nil {
		lines = append(lines,
			fmt.Sprintf("Cards        %d", view.profile.Cards),
			"Context      "+formatContext(view.profile.Context),
			"Mode         "+modeName(view.pack.manifest, view.profile.Mode),
		)
	}
	if view.status.BindAddress != "" {
		lines = append(lines, "Access       "+accessFromBind(view.status.BindAddress))
	}
	if view.status.HostPort != 0 {
		lines = append(lines, fmt.Sprintf("Port         %d", view.status.HostPort))
	}
	if view.status.ExitCode != nil {
		lines = append(lines, fmt.Sprintf("Exit Code    %d", *view.status.ExitCode))
	}
	return lines
}

func buildTargets(packs []loadedPack) []targetChoice {
	var result []targetChoice
	names := map[string]int{}
	for _, pack := range packs {
		for _, model := range pack.manifest.Models {
			if model.Kind != "target" {
				continue
			}
			var profiles []modelpack.Profile
			for _, profile := range pack.manifest.Profiles {
				if profile.ModelID == model.ID {
					profiles = append(profiles, profile)
				}
			}
			if len(profiles) > 0 {
				result = append(result, targetChoice{pack: pack, model: model, profiles: profiles, label: model.Name})
				names[model.Name]++
			}
		}
	}
	for index := range result {
		if names[result[index].model.Name] > 1 {
			result[index].label += fmt.Sprintf("  [%s %s]", result[index].pack.installed.ID, result[index].pack.installed.Version)
		}
	}
	return result
}

func evaluate(target targetChoice, profile modelpack.Profile, modelRoot string) evaluation {
	models, err := modelstore.Scan(modelRoot)
	if err != nil {
		return evaluation{err: err}
	}
	devices, err := hardware.DetectB70()
	if err != nil {
		return evaluation{models: models, err: err}
	}
	images := runtime.InspectRuntimes(target.pack.manifest.Runtimes)
	resolved := runtime.Resolve(target.pack.manifest, models, images, len(devices))
	result := evaluation{models: models, detected: len(devices)}
	for _, availability := range resolved {
		if availability.ProfileID == profile.ID {
			result.availability = availability
			break
		}
	}
	for _, image := range images {
		if image.Status == runtime.ImageError {
			result.err = image.Err
			break
		}
	}
	return result
}

func detectedCards() int {
	devices, err := hardware.DetectB70()
	if err != nil {
		return 0
	}
	return len(devices)
}

func resetContextMode(selection *runSelection, target targetChoice) {
	contexts := contextOptions(target.profiles, target.model.ID, selection.cards)
	selection.context = contexts[0]
	modes := modeOptions(target.pack.manifest, target.profiles, target.model.ID, selection.cards, selection.context)
	selection.mode = modes[0]
}

func uniqueRequirements(packs []loadedPack) []modelpack.Model {
	seen := map[string]struct{}{}
	var result []modelpack.Model
	for _, pack := range packs {
		for _, model := range pack.manifest.Models {
			key := model.Repo + "\x00" + model.Revision
			if _, exists := seen[key]; exists {
				continue
			}
			seen[key] = struct{}{}
			result = append(result, model)
		}
	}
	sort.Slice(result, func(i, j int) bool {
		if result[i].Repo == result[j].Repo {
			return result[i].Revision < result[j].Revision
		}
		return result[i].Repo < result[j].Repo
	})
	return result
}

func packsRequiring(packs []loadedPack, repo, revision string) []loadedPack {
	var matches []loadedPack
	for _, pack := range packs {
		for _, model := range pack.manifest.Models {
			if model.Repo == repo && model.Revision == revision {
				matches = append(matches, pack)
				break
			}
		}
	}
	return matches
}

// modelsEntry is one row of the Models screen inventory: a model required by
// an installed pack (model carries the pack's expected file inventory), or a
// Controller-managed local model no installed pack references (orphan).
type modelsEntry struct {
	model     modelpack.Model
	orphan    bool
	readiness modelstore.ReadinessResult
}

// modelsInventory merges the two Models screen sources: models required by
// installed packs first, then Controller-managed local models that no
// installed pack references. Identities are exact repo+revision; orphans are
// deduplicated by that identity and limited to managed local models
// modelstore.Remove can safely delete — external Hugging Face cache entries,
// download staging, and unrecognized directories never qualify.
func modelsInventory(packs []loadedPack, artifacts []modelstore.Artifact) []modelsEntry {
	var entries []modelsEntry
	for _, model := range uniqueRequirements(packs) {
		entries = append(entries, modelsEntry{model: model, readiness: modelstore.Assess(artifacts, model)})
	}
	seen := map[string]struct{}{}
	for _, artifact := range artifacts {
		if !artifact.Managed() || len(packsRequiring(packs, artifact.Repo, artifact.Revision)) > 0 {
			continue
		}
		key := artifact.Repo + "\x00" + artifact.Revision
		if _, duplicate := seen[key]; duplicate {
			continue
		}
		seen[key] = struct{}{}
		entries = append(entries, modelsEntry{
			model:     modelpack.Model{Repo: artifact.Repo, Revision: artifact.Revision},
			orphan:    true,
			readiness: modelstore.ReadinessResult{State: modelstore.Present, Artifact: artifact},
		})
	}
	return entries
}

// mountedModels returns the models a profile mounts, or nil when the profile
// cannot be resolved against its manifest.
func mountedModels(manifest *modelpack.Manifest, profile modelpack.Profile) []modelpack.Model {
	resolved, err := modelpack.Resolve(manifest, profile)
	if err != nil {
		return nil
	}
	byID := make(map[string]modelpack.Model, len(manifest.Models))
	for _, model := range manifest.Models {
		byID[model.ID] = model
	}
	var mounted []modelpack.Model
	for _, mount := range resolved.Mounts {
		if model, exists := byID[mount.ModelID]; exists {
			mounted = append(mounted, model)
		}
	}
	return mounted
}

func completenessSummary(check modelstore.CheckResult) []string {
	var lines []string
	if count := len(check.MissingFiles); count > 0 {
		lines = append(lines, fmt.Sprintf("  Missing files: %d", count))
	}
	if count := len(check.SizeMismatches); count > 0 {
		lines = append(lines, fmt.Sprintf("  Size mismatches: %d", count))
	}
	return lines
}

func affectedPaths(check modelstore.CheckResult, limit int) []string {
	paths := append([]string{}, check.MissingFiles...)
	paths = append(paths, check.SizeMismatches...)
	if len(paths) == 0 || limit <= 0 {
		return nil
	}
	lines := []string{"", "Affected files:"}
	for index, path := range paths {
		if index == limit {
			break
		}
		lines = append(lines, "- "+path)
	}
	if len(paths) > limit {
		lines = append(lines, fmt.Sprintf("- and %d more", len(paths)-limit))
	}
	return lines
}

func newestLogLines(logs string, count int) []string {
	lines := strings.Split(strings.TrimRight(logs, "\r\n"), "\n")
	if count < 0 {
		count = 0
	}
	if len(lines) > count {
		lines = lines[len(lines)-count:]
	}
	return lines
}

func sanitize(value string) string {
	return strings.Map(func(character rune) rune {
		if character == '\t' {
			return ' '
		}
		if character < 32 || character == 127 {
			return -1
		}
		return character
	}, strings.TrimSuffix(value, "\r"))
}

func menuLine(label string, selected, disabled bool) string {
	prefix := "  "
	if selected {
		prefix = "> "
	}
	if disabled {
		return prefix + "\x1b[2m" + label + reset
	}
	if selected {
		return blue + prefix + label + reset
	}
	return prefix + label
}

func accessName(value string) string {
	if value == config.AccessLAN {
		return "LAN"
	}
	return "Local"
}

func accessFromBind(value string) string {
	if value == "0.0.0.0" {
		return "LAN"
	}
	return "Local"
}

func otherAccess(value string) string {
	if value == config.AccessLAN {
		return config.AccessLocal
	}
	return config.AccessLAN
}

func sourceName(value string) string {
	if value == packstore.SourceLocal {
		return "Local"
	}
	if value == packstore.SourceRemote {
		return "Remote"
	}
	return value
}

func parsePort(value string) (int, error) {
	port, err := strconv.Atoi(value)
	if err != nil || port < 1024 || port > 65535 {
		return 0, fmt.Errorf("port must be between 1024 and 65535")
	}
	return port, nil
}

func absolutePath(value string) (string, error) {
	value = strings.TrimSpace(value)
	if value == "" {
		return "", fmt.Errorf("path is required")
	}
	if value == "~" || strings.HasPrefix(value, "~/") {
		home, err := os.UserHomeDir()
		if err != nil {
			return "", err
		}
		value = filepath.Join(home, strings.TrimPrefix(value, "~/"))
	}
	path, err := filepath.Abs(value)
	if err != nil {
		return "", fmt.Errorf("resolve path: %w", err)
	}
	return filepath.Clean(path), nil
}

func truncateRevision(value string) string {
	if len(value) <= 12 {
		return value
	}
	return value[:12] + "…"
}

func newModelDownloadState(repository hf.Repository, now time.Time) modelDownloadState {
	progress := hf.Progress{TotalFiles: len(repository.Files)}
	progress.TotalBytes, progress.TotalKnown = repository.TotalSize()
	if len(repository.Files) > 0 {
		progress.CurrentFile = repository.Files[0].Path
		progress.FileNumber = 1
	}
	return modelDownloadState{
		repository: repository.Repo,
		status:     "Downloading",
		progress:   progress,
		started:    now,
		lastSample: now,
	}
}

func (state *modelDownloadState) update(progress hf.Progress) {
	if !state.terminal {
		state.progress = progress
	}
}

func (state *modelDownloadState) refresh(now time.Time) {
	state.elapsed = now.Sub(state.started)
	if state.elapsed < 0 {
		state.elapsed = 0
	}
	if state.terminal {
		return
	}
	interval := now.Sub(state.lastSample)
	if interval > 0 {
		state.speed = float64(state.progress.BytesDownloaded-state.lastBytes) / interval.Seconds()
		state.lastSample = now
		state.lastBytes = state.progress.BytesDownloaded
	}
	state.activity = (state.activity + 1) % len(`|/-\`)
}

func (state *modelDownloadState) finish(status string, now time.Time) {
	state.refresh(now)
	state.status = status
	state.terminal = true
}

func sendLatestProgress(progresses chan hf.Progress, progress hf.Progress) {
	select {
	case progresses <- progress:
	default:
		select {
		case <-progresses:
		default:
		}
		select {
		case progresses <- progress:
		default:
		}
	}
}

func drainProgress(progresses chan hf.Progress, state *modelDownloadState) {
	for {
		select {
		case progress := <-progresses:
			state.update(progress)
		default:
			return
		}
	}
}

func formatDownloadProgress(progress hf.Progress) string {
	downloaded := formatBytes(progress.BytesDownloaded)
	if progress.TotalKnown {
		return downloaded + " / " + formatBytes(progress.TotalBytes)
	}
	return downloaded
}

func formatRate(bytesPerSecond float64) string {
	if bytesPerSecond < 0 {
		bytesPerSecond = 0
	}
	return formatBytes(int64(bytesPerSecond)) + "/s"
}

func formatElapsed(elapsed time.Duration) string {
	if elapsed < 0 {
		elapsed = 0
	}
	totalSeconds := int64(elapsed / time.Second)
	hours := totalSeconds / 3600
	minutes := totalSeconds % 3600 / 60
	seconds := totalSeconds % 60
	if hours > 0 {
		return fmt.Sprintf("%02d:%02d:%02d", hours, minutes, seconds)
	}
	return fmt.Sprintf("%02d:%02d", minutes, seconds)
}

func formatBytes(value int64) string {
	if value < 1000 {
		return fmt.Sprintf("%d B", value)
	}
	units := []string{"KB", "MB", "GB", "TB"}
	size := float64(value)
	unit := "B"
	for _, nextUnit := range units {
		size /= 1000
		unit = nextUnit
		if size < 1000 {
			break
		}
	}
	return fmt.Sprintf("%.1f %s", size, unit)
}

func truncate(value string, width int) string {
	characters := []rune(value)
	if width <= 0 {
		return ""
	}
	if len(characters) <= width {
		return value
	}
	if width == 1 {
		return "…"
	}
	return string(characters[:width-1]) + "…"
}

func previous(index, length int) int {
	if index <= 0 {
		return length - 1
	}
	return index - 1
}

func next(index, length int) int {
	if index >= length-1 {
		return 0
	}
	return index + 1
}

func max(left, right int) int {
	if left > right {
		return left
	}
	return right
}

func min(left, right int) int {
	if left < right {
		return left
	}
	return right
}
