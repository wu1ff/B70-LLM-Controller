package tui

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"b70ctl/internal/install"
	"b70ctl/internal/modelpack"
	"b70ctl/internal/modelstore"
	"b70ctl/internal/runtime"
	"b70ctl/internal/uninstall"
)

func inventoryEntries(t *testing.T, application *app) []modelsEntry {
	t.Helper()
	packs, err := application.loadPacks()
	if err != nil {
		t.Fatal(err)
	}
	artifacts, err := modelstore.Scan(application.config.ModelDirectory)
	if err != nil {
		t.Fatal(err)
	}
	return modelsInventory(packs, artifacts)
}

func findEntry(entries []modelsEntry, repo, revision string) (modelsEntry, bool) {
	for _, entry := range entries {
		if entry.model.Repo == repo && entry.model.Revision == revision {
			return entry, true
		}
	}
	return modelsEntry{}, false
}

func lastFrame(output string) string {
	frames := strings.Split(output, "\x1b[2J\x1b[H")
	return frames[len(frames)-1]
}

func sendTestBytes(t *testing.T, application *app, bytes ...byte) {
	t.Helper()
	for _, value := range bytes {
		select {
		case application.in.bytes <- value:
		case <-time.After(5 * time.Second):
			t.Fatal("models screen stopped consuming input")
		}
	}
}

func runModelsScreen(t *testing.T, application *app) {
	t.Helper()
	done := make(chan error, 1)
	go func() { done <- application.modelsScreen() }()
	t.Cleanup(func() {
		select {
		case err := <-done:
			if err != nil {
				t.Fatalf("modelsScreen() error = %v", err)
			}
		case <-time.After(5 * time.Second):
			t.Fatal("models screen did not exit")
		}
	})
}

func TestModelsInventoryMergesPackRequirementsAndOrphans(t *testing.T) {
	models := t.TempDir()
	packsRoot := t.TempDir()
	application, _ := modelActionTestApp(t, nil, models, packsRoot)
	present := tuiTestModel()
	missing := tuiTestModelAt("example/missing", "1111111111111111111111111111111111111111")
	installTuiPack(t, packsRoot, "pack-a", "Pack A", present)
	installTuiPack(t, packsRoot, "pack-b", "Pack B", missing)
	writePresentModelAt(t, models, modelstore.DestinationName(present.Repo), present.Repo, present.Revision, 6)
	writePresentModelAt(t, models, "orphan__one", "orphan/one", "2222222222222222222222222222222222222222", 6)

	entries := inventoryEntries(t, application)
	if entry, ok := findEntry(entries, present.Repo, present.Revision); !ok || entry.orphan || entry.readiness.State != modelstore.Present {
		t.Fatalf("pack-required present model not a normal Present row: %#v, %v", entry, ok)
	}
	if entry, ok := findEntry(entries, missing.Repo, missing.Revision); !ok || entry.orphan || entry.readiness.State != modelstore.Missing {
		t.Fatalf("pack-required missing model not kept as a Missing row: %#v, %v", entry, ok)
	}
	if entry, ok := findEntry(entries, "orphan/one", "2222222222222222222222222222222222222222"); !ok || !entry.orphan || entry.readiness.State != modelstore.Present {
		t.Fatalf("managed unreferenced model not an orphan Present row: %#v, %v", entry, ok)
	}
	if len(entries) != 3 {
		t.Fatalf("inventory = %#v", entries)
	}
}

func TestModelsInventoryOrphanIdentityIsExactRepoAndRevision(t *testing.T) {
	models := t.TempDir()
	packsRoot := t.TempDir()
	application, _ := modelActionTestApp(t, nil, models, packsRoot)
	required := tuiTestModel()
	installTuiPack(t, packsRoot, "pack-a", "Pack A", required)
	// Same repository at another revision, and another repository at the
	// required revision: both remain orphans because identity is exact.
	writePresentModelAt(t, models, modelstore.DestinationName(required.Repo), required.Repo, "9999999999999999999999999999999999999999", 6)
	writePresentModelAt(t, models, "neighbor__model", "neighbor/model", required.Revision, 6)

	entries := inventoryEntries(t, application)
	if entry, ok := findEntry(entries, required.Repo, required.Revision); !ok || entry.orphan {
		t.Fatalf("exact requirement not a normal row: %#v, %v", entry, ok)
	}
	if entry, ok := findEntry(entries, required.Repo, "9999999999999999999999999999999999999999"); !ok || !entry.orphan {
		t.Fatalf("other revision of the same repo is not an orphan: %#v, %v", entry, ok)
	}
	if entry, ok := findEntry(entries, "neighbor/model", required.Revision); !ok || !entry.orphan {
		t.Fatalf("other repo at the required revision is not an orphan: %#v, %v", entry, ok)
	}
}

func TestModelsInventorySharedModelIsNotOrphan(t *testing.T) {
	models := t.TempDir()
	packsRoot := t.TempDir()
	application, _ := modelActionTestApp(t, nil, models, packsRoot)
	model := tuiTestModel()
	installTuiPack(t, packsRoot, "pack-a", "Pack A", model)
	installTuiPack(t, packsRoot, "pack-b", "Pack B", tuiTestModelAt(model.Repo, model.Revision))
	writePresentModelAt(t, models, modelstore.DestinationName(model.Repo), model.Repo, model.Revision, 6)

	entries := inventoryEntries(t, application)
	if len(entries) != 1 {
		t.Fatalf("shared model produced %d rows: %#v", len(entries), entries)
	}
	if entries[0].orphan {
		t.Fatalf("model referenced by two packs classified as orphan: %#v", entries[0])
	}
}

func TestModelsInventoryHidesStagingAndUnmanagedEntries(t *testing.T) {
	models := t.TempDir()
	// Persistent download staging, never a model.
	staged := filepath.Join(models, ".b70ctl-staging", modelstore.DestinationName("staged/model")+"-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
	writeTestModelMarker(t, staged, "staged/model", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
	writeTestModelFile(t, filepath.Join(staged, "model.bin"), "weight")
	// Legacy/unrecognized directories.
	writeTestModelFile(t, filepath.Join(models, "random", "config.json"), `{}`)
	writeTestModelFile(t, filepath.Join(models, "loose.bin"), "junk")
	// Unrelated external Hugging Face cache content.
	snapshot := filepath.Join(models, "hub", "models--external--model", "snapshots", "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")
	writeTestModelFile(t, filepath.Join(snapshot, "model.bin"), "weight")

	artifacts, err := modelstore.Scan(models)
	if err != nil {
		t.Fatal(err)
	}
	if len(artifacts) == 0 {
		t.Fatal("Scan() discovered nothing to classify")
	}
	entries := modelsInventory(nil, artifacts)
	if len(entries) != 0 {
		t.Fatalf("staging, unrecognized, and external cache entries became rows: %#v", entries)
	}
}

func TestModelsScreenRendersOrphanIndicator(t *testing.T) {
	models := t.TempDir()
	packsRoot := t.TempDir()
	application, output := modelActionTestApp(t, []byte{27}, models, packsRoot)
	model := tuiTestModel()
	installTuiPack(t, packsRoot, "pack-a", "Pack A", model)
	writePresentModelAt(t, models, modelstore.DestinationName(model.Repo), model.Repo, model.Revision, 6)
	writePresentModelAt(t, models, "orphan__one", "orphan/one", "2222222222222222222222222222222222222222", 6)

	if err := application.modelsScreen(); err != nil {
		t.Fatal(err)
	}
	got := stripANSI(lastFrame(readTestOutput(t, output)))
	if !strings.Contains(got, "example/model") || !strings.Contains(got, "orphan/one") {
		t.Fatalf("models screen is missing rows:\n%s", got)
	}
	orphanLine := lineContaining(got, "orphan/one")
	next := linesAfter(got, orphanLine)
	if !strings.Contains(next, "ORPHANED") {
		t.Fatalf("orphan row is not labeled Orphaned:\n%s", got)
	}
	requiredLine := lineContaining(got, "example/model")
	next = linesAfter(got, requiredLine)
	if strings.Contains(next, "ORPHANED") {
		t.Fatalf("pack-required row is mislabeled Orphaned:\n%s", got)
	}
}

func TestOrphanDetailIdentifiesStatusAndOffersDeleteOnly(t *testing.T) {
	models := t.TempDir()
	model := tuiTestModel()
	writePresentModelAt(t, models, modelstore.DestinationName(model.Repo), model.Repo, model.Revision, 6)
	// Simulate externally damaged contents: without a pack there is no
	// expected inventory to compare against, so the artifact is Present.
	os.Remove(filepath.Join(models, modelstore.DestinationName(model.Repo), "model.bin"))
	application, output := modelActionTestApp(t, []byte{27}, models, t.TempDir())
	orphanModel := modelpack.Model{Repo: model.Repo, Revision: model.Revision}
	artifacts, err := modelstore.Scan(models)
	if err != nil {
		t.Fatal(err)
	}
	readiness := modelstore.ReadinessResult{State: modelstore.Present, Artifact: artifacts[0]}

	if err := application.modelScreen(orphanModel, readiness); err != nil {
		t.Fatal(err)
	}
	got := stripANSI(readTestOutput(t, output))
	for _, want := range []string{"ORPHANED", "DELETE", "No installed pack requires this model"} {
		if !strings.Contains(got, want) {
			t.Fatalf("orphan detail missing %q:\n%s", want, got)
		}
	}
	if strings.Contains(got, "REPLACE") || strings.Contains(got, "DOWNLOAD") {
		t.Fatalf("orphan detail offers a repair action:\n%s", got)
	}
}

func TestOrphanDeleteFlowRemovesModelAndDropsRow(t *testing.T) {
	models := t.TempDir()
	application, output := modelActionTestApp(t, []byte{'\n', '\n', 27, '[', 'B', '\n', 27}, models, t.TempDir())
	writePresentModelAt(t, models, modelstore.DestinationName("example/model"), "example/model", tuiTestRevision, 6)
	writePresentModelAt(t, models, "other__model", "example/other", strings.Repeat("8", 40), 6)

	if err := application.modelsScreen(); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(models, "example__model")); !os.IsNotExist(err) {
		t.Fatalf("orphan was not deleted: %v", err)
	}
	if _, err := os.Stat(filepath.Join(models, "other__model")); err != nil {
		t.Fatalf("unrelated orphan was removed: %v", err)
	}
	frame := stripANSI(lastFrame(readTestOutput(t, output)))
	if !strings.Contains(frame, "Model deleted.") {
		t.Fatalf("orphan delete did not report success:\n%s", frame)
	}
	if strings.Contains(frame, "example/model") || strings.Contains(frame, "MISSING") {
		t.Fatalf("deleted orphan kept a row instead of disappearing:\n%s", frame)
	}
	if !strings.Contains(frame, "example/other") {
		t.Fatalf("surviving orphan lost its row:\n%s", frame)
	}
}

func TestPackRequiredDeleteKeepsMissingRow(t *testing.T) {
	models := t.TempDir()
	packsRoot := t.TempDir()
	application, output := modelActionTestApp(t, backOutAfterDetailDeleteEvents, models, packsRoot)
	model := tuiTestModel()
	installTuiPack(t, packsRoot, "pack-a", "Pack A", model)
	writePresentModelAt(t, models, modelstore.DestinationName(model.Repo), model.Repo, model.Revision, 6)

	if err := application.modelsScreen(); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(models, "example__model")); !os.IsNotExist(err) {
		t.Fatalf("model was not deleted: %v", err)
	}
	frame := stripANSI(lastFrame(readTestOutput(t, output)))
	if !strings.Contains(frame, "example/model") || !strings.Contains(frame, "MISSING") {
		t.Fatalf("pack-required model row did not remain as Missing:\n%s", frame)
	}
}

func TestOrphanDeleteRefusedWhileRuntimeActive(t *testing.T) {
	models := t.TempDir()
	application, output := modelActionTestApp(t, backOutAfterDetailRefusalEvents, models, t.TempDir())
	writePresentModelAt(t, models, modelstore.DestinationName("example/model"), "example/model", tuiTestRevision, 6)
	stubRuntimeStatus(t, statusResultRunning("running-pack", "1.0.0"), nil)

	if err := application.modelsScreen(); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(models, "example__model")); err != nil {
		t.Fatalf("orphan mounted by an active runtime was deleted: %v", err)
	}
	got := stripANSI(readTestOutput(t, output))
	if !strings.Contains(got, "not deleted") {
		t.Fatalf("delete of an in-use orphan was not refused:\n%s", got)
	}
	frame := stripANSI(lastFrame(readTestOutput(t, output)))
	if !strings.Contains(frame, "example/model") || !strings.Contains(frame, "ORPHANED") {
		t.Fatalf("refused delete lost the orphan row:\n%s", frame)
	}
}

func TestOrphanMultipleCopiesPreserveRemovalRefusal(t *testing.T) {
	models := t.TempDir()
	application, output := modelActionTestApp(t, backOutAfterDetailDeleteEvents, models, t.TempDir())
	writePresentModelAt(t, models, "first", "example/model", tuiTestRevision, 6)
	writePresentModelAt(t, models, "second", "example/model", tuiTestRevision, 6)

	entries := inventoryEntries(t, application)
	if len(entries) != 1 || !entries[0].orphan {
		t.Fatalf("ambiguous copies produced %d rows: %#v", len(entries), entries)
	}
	if err := application.modelsScreen(); err != nil {
		t.Fatal(err)
	}
	for _, copy := range []string{"first", "second"} {
		if _, err := os.Stat(filepath.Join(models, copy)); err != nil {
			t.Fatalf("copy %q was removed: %v", copy, err)
		}
	}
	got := stripANSI(readTestOutput(t, output))
	if !strings.Contains(got, "Not deleted — multiple local copies found.") {
		t.Fatalf("multiple-copy refusal did not surface:\n%s", got)
	}
	frame := stripANSI(lastFrame(readTestOutput(t, output)))
	if !strings.Contains(frame, "example/model") || !strings.Contains(frame, "ORPHANED") {
		t.Fatalf("refused delete lost the ambiguous orphan row:\n%s", frame)
	}
}

func TestUninstallKeepModelsProducesOrphan(t *testing.T) {
	models := t.TempDir()
	packsRoot := t.TempDir()
	application, _ := modelActionTestApp(t, nil, models, packsRoot)
	model := tuiTestModel()
	installTuiPack(t, packsRoot, "pack-a", "Pack A", model)
	writePresentModelAt(t, models, modelstore.DestinationName(model.Repo), model.Repo, model.Revision, 6)

	if entry, ok := findEntry(inventoryEntries(t, application), model.Repo, model.Revision); !ok || entry.orphan {
		t.Fatalf("pack-referenced model started out orphaned: %#v, %v", entry, ok)
	}
	stubDockerForUninstall(t)
	result, err := uninstall.Run(packsRoot, t.TempDir(), models, "pack-a", "1.0.0", uninstall.KeepModels)
	if err != nil {
		t.Fatalf("uninstall.Run() error = %v", err)
	}
	if !result.PackRemoved {
		t.Fatalf("pack was not removed: %#v", result)
	}
	if _, err := os.Stat(filepath.Join(models, "example__model")); err != nil {
		t.Fatalf("Keep models uninstalled removed the model: %v", err)
	}
	if entry, ok := findEntry(inventoryEntries(t, application), model.Repo, model.Revision); !ok || !entry.orphan || entry.readiness.State != modelstore.Present {
		t.Fatalf("retained model is not shown as an orphan Present row: %#v, %v", entry, ok)
	}
}

func TestInstallingPackReclaimsOrphanWithoutRedownload(t *testing.T) {
	models := t.TempDir()
	packsRoot := t.TempDir()
	application, _ := modelActionTestApp(t, nil, models, packsRoot)
	model := tuiTestModel()
	writePresentModelAt(t, models, modelstore.DestinationName(model.Repo), model.Repo, model.Revision, 6)

	if entry, ok := findEntry(inventoryEntries(t, application), model.Repo, model.Revision); !ok || !entry.orphan {
		t.Fatalf("unreferenced model did not start out orphaned: %#v, %v", entry, ok)
	}
	installTuiPack(t, packsRoot, "pack-a", "Pack A", model)

	entries := inventoryEntries(t, application)
	if len(entries) != 1 {
		t.Fatalf("reclaimed orphan produced %d rows: %#v", len(entries), entries)
	}
	if entries[0].orphan || entries[0].readiness.State != modelstore.Present {
		t.Fatalf("installed pack did not reclaim the orphan row: %#v", entries[0])
	}
	packs, err := application.loadPacks()
	if err != nil {
		t.Fatal(err)
	}
	inventory, err := modelstore.Scan(models)
	if err != nil {
		t.Fatal(err)
	}
	plan, err := install.BuildPlan(packs[0].manifest, []string{"target"}, inventory)
	if err != nil {
		t.Fatal(err)
	}
	if counts := plan.Counts(); counts.Existing != 1 || counts.Downloads != 0 {
		t.Fatalf("exact complete artifact is not reused: %#v", counts)
	}
}

func TestRescanDiscoversNewlyCreatedOrphan(t *testing.T) {
	models := t.TempDir()
	packsRoot := t.TempDir()
	application, output := modelActionTestApp(t, nil, models, packsRoot)
	installTuiPack(t, packsRoot, "pack-a", "Pack A", tuiTestModel())

	runModelsScreen(t, application)
	writePresentModelAt(t, models, "orphan__one", "orphan/one", "2222222222222222222222222222222222222222", 6)
	// Two rows now (requirement + orphan): Rescan sits at index 2.
	sendTestBytes(t, application, 27, '[', 'B', 27, '[', 'B', '\n', 27)

	got := stripANSI(readTestOutput(t, output))
	if !strings.Contains(lastFrame(got), "orphan/one") || !strings.Contains(lastFrame(got), "ORPHANED") {
		t.Fatalf("rescan did not surface the new orphan:\n%s", lastFrame(got))
	}
}

func TestRescanDropsRemovedOrphan(t *testing.T) {
	models := t.TempDir()
	packsRoot := t.TempDir()
	application, output := modelActionTestApp(t, nil, models, packsRoot)
	installTuiPack(t, packsRoot, "pack-a", "Pack A", tuiTestModel())
	orphan := filepath.Join(models, "orphan__one")
	writePresentModelAt(t, models, "orphan__one", "orphan/one", "2222222222222222222222222222222222222222", 6)

	runModelsScreen(t, application)
	if err := os.RemoveAll(orphan); err != nil {
		t.Fatal(err)
	}
	// Only the requirement row remains: Rescan sits at index 1.
	sendTestBytes(t, application, 27, '[', 'B', '\n', 27)

	frame := stripANSI(lastFrame(readTestOutput(t, output)))
	if strings.Contains(frame, "orphan/one") {
		t.Fatalf("rescan kept a manually removed orphan:\n%s", frame)
	}
	if !strings.Contains(frame, "example/model") {
		t.Fatalf("rescan lost the pack-required row:\n%s", frame)
	}
}

// backOutAfterDetailDeleteEvents opens the first row's detail, activates
// Delete, confirms the destructive prompt, leaves the detail through its Back
// action, then leaves the Models list through its Back action. Back-to-back
// Escape bytes are avoided on purpose: the input parser reads Esc plus any
// immediately following byte as one potential escape sequence.
var backOutAfterDetailDeleteEvents = []byte{
	'\n', '\n', 27, '[', 'B', '\n', // open detail, Delete, confirm Yes
	27, '[', 'B', '\n', // detail Back
	27, '[', 'B', 27, '[', 'B', '\n', // list Rescan, Back
}

// backOutAfterDetailRefusalEvents drives a Delete that is refused before the
// confirmation appears, then leaves both screens through their Back actions.
var backOutAfterDetailRefusalEvents = []byte{
	'\n', '\n', // open detail, Delete (refused with a message)
	27, '[', 'B', '\n', // detail Back
	27, '[', 'B', 27, '[', 'B', '\n', // list Rescan, Back
}

func lineContaining(text, fragment string) string {
	for _, line := range strings.Split(text, "\n") {
		if strings.Contains(line, fragment) {
			return line
		}
	}
	return ""
}

func linesAfter(text, line string) string {
	lines := strings.Split(text, "\n")
	for index, candidate := range lines {
		if candidate == line && index+1 < len(lines) {
			return lines[index+1]
		}
	}
	return ""
}

func statusResultRunning(packID, version string) runtime.StatusResult {
	return runtime.StatusResult{State: runtime.StateRunning, PackID: packID, PackVersion: version, ProfileID: "profile"}
}

func stubDockerForUninstall(t *testing.T) {
	t.Helper()
	bin := t.TempDir()
	script := `#!/bin/sh
if [ "$1" = "inspect" ]; then
  echo "Error: No such object: b70ctl-runtime" >&2
  exit 1
fi
if [ "$1" = "image" ] && [ "$2" = "inspect" ]; then
  echo '"sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"'
  exit 0
fi
if [ "$1" = "image" ] && [ "$2" = "rm" ]; then
  echo "Untagged: $3"
  exit 0
fi
exit 2
`
	path := filepath.Join(bin, "docker")
	if err := os.WriteFile(path, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", bin+string(os.PathListSeparator)+os.Getenv("PATH"))
}
