package tui

import (
	"context"
	"reflect"
	"strings"
	"testing"
	"time"

	"b70ctl/internal/hf"
	"b70ctl/internal/modelpack"
	"b70ctl/internal/modelstore"
	"b70ctl/internal/runtime"
)

func TestModelDownloadStateRefreshAndTerminalStates(t *testing.T) {
	started := time.Date(2026, 8, 26, 12, 0, 0, 0, time.UTC)
	repository := hf.Repository{Repo: "example/model", Files: []hf.File{
		{Path: "model.bin", Size: 300, SizeKnown: true},
	}}
	state := newModelDownloadState(repository, started)
	state.update(hf.Progress{
		CurrentFile: "model.bin", FileNumber: 1, TotalFiles: 1,
		BytesDownloaded: 200, TotalBytes: 300, TotalKnown: true,
	})
	state.refresh(started.Add(2 * time.Second))
	if state.status != "Downloading" || state.elapsed != 2*time.Second || state.speed != 100 {
		t.Fatalf("refreshed state = %#v", state)
	}
	if got := formatElapsed(state.elapsed); got != "00:02" {
		t.Fatalf("formatElapsed() = %q", got)
	}
	if got := formatRate(state.speed); got != "100 B/s" {
		t.Fatalf("formatRate() = %q", got)
	}

	state.finish("Complete", started.Add(3*time.Second))
	state.update(hf.Progress{BytesDownloaded: 999})
	if !state.terminal || state.status != "Complete" || state.progress.BytesDownloaded != 200 {
		t.Fatalf("completed state = %#v", state)
	}

	failed := newModelDownloadState(repository, started)
	failed.finish("Failed", started.Add(time.Second))
	activity := failed.activity
	failed.refresh(started.Add(2 * time.Second))
	if !failed.terminal || failed.status != "Failed" || failed.activity != activity {
		t.Fatalf("failed state = %#v", failed)
	}
}

func TestRuntimeDownloadStateRemainsVisiblyActiveDuringLongPull(t *testing.T) {
	started := time.Date(2026, 8, 26, 12, 0, 0, 0, time.UTC)
	state := newRuntimeDownloadState(modelpack.Runtime{ID: "runtime", Image: "example/runtime:1"}, started)
	state.update(runtime.PullProgress{Status: "Pulling", Activity: "Downloading layers", Latest: "abc: Downloading"})
	firstFrame := state.frame
	state.refresh(started.Add(42 * time.Second))
	if state.elapsed != 42*time.Second || state.frame == firstFrame || state.activity != "Downloading layers" {
		t.Fatalf("runtime state = %#v", state)
	}
	secondFrame := state.frame
	state.refresh(started.Add(43 * time.Second))
	if state.frame == secondFrame || formatElapsed(state.elapsed) != "00:43" {
		t.Fatalf("refreshed runtime state = %#v", state)
	}
}

func TestInstalledPackRuntimeRetryUsesAcquisitionBackend(t *testing.T) {
	application, output := uninstallTestApp(t, nil)
	digest := "sha256:" + strings.Repeat("a", 64)
	declared := modelpack.Runtime{
		ID: "runtime", Image: "local/runtime:latest", Digest: digest,
		Registry: "ghcr.io/example/runtime@" + digest,
	}
	original := acquireRuntime
	called := false
	acquireRuntime = func(_ context.Context, dataRoot string, got modelpack.Runtime, notify func(runtime.PullProgress)) runtime.AcquisitionResult {
		called = true
		if dataRoot != application.paths.DataRoot {
			t.Fatalf("acquisition data root = %q", dataRoot)
		}
		if !reflect.DeepEqual(got, declared) {
			t.Fatalf("acquired runtime = %#v", got)
		}
		notify(runtime.PullProgress{Status: "Pulling", Activity: "Downloading layers", Latest: "abc: Download complete"})
		return runtime.AcquisitionResult{Outcome: runtime.AcquisitionDownloaded}
	}
	t.Cleanup(func() { acquireRuntime = original })

	result, err := application.downloadRuntime(declared)
	if err != nil || !called || result.Outcome != runtime.AcquisitionDownloaded {
		t.Fatalf("downloadRuntime() = %#v, %v; called = %v", result, err, called)
	}
	if got := readTestOutput(t, output); !strings.Contains(stripANSI(got), "B70 // INSTALL PACK") || !strings.Contains(got, "Download complete") {
		t.Fatalf("runtime progress output = %q", got)
	}
}

func TestPackRuntimeStatusDeduplicatesDigestAndOffersRegistryRetry(t *testing.T) {
	digest := "sha256:" + strings.Repeat("a", 64)
	manifest := &modelpack.Manifest{Runtimes: []modelpack.Runtime{
		{ID: "one", Image: "example/one:1", Digest: digest, Registry: "ghcr.io/example/one@" + digest},
		{ID: "two", Image: "example/two:1", Digest: digest, Registry: "ghcr.io/example/two@" + digest},
	}}
	calls := 0
	state, downloadable := packRuntimeStatusWithInspect(manifest, func(modelpack.Runtime) runtime.ImageResult {
		calls++
		return runtime.ImageResult{Status: runtime.ImageMissing}
	})
	if state != "Missing" || calls != 1 || len(downloadable) != 1 || downloadable[0].ID != "one" {
		t.Fatalf("pack runtime status = %q, %#v; calls = %d", state, downloadable, calls)
	}
}

func TestUnknownDownloadTotalIsNotInvented(t *testing.T) {
	progress := hf.Progress{BytesDownloaded: 1250000}
	if got := formatDownloadProgress(progress); got != "1.2 MB" {
		t.Fatalf("formatDownloadProgress() = %q", got)
	}
	progress.TotalBytes = 9000000
	if got := formatDownloadProgress(progress); got != "1.2 MB" {
		t.Fatalf("unknown formatDownloadProgress() = %q", got)
	}
	progress.TotalKnown = true
	if got := formatDownloadProgress(progress); got != "1.2 MB / 9.0 MB" {
		t.Fatalf("known formatDownloadProgress() = %q", got)
	}
}

func TestIncompleteModelFormattingIsConcise(t *testing.T) {
	check := modelstore.CheckResult{
		MissingFiles:   []string{"one", "two"},
		SizeMismatches: []string{"three", "four"},
	}
	if got := strings.Join(completenessSummary(check), "\n"); got != "  Missing files: 2\n  Size mismatches: 2" {
		t.Fatalf("completenessSummary() = %q", got)
	}
	if got := strings.Join(affectedPaths(check, 3), "\n"); got != "\nAffected files:\n- one\n- two\n- three\n- and 1 more" {
		t.Fatalf("affectedPaths() = %q", got)
	}
}
