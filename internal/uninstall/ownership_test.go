package uninstall

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"b70ctl/internal/packstore"
	"b70ctl/internal/runtime"
)

func TestUninstallRetainsRuntimeNotAcquiredByController(t *testing.T) {
	store := t.TempDir()
	data := t.TempDir()
	models := t.TempDir()
	digest := "sha256:" + strings.Repeat("1", 64)
	source := t.TempDir()
	manifest := uninstallManifest("legacy-pack", "Legacy Pack", "example/model", "revision-1", digest)
	writePack(t, source, manifest)
	if _, err := packstore.Import(source, store, packstore.SourceLocal); err != nil {
		t.Fatal(err)
	}
	removals := installLoggingDockerStub(t, digest, false, false)

	result, err := Run(store, data, models, manifest.ID, manifest.Version, DeleteEverything)
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if !result.PackRemoved {
		t.Fatalf("pack was not removed: %#v", result)
	}
	if got := result.Items[len(result.Items)-1].Outcome; got != "Retained — it was not acquired by b70ctl" {
		t.Fatalf("runtime outcome = %q", got)
	}
	if removals("image", "rm") != 0 {
		t.Fatal("unowned runtime image removal was attempted")
	}
	if records, err := runtime.LoadOwnership(data); err != nil || len(records) != 0 {
		t.Fatalf("ownership metadata created for pre-existing image: %#v, %v", records, err)
	}
}

func TestUninstallRetainsRuntimeWhenOwnershipStoreUnreadable(t *testing.T) {
	store := t.TempDir()
	data := t.TempDir()
	models := t.TempDir()
	digest := "sha256:" + strings.Repeat("2", 64)
	source := t.TempDir()
	manifest := uninstallManifest("corrupt-store-pack", "Corrupt Store Pack", "example/model", "revision-1", digest)
	writePack(t, source, manifest)
	if _, err := packstore.Import(source, store, packstore.SourceLocal); err != nil {
		t.Fatal(err)
	}
	installDockerStub(t, digest, false, false)
	if err := os.WriteFile(filepath.Join(data, "runtime-ownership.json"), []byte("{not json"), 0o644); err != nil {
		t.Fatal(err)
	}

	result, err := Run(store, data, models, manifest.ID, manifest.Version, DeleteEverything)
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if !result.PackRemoved {
		t.Fatalf("pack was not removed: %#v", result)
	}
	if got := result.Items[len(result.Items)-1].Outcome; got != "Retained — ownership could not be verified" {
		t.Fatalf("runtime outcome = %q", got)
	}
}

func TestSharedOwnedRuntimeSurvivesFirstUninstallAndLeavesWithLast(t *testing.T) {
	store := t.TempDir()
	data := t.TempDir()
	models := t.TempDir()
	digest := "sha256:" + strings.Repeat("3", 64)
	for _, id := range []string{"first-shared", "second-shared"} {
		source := t.TempDir()
		manifest := uninstallManifest(id, id, "example/shared", "revision-1", digest)
		writePack(t, source, manifest)
		if _, err := packstore.Import(source, store, packstore.SourceLocal); err != nil {
			t.Fatal(err)
		}
	}
	installDockerStub(t, digest, false, false)
	mustRecordOwnership(t, data, digest, "example/runtime:1")

	first, err := Run(store, data, models, "first-shared", "1.0.0", DeleteEverything)
	if err != nil {
		t.Fatalf("first Run() error = %v", err)
	}
	if got := first.Items[len(first.Items)-1].Outcome; got != "Retained — used by another installed pack" {
		t.Fatalf("first runtime outcome = %q", got)
	}
	if records, _ := runtime.LoadOwnership(data); len(records) != 1 {
		t.Fatalf("ownership record dropped while shared: %#v", records)
	}

	last, err := Run(store, data, models, "second-shared", "1.0.0", DeleteEverything)
	if err != nil {
		t.Fatalf("last Run() error = %v", err)
	}
	if got := last.Items[len(last.Items)-1].Outcome; got != "Removed" {
		t.Fatalf("last runtime outcome = %q", got)
	}
	if records, _ := runtime.LoadOwnership(data); len(records) != 0 {
		t.Fatalf("ownership record survived the removed image: %#v", records)
	}
}

// FLOW A — pre-existing image: the exact required runtime is already present,
// b70ctl reuses it, and uninstalling the last pack keeps both the image and
// the absent ownership metadata absent.
func TestFlowPreExistingRuntimeIsReusedNeverOwnedNeverDeleted(t *testing.T) {
	store := t.TempDir()
	data := t.TempDir()
	models := t.TempDir()
	source := t.TempDir()
	digest := "sha256:" + strings.Repeat("4", 64)
	manifest := uninstallManifest("preexisting-pack", "Pre-existing Pack", "example/model", "revision-1", digest)
	writePack(t, source, manifest)
	removals := installLoggingDockerStub(t, digest, false, false)
	if _, err := packstore.Import(source, store, packstore.SourceLocal); err != nil {
		t.Fatal(err)
	}

	acquisition := runtime.AcquireOwned(context.Background(), data, manifest.Runtimes[0], nil)
	if acquisition.Outcome != runtime.AcquisitionReused {
		t.Fatalf("pre-existing runtime was not reused: %#v", acquisition)
	}
	if records, err := runtime.LoadOwnership(data); err != nil || len(records) != 0 {
		t.Fatalf("reuse created ownership metadata: %#v, %v", records, err)
	}

	result, err := Run(store, data, models, manifest.ID, manifest.Version, DeleteEverything)
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if got := result.Items[len(result.Items)-1].Outcome; got != "Retained — it was not acquired by b70ctl" {
		t.Fatalf("runtime outcome = %q", got)
	}
	if removals("image", "rm") != 0 {
		t.Fatal("pre-existing runtime image removal was attempted")
	}
}

// FLOW B — b70ctl-acquired image: the runtime is absent, b70ctl pulls and
// verifies it, ownership is recorded, and the final uninstall removing the
// last reference removes the image and its record.
func TestFlowControllerAcquiredRuntimeIsOwnedAndLeavesWithLastPack(t *testing.T) {
	store := t.TempDir()
	data := t.TempDir()
	models := t.TempDir()
	source := t.TempDir()
	digest := "sha256:" + strings.Repeat("6", 64)
	manifest := uninstallManifest("acquired-pack", "Acquired Pack", "example/model", "revision-1", digest)
	manifest.Runtimes[0].Registry = "ghcr.io/example/runtime@" + digest
	writePack(t, source, manifest)
	removals := installPullingDockerStub(t, digest)
	if _, err := packstore.Import(source, store, packstore.SourceLocal); err != nil {
		t.Fatal(err)
	}

	acquisition := runtime.AcquireOwned(context.Background(), data, manifest.Runtimes[0], nil)
	if acquisition.Outcome != runtime.AcquisitionDownloaded {
		t.Fatalf("absent runtime was not pulled: %#v", acquisition)
	}
	records, err := runtime.LoadOwnership(data)
	if err != nil {
		t.Fatal(err)
	}
	if _, owned := records[digest]; !owned {
		t.Fatalf("pull did not record ownership: %#v", records)
	}

	result, err := Run(store, data, models, manifest.ID, manifest.Version, DeleteEverything)
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if got := result.Items[len(result.Items)-1].Outcome; got != "Removed" {
		t.Fatalf("runtime outcome = %q", got)
	}
	if removals("image", "rm") != 1 {
		t.Fatalf("owned runtime removals = %d", removals("image", "rm"))
	}
	if records, _ := runtime.LoadOwnership(data); len(records) != 0 {
		t.Fatalf("ownership record survived the removed image: %#v", records)
	}
}

// FLOW B2 — digest-pulled image: Docker never tags an image pulled by its
// immutable digest, so the declared tag stays absent while the digest
// reference resolves. Uninstalling the last pack must still remove the owned
// image (by digest reference), not misreport it as missing and leak it.
func TestFlowDigestPulledRuntimeRemovesByDigestReference(t *testing.T) {
	store := t.TempDir()
	data := t.TempDir()
	models := t.TempDir()
	source := t.TempDir()
	digest := "sha256:" + strings.Repeat("7", 64)
	manifest := uninstallManifest("digest-pull-pack", "Digest Pull Pack", "example/model", "revision-1", digest)
	writePack(t, source, manifest)
	removals := installTaglessDockerStub(t, digest)
	if _, err := packstore.Import(source, store, packstore.SourceLocal); err != nil {
		t.Fatal(err)
	}
	mustRecordOwnership(t, data, digest, manifest.Runtimes[0].Image)

	result, err := Run(store, data, models, manifest.ID, manifest.Version, DeleteEverything)
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if got := result.Items[len(result.Items)-1].Outcome; got != "Removed" {
		t.Fatalf("runtime outcome = %q", got)
	}
	if removals("image", "rm", digest) != 1 {
		t.Fatalf("digest-reference removals = %d", removals("image", "rm", digest))
	}
	if records, _ := runtime.LoadOwnership(data); len(records) != 0 {
		t.Fatalf("ownership record survived the removed image: %#v", records)
	}
}

// installTaglessDockerStub fakes Docker holding an image that is reachable
// only through its digest reference — the state a digest pull leaves behind.
// The declared tag ("example/runtime:1") never resolves.
func installTaglessDockerStub(t *testing.T, digest string) func(...string) int {
	t.Helper()
	bin := t.TempDir()
	logPath := filepath.Join(bin, "docker.log")
	script := `#!/bin/sh
echo "$@" >> "` + logPath + `"
if [ "$1" = "inspect" ]; then
  echo "Error: No such object: b70ctl-runtime" >&2
  exit 1
fi
if [ "$1" = "image" ] && [ "$2" = "inspect" ]; then
  if [ "$5" = "` + digest + `" ]; then
    echo "\"` + digest + `\""
    exit 0
  fi
  echo "Error response from daemon: No such image: $5" >&2
  exit 1
fi
if [ "$1" = "image" ] && [ "$2" = "rm" ]; then
  echo "Untagged: $3"
  exit 0
fi
exit 2
`
	if err := os.WriteFile(filepath.Join(bin, "docker"), []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", bin+string(os.PathListSeparator)+os.Getenv("PATH"))
	return func(words ...string) int {
		data, err := os.ReadFile(logPath)
		if err != nil {
			t.Fatal(err)
		}
		count := 0
		for _, line := range strings.Split(strings.TrimSpace(string(data)), "\n") {
			fields := strings.Fields(line)
			if len(fields) >= len(words) {
				matches := true
				for index, word := range words {
					if fields[index] != word {
						matches = false
						break
					}
				}
				if matches {
					count++
				}
			}
		}
		return count
	}
}

// installPullingDockerStub fakes Docker with a pullable image: image
// references resolve only after a pull, mimicking a machine that lacks the
// runtime until b70ctl acquires it.
func installPullingDockerStub(t *testing.T, digest string) func(...string) int {
	t.Helper()
	bin := t.TempDir()
	logPath := filepath.Join(bin, "docker.log")
	script := `#!/bin/sh
echo "$@" >> "` + logPath + `"
PULLED="` + filepath.Join(bin, "pulled") + `"
if [ "$1" = "pull" ]; then
  touch "$PULLED"
  exit 0
fi
if [ "$1" = "inspect" ]; then
  echo "Error: No such object: b70ctl-runtime" >&2
  exit 1
fi
if [ "$1" = "image" ] && [ "$2" = "inspect" ]; then
  if [ -f "$PULLED" ]; then
    echo "\"` + digest + `\""
    exit 0
  fi
  echo "Error response from daemon: No such image: $5" >&2
  exit 1
fi
if [ "$1" = "image" ] && [ "$2" = "rm" ]; then
  echo "Untagged: $3"
  exit 0
fi
exit 2
`
	if err := os.WriteFile(filepath.Join(bin, "docker"), []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", bin+string(os.PathListSeparator)+os.Getenv("PATH"))
	return func(words ...string) int {
		data, err := os.ReadFile(logPath)
		if err != nil {
			t.Fatal(err)
		}
		count := 0
		for _, line := range strings.Split(strings.TrimSpace(string(data)), "\n") {
			fields := strings.Fields(line)
			if len(fields) >= len(words) {
				matches := true
				for index, word := range words {
					if fields[index] != word {
						matches = false
						break
					}
				}
				if matches {
					count++
				}
			}
		}
		return count
	}
}
