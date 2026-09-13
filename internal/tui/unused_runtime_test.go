package tui

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"b70ctl/internal/config"
	"b70ctl/internal/modelpack"
	"b70ctl/internal/packstore"
	"b70ctl/internal/runtime"
)

func unusedDigest(char byte) string {
	return "sha256:" + strings.Repeat(string(char), 64)
}

func unusedRuntimeTestApp(t *testing.T, events []byte, dataRoot, packsRoot string) (*app, *os.File) {
	t.Helper()
	application, output := uninstallTestApp(t, events)
	application.config = config.Config{ModelDirectory: t.TempDir(), DefaultAccess: config.AccessLocal, DefaultPort: 8000}
	application.paths = config.Paths{DataRoot: dataRoot, PacksRoot: packsRoot}
	return application, output
}

func mustOwn(t *testing.T, dataRoot, digest, image string) {
	t.Helper()
	if err := runtime.RecordOwnership(dataRoot, digest, image); err != nil {
		t.Fatal(err)
	}
}

func ownedRecords(t *testing.T, dataRoot string) map[string]runtime.OwnershipRecord {
	t.Helper()
	records, err := runtime.LoadOwnership(dataRoot)
	if err != nil {
		t.Fatal(err)
	}
	return records
}

// dockerStubForUnusedRuntimes installs a Docker stub serving the given image
// ID for every image reference, an optional managed container, and optional
// removal refusal; it returns the path of the invocation log. The image ID
// travels through an environment variable so no digest is ever shell-quoted.
func dockerStubForUnusedRuntimes(t *testing.T, imageID, container string, refuseRemoval bool) string {
	t.Helper()
	bin := t.TempDir()
	logPath := filepath.Join(bin, "calls.log")
	script := `#!/bin/sh
echo "$@" >> "` + logPath + `"
IMAGE_ID="` + imageID + `"
CONTAINER_FILE="` + filepath.Join(bin, "container.json") + `"
if [ "$1" = "inspect" ]; then
  if [ -s "$CONTAINER_FILE" ]; then
    cat "$CONTAINER_FILE"
    exit 0
  fi
  echo "Error: No such object: b70ctl-runtime" >&2
  exit 1
fi
if [ "$1" = "image" ] && [ "$2" = "inspect" ]; then
`
	if imageID != "" {
		script += `  echo "\"$IMAGE_ID\""
  exit 0
`
	} else {
		script += `  echo "Error response from daemon: No such image: $5" >&2
  exit 1
`
	}
	script += `fi
if [ "$1" = "image" ] && [ "$2" = "rm" ]; then
`
	if refuseRemoval {
		script += `  echo "Error response from daemon: conflict: unable to delete - image is being used by stopped container" >&2
  exit 1
`
	} else {
		script += `  echo "Untagged: $3"
  exit 0
`
	}
	script += `fi
exit 2
`
	if container != "" {
		if err := os.WriteFile(filepath.Join(bin, "container.json"), []byte(container), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.WriteFile(filepath.Join(bin, "docker"), []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(logPath, nil, 0o644); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", bin+string(os.PathListSeparator)+os.Getenv("PATH"))
	return logPath
}

func readDockerCalls(t *testing.T, logPath string) []string {
	t.Helper()
	data, err := os.ReadFile(logPath)
	if err != nil {
		t.Fatal(err)
	}
	return strings.Split(strings.TrimSpace(string(data)), "\n")
}

func TestUnusedOwnedRuntimesListsOnlyOwnedUnreferencedRecords(t *testing.T) {
	data := t.TempDir()
	packsRoot := t.TempDir()
	application, _ := unusedRuntimeTestApp(t, nil, data, packsRoot)

	referenced := unusedDigest('a')
	ownedUnused := unusedDigest('b')
	mustOwn(t, data, referenced, "example/referenced:1")
	mustOwn(t, data, ownedUnused, "example/unused:1")
	// A pack referencing one digest keeps that runtime out of the list even
	// though it is owned.
	installTuiPack(t, packsRoot, "pack-a", "Pack A", tuiTestModel())
	rewritePackRuntimeDigest(t, packsRoot, "pack-a", referenced)

	entries, err := application.unusedOwnedRuntimes()
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 1 || entries[0].Digest != ownedUnused {
		t.Fatalf("unused runtimes = %#v", entries)
	}
}

// FLOW D — a local Docker image can look exactly like a b70ctl runtime, but
// without an ownership record it never appears and is never deletable here.
func TestArbitraryAndPreExistingImagesNeverAppearUnused(t *testing.T) {
	data := t.TempDir()
	packsRoot := t.TempDir()
	application, _ := unusedRuntimeTestApp(t, nil, data, packsRoot)
	dockerStubForUnusedRuntimes(t, unusedDigest('a'), "", false)

	entries, err := application.unusedOwnedRuntimes()
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 0 {
		t.Fatalf("images without provenance were listed: %#v", entries)
	}
}

func TestInstallingPackHidesUnusedOwnedRuntime(t *testing.T) {
	data := t.TempDir()
	packsRoot := t.TempDir()
	application, _ := unusedRuntimeTestApp(t, nil, data, packsRoot)
	digest := unusedDigest('c')
	mustOwn(t, data, digest, "example/runtime:1")

	if entries, _ := application.unusedOwnedRuntimes(); len(entries) != 1 {
		t.Fatalf("owned unreferenced runtime not listed: %#v", entries)
	}
	installTuiPack(t, packsRoot, "pack-a", "Pack A", tuiTestModel())
	rewritePackRuntimeDigest(t, packsRoot, "pack-a", digest)

	entries, err := application.unusedOwnedRuntimes()
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 0 {
		t.Fatalf("referenced runtime still listed as unused: %#v", entries)
	}
	if _, kept := ownedRecords(t, data)[digest]; !kept {
		t.Fatal("ownership record dropped while the image became referenced")
	}
}

func rewritePackRuntimeDigest(t *testing.T, packsRoot, packID, digest string) {
	t.Helper()
	packs, err := packstore.List(packsRoot)
	if err != nil {
		t.Fatal(err)
	}
	for _, pack := range packs {
		if pack.ID != packID {
			continue
		}
		installed := filepath.Join(packsRoot, pack.ID, pack.Version)
		manifest, err := modelpack.Load(installed)
		if err != nil {
			t.Fatal(err)
		}
		manifest.Runtimes[0].Digest = digest
		source := t.TempDir()
		writeTuiPack(t, source, *manifest)
		if err := os.RemoveAll(installed); err != nil {
			t.Fatal(err)
		}
		if _, err := packstore.Import(source, packsRoot, packstore.SourceLocal); err != nil {
			t.Fatal(err)
		}
		return
	}
	t.Fatalf("pack %s not installed", packID)
}

func runUnusedRuntimesScreen(t *testing.T, application *app) {
	t.Helper()
	done := make(chan error, 1)
	go func() { done <- application.unusedRuntimesScreen() }()
	t.Cleanup(func() {
		select {
		case err := <-done:
			if err != nil {
				t.Fatalf("unusedRuntimesScreen() error = %v", err)
			}
		case <-time.After(5 * time.Second):
			t.Fatal("unused runtimes screen did not exit")
		}
	})
}

// FLOW C — the standalone cleanup flow: list, confirm, exact removal, and
// unrelated images untouched.
func TestUnusedRuntimeRemovalFlowRemovesExactImageAndRecord(t *testing.T) {
	data := t.TempDir()
	packsRoot := t.TempDir()
	digest := unusedDigest('d')
	mustOwn(t, data, digest, "example/runtime:1")
	events := []byte{'\n', '\n', 27, '[', 'A', '\n', 27, '[', 'B', '\n', 27}
	application, output := unusedRuntimeTestApp(t, events, data, packsRoot)
	logPath := dockerStubForUnusedRuntimes(t, digest, "", false)

	runUnusedRuntimesScreen(t, application)
	sendTestBytes(t, application, events...)

	got := stripANSI(readTestOutput(t, output))
	if !strings.Contains(got, "UNUSED") || !strings.Contains(got, "runtime:1") {
		t.Fatalf("unused runtime row missing:\n%s", got)
	}
	if !strings.Contains(got, "Remove runtime image") || !strings.Contains(got, "▸ NO") {
		t.Fatalf("destructive confirmation missing or unsafe default:\n%s", got)
	}
	if !strings.Contains(got, "Runtime image removed.") {
		t.Fatalf("removal outcome missing:\n%s", got)
	}
	if len(ownedRecords(t, data)) != 0 {
		t.Fatalf("ownership record survived removal: %#v", ownedRecords(t, data))
	}
	removals := 0
	for _, call := range readDockerCalls(t, logPath) {
		if strings.HasPrefix(call, "image rm ") {
			removals++
			if !strings.Contains(call, "example/runtime:1") {
				t.Fatalf("unexpected image removed: %q", call)
			}
		}
	}
	if removals != 1 {
		t.Fatalf("image rm invocations = %d in %v", removals, readDockerCalls(t, logPath))
	}
}

func TestUnusedRuntimeRemovalRequiresExplicitConfirmation(t *testing.T) {
	data := t.TempDir()
	packsRoot := t.TempDir()
	digest := unusedDigest('e')
	mustOwn(t, data, digest, "example/runtime:1")
	// Open detail, choose Remove, confirm the default No, then leave.
	events := []byte{'\n', '\n', '\n', 27, '[', 'B', '\n', 27}
	application, output := unusedRuntimeTestApp(t, events, data, packsRoot)
	logPath := dockerStubForUnusedRuntimes(t, digest, "", false)

	runUnusedRuntimesScreen(t, application)
	sendTestBytes(t, application, events...)

	if strings.Contains(stripANSI(readTestOutput(t, output)), "Runtime image removed.") {
		t.Fatal("removal happened without explicit confirmation")
	}
	if _, kept := ownedRecords(t, data)[digest]; !kept {
		t.Fatal("ownership record dropped without confirmation")
	}
	for _, call := range readDockerCalls(t, logPath) {
		if strings.HasPrefix(call, "image rm ") {
			t.Fatalf("docker removal attempted without confirmation: %q", call)
		}
	}
}

func TestUnusedRuntimeDockerRefusalRetainsImageAndRecord(t *testing.T) {
	data := t.TempDir()
	packsRoot := t.TempDir()
	digest := unusedDigest('f')
	mustOwn(t, data, digest, "example/runtime:1")
	events := []byte{'\n', '\n', 27, '[', 'A', '\n', 27, '[', 'B', '\n', 27}
	application, output := unusedRuntimeTestApp(t, events, data, packsRoot)
	dockerStubForUnusedRuntimes(t, digest, "", true)

	runUnusedRuntimesScreen(t, application)
	sendTestBytes(t, application, events...)

	got := stripANSI(readTestOutput(t, output))
	if !strings.Contains(got, "Not removed") || !strings.Contains(got, "Docker reports image is in use") {
		t.Fatalf("docker refusal not surfaced:\n%s", got)
	}
	if _, kept := ownedRecords(t, data)[digest]; !kept {
		t.Fatal("ownership record dropped after Docker refusal")
	}
}

func TestUnusedRuntimeMissingImageCleansOnlyStaleMetadata(t *testing.T) {
	data := t.TempDir()
	packsRoot := t.TempDir()
	digest := unusedDigest('1')
	mustOwn(t, data, digest, "example/runtime:1")
	events := []byte{'\n', '\n', 27, '[', 'A', '\n', 27, '[', 'B', '\n', 27}
	application, output := unusedRuntimeTestApp(t, events, data, packsRoot)
	logPath := dockerStubForUnusedRuntimes(t, "", "", false)

	runUnusedRuntimesScreen(t, application)
	sendTestBytes(t, application, events...)

	got := stripANSI(readTestOutput(t, output))
	if !strings.Contains(got, "ownership record cleared") {
		t.Fatalf("stale metadata cleanup not reported:\n%s", got)
	}
	if len(ownedRecords(t, data)) != 0 {
		t.Fatalf("stale ownership record kept: %#v", ownedRecords(t, data))
	}
	for _, call := range readDockerCalls(t, logPath) {
		if strings.HasPrefix(call, "image rm ") {
			t.Fatalf("docker removal attempted for missing image: %q", call)
		}
	}
}

func TestUnusedRuntimeActiveModelBlocksRemoval(t *testing.T) {
	data := t.TempDir()
	packsRoot := t.TempDir()
	digest := unusedDigest('2')
	mustOwn(t, data, digest, "example/runtime:1")
	events := []byte{'\n', '\n', 27, '[', 'A', '\n', 27, '[', 'B', '\n', 27}
	application, output := unusedRuntimeTestApp(t, events, data, packsRoot)
	container := `[{"Config":{"Labels":{"b70ctl.managed":"true","b70ctl.pack_id":"pack","b70ctl.pack_version":"1.0.0","b70ctl.profile_id":"profile","b70ctl.health_path":"/health"}},"Image":"` + digest + `","State":{"Running":true,"ExitCode":0},"NetworkSettings":{"Ports":{"8000/tcp":[{"HostIp":"127.0.0.1","HostPort":"65534"}]}}}]`
	logPath := dockerStubForUnusedRuntimes(t, digest, container, false)

	runUnusedRuntimesScreen(t, application)
	sendTestBytes(t, application, events...)

	got := stripANSI(readTestOutput(t, output))
	if !strings.Contains(got, "in use by the running model") {
		t.Fatalf("active runtime did not block removal:\n%s", got)
	}
	if _, kept := ownedRecords(t, data)[digest]; !kept {
		t.Fatal("ownership record dropped while the runtime was active")
	}
	for _, call := range readDockerCalls(t, logPath) {
		if strings.HasPrefix(call, "image rm ") {
			t.Fatalf("docker removal attempted for active runtime: %q", call)
		}
	}
}

func TestUnusedRuntimeUnresolvableStateFailsClosed(t *testing.T) {
	data := t.TempDir()
	packsRoot := t.TempDir()
	digest := unusedDigest('3')
	mustOwn(t, data, digest, "example/runtime:1")
	events := []byte{'\n', '\n', 27, '[', 'A', '\n', 27, '[', 'B', '\n', 27}
	application, output := unusedRuntimeTestApp(t, events, data, packsRoot)

	bin := t.TempDir()
	script := "#!/bin/sh\necho 'cannot connect to the docker daemon' >&2\nexit 1\n"
	if err := os.WriteFile(filepath.Join(bin, "docker"), []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", bin+string(os.PathListSeparator)+os.Getenv("PATH"))

	runUnusedRuntimesScreen(t, application)
	sendTestBytes(t, application, events...)

	got := stripANSI(readTestOutput(t, output))
	if !strings.Contains(got, "runtime state is unavailable") {
		t.Fatalf("unresolvable runtime state did not fail closed:\n%s", got)
	}
	if _, kept := ownedRecords(t, data)[digest]; !kept {
		t.Fatal("ownership record dropped despite unresolved state")
	}
}

func TestModelPacksMenuOffersUnusedRuntimes(t *testing.T) {
	data := t.TempDir()
	packsRoot := t.TempDir()
	application, output := unusedRuntimeTestApp(t, []byte{27}, data, packsRoot)
	if err := application.modelPacksScreen(); err != nil {
		t.Fatal(err)
	}
	got := stripANSI(readTestOutput(t, output))
	if !strings.Contains(got, "UNUSED RUNTIMES") {
		t.Fatalf("Model Packs menu lost the Unused Runtimes entry:\n%s", got)
	}
}
