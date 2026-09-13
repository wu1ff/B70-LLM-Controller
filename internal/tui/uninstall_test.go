package tui

import (
	"os"
	"strings"
	"testing"

	"b70ctl/internal/packstore"
)

func TestUninstallSelectionDefaultsToCancel(t *testing.T) {
	application, output := uninstallTestApp(t, []byte{'\n'})
	removed, err := application.uninstallPack(testLoadedPack())
	if err != nil || removed {
		t.Fatalf("uninstallPack() = %v, %v", removed, err)
	}
	if got := readTestOutput(t, output); !strings.Contains(stripANSI(got), "▸ CANCEL") {
		t.Fatalf("uninstall screen did not default to Cancel: %q", got)
	}
}

func TestUninstallConfirmationDefaultsToNo(t *testing.T) {
	events := []byte{27, '[', 'A', 27, '[', 'A', '\n', '\n'}
	application, output := uninstallTestApp(t, events)
	removed, err := application.uninstallPack(testLoadedPack())
	if err != nil || removed {
		t.Fatalf("uninstallPack() = %v, %v", removed, err)
	}
	if got := readTestOutput(t, output); !strings.Contains(got, "Uninstall test-pack 1.0.0?") || !strings.Contains(stripANSI(got), "▸ NO") {
		t.Fatalf("confirmation did not default to No: %q", got)
	}
}

func uninstallTestApp(t *testing.T, events []byte) (*app, *os.File) {
	t.Helper()
	output, err := os.CreateTemp(t.TempDir(), "output")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { output.Close() })
	input := &input{
		bytes:  make(chan byte, len(events)),
		errors: make(chan error, 1),
		resize: make(chan os.Signal, 1),
	}
	for _, event := range events {
		input.bytes <- event
	}
	return &app{in: input, out: output}, output
}

func testLoadedPack() loadedPack {
	return loadedPack{installed: packstore.InstalledPack{ID: "test-pack", Name: "Test Pack", Version: "1.0.0", Source: "local"}}
}

func readTestOutput(t *testing.T, output *os.File) string {
	t.Helper()
	if _, err := output.Seek(0, 0); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(output.Name())
	if err != nil {
		t.Fatal(err)
	}
	return string(data)
}
