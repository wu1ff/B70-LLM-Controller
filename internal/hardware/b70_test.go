package hardware

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func TestDetectB70IdentityFiltering(t *testing.T) {
	root := t.TempDir()
	writePCIDevice(t, root, "0000:03:00.0", b70VendorID, b70DeviceID)
	writePCIDevice(t, root, "0000:04:00.0", b70VendorID, "0x56a0")
	writePCIDevice(t, root, "0000:05:00.0", "0x10de", b70DeviceID)

	devices, err := detectB70(root)
	if err != nil {
		t.Fatalf("detectB70() error = %v", err)
	}
	want := []B70Device{{PCIAddress: "0000:03:00.0"}}
	if !reflect.DeepEqual(devices, want) {
		t.Fatalf("detectB70() = %#v, want %#v", devices, want)
	}
}

func TestDetectB70ReturnsPCIOrder(t *testing.T) {
	root := t.TempDir()
	writePCIDevice(t, root, "0000:c8:00.0", b70VendorID, b70DeviceID)
	writePCIDevice(t, root, "0000:03:00.0", b70VendorID, b70DeviceID)
	writePCIDevice(t, root, "0000:83:00.0", b70VendorID, b70DeviceID)
	writePCIDevice(t, root, "0000:43:00.0", b70VendorID, b70DeviceID)

	devices, err := detectB70(root)
	if err != nil {
		t.Fatalf("detectB70() error = %v", err)
	}
	want := []B70Device{
		{PCIAddress: "0000:03:00.0"},
		{PCIAddress: "0000:43:00.0"},
		{PCIAddress: "0000:83:00.0"},
		{PCIAddress: "0000:c8:00.0"},
	}
	if !reflect.DeepEqual(devices, want) {
		t.Fatalf("detectB70() = %#v, want %#v", devices, want)
	}
}

func TestDetectB70ZeroCards(t *testing.T) {
	devices, err := detectB70(t.TempDir())
	if err != nil {
		t.Fatalf("detectB70() error = %v", err)
	}
	if devices == nil || len(devices) != 0 {
		t.Fatalf("detectB70() = %#v, want empty inventory", devices)
	}
}

func writePCIDevice(t *testing.T, root, address, vendor, device string) {
	t.Helper()
	directory := filepath.Join(root, address)
	if err := os.MkdirAll(directory, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(directory, "vendor"), []byte(vendor+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(directory, "device"), []byte(device+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
}
