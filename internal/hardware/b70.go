package hardware

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

const pciDevicesPath = "/sys/bus/pci/devices"

const (
	b70VendorID = "0x8086"
	b70DeviceID = "0xe223"
)

type B70Device struct {
	PCIAddress string
}

func DetectB70() ([]B70Device, error) {
	return detectB70(pciDevicesPath)
}

func detectB70(root string) ([]B70Device, error) {
	entries, err := os.ReadDir(root)
	if err != nil {
		return nil, fmt.Errorf("read PCI devices: %w", err)
	}

	devices := make([]B70Device, 0)
	for _, entry := range entries {
		vendor, err := os.ReadFile(filepath.Join(root, entry.Name(), "vendor"))
		if err != nil {
			continue
		}
		device, err := os.ReadFile(filepath.Join(root, entry.Name(), "device"))
		if err != nil {
			continue
		}
		if strings.ToLower(strings.TrimSpace(string(vendor))) != b70VendorID ||
			strings.ToLower(strings.TrimSpace(string(device))) != b70DeviceID {
			continue
		}
		devices = append(devices, B70Device{PCIAddress: entry.Name()})
	}

	sort.Slice(devices, func(i, j int) bool {
		return devices[i].PCIAddress < devices[j].PCIAddress
	})
	return devices, nil
}
