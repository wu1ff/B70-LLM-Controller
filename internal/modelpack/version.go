package modelpack

import (
	"fmt"
	"regexp"
	"slices"
	"strconv"
	"strings"
)

var stableVersion = regexp.MustCompile(`^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$`)

func requireControllerVersion(minimum, installed string) error {
	if minimum == "" {
		return nil
	}
	required, ok := versionParts(minimum)
	if !ok {
		return fmt.Errorf("min_controller_version %q must be a stable version (major.minor.patch)", minimum)
	}
	base, _, prerelease := strings.Cut(installed, "-")
	current, ok := versionParts(base)
	comparison := slices.Compare(current, required)
	if !ok || comparison < 0 || (comparison == 0 && prerelease) {
		return fmt.Errorf("this pack requires Controller %s or newer (running %s)", minimum, installed)
	}
	return nil
}

func versionParts(version string) ([]uint64, bool) {
	parts := stableVersion.FindStringSubmatch(version)
	if parts == nil {
		return nil, false
	}
	numbers := make([]uint64, 0, 3)
	for _, part := range parts[1:] {
		number, err := strconv.ParseUint(part, 10, 64)
		if err != nil {
			return nil, false
		}
		numbers = append(numbers, number)
	}
	return numbers, true
}

// CatalogName lets catalog schema 1 show the requirement in older Controllers.
func (manifest *Manifest) CatalogName() string {
	if manifest.MinControllerVersion == "" {
		return manifest.Name
	}
	return fmt.Sprintf("%s (Controller %s+)", manifest.Name, manifest.MinControllerVersion)
}
