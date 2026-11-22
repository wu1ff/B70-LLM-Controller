package runtime

import (
	"crypto/sha256"
	"encoding/csv"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"

	"b70ctl/internal/config"
	"b70ctl/internal/modelpack"
)

func persistentJITCacheMount(imageID string) (string, error) {
	paths, err := config.ResolvePaths()
	if err != nil {
		return "", err
	}
	if !filepath.IsAbs(paths.DataRoot) {
		return "", errors.New("persistent JIT cache requires an absolute Controller data directory")
	}
	identity, err := jitHostIdentity()
	if err != nil {
		return "", fmt.Errorf("identify host for persistent JIT cache: %w", err)
	}
	directory, err := prepareJITCache(paths.DataRoot, imageID, identity)
	if err != nil {
		return "", err
	}
	var mount strings.Builder
	writer := csv.NewWriter(&mount)
	if err := writer.Write([]string{"type=bind", "source=" + directory, "target=" + modelpack.JITCacheMountPath}); err != nil {
		return "", err
	}
	writer.Flush()
	return strings.TrimSuffix(mount.String(), "\n"), writer.Error()
}

func prepareJITCache(dataRoot, imageID string, hostIdentity []string) (string, error) {
	if !imageIDPattern.MatchString(imageID) {
		return "", errors.New("persistent JIT cache requires an immutable image ID")
	}
	hostKey := sha256.Sum256([]byte(strings.Join(hostIdentity, "\x00")))
	directory := filepath.Join(dataRoot, "jit-cache", strings.TrimPrefix(imageID, "sha256:"), fmt.Sprintf("%x", hostKey))
	if err := os.MkdirAll(directory, 0o700); err != nil {
		return "", fmt.Errorf("create persistent JIT cache: %w", err)
	}
	return directory, nil
}

func jitHostIdentity() ([]string, error) {
	var identity []string
	for _, name := range []string{"/proc/sys/kernel/osrelease", "/proc/sys/kernel/version", "/sys/module/xe/srcversion", "/sys/module/xe/notes/.note.gnu.build-id"} {
		value, err := os.ReadFile(name)
		if err != nil && !(strings.HasPrefix(name, "/sys/") && errors.Is(err, os.ErrNotExist)) {
			return nil, fmt.Errorf("read %s: %w", name, err)
		}
		identity = append(identity, fmt.Sprintf("%s=%x", name, value))
	}
	packages, err := exec.Command("dpkg-query", "-W", "-f=${db:Status-Status}\t${binary:Package}\t${Version}\n").Output()
	if err != nil {
		return nil, fmt.Errorf("read installed Intel runtime package versions: %w", err)
	}
	return append(identity, intelRuntimePackages(string(packages))...), nil
}

func intelRuntimePackages(output string) []string {
	var versions []string
	for _, line := range strings.Split(output, "\n") {
		fields := strings.Split(line, "\t")
		if len(fields) != 3 || fields[0] != "installed" {
			continue
		}
		name, _, _ := strings.Cut(fields[1], ":")
		if name == "intel-opencl-icd" || name == "intel-level-zero-gpu" || name == "libdrm-intel1" ||
			strings.HasPrefix(name, "libze") || strings.HasPrefix(name, "libigc") || strings.HasPrefix(name, "libigdgmm") {
			versions = append(versions, fields[1]+"="+fields[2])
		}
	}
	slices.Sort(versions)
	return versions
}
