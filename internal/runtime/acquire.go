package runtime

import (
	"context"
	"errors"
	"os/exec"
	"regexp"
	"strings"
	"sync"
	"unicode"

	"b70ctl/internal/modelpack"
)

type AcquisitionOutcome string

const (
	AcquisitionReused     AcquisitionOutcome = "Reused"
	AcquisitionDownloaded AcquisitionOutcome = "Downloaded"
	AcquisitionFailed     AcquisitionOutcome = "Failed"
)

type AcquisitionResult struct {
	Outcome AcquisitionOutcome
	Reason  string
	Err     error
}

type PullProgress struct {
	Status   string
	Activity string
	Latest   string
}

var immutableRegistryPattern = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._:/-]*@sha256:[0-9a-f]{64}$`)

func Acquire(ctx context.Context, declared modelpack.Runtime, notify func(PullProgress)) AcquisitionResult {
	inspection := InspectRuntime(declared)
	switch inspection.Status {
	case ImagePresent:
		return AcquisitionResult{Outcome: AcquisitionReused}
	case ImageError:
		return failedAcquisition(acquisitionFailureReason(inspection.Err), inspection.Err)
	case ImageMissing:
	default:
		return failedAcquisition("runtime image inspection failed", errors.New("unknown Docker image inspection state"))
	}

	if !immutableRegistryPattern.MatchString(declared.Registry) || strings.Contains(declared.Registry, "://") {
		return failedAcquisition("immutable registry reference not configured", errors.New("runtime has no valid immutable registry reference"))
	}
	if notify != nil {
		notify(PullProgress{Status: "Pulling", Activity: "Contacting registry"})
	}
	output, err := runDockerPull(ctx, declared.Registry, notify)
	if err != nil {
		if errors.Is(err, exec.ErrNotFound) {
			return failedAcquisition("Docker unavailable", err)
		}
		return failedAcquisition("Docker pull failed", dockerError("pull runtime image", output, err))
	}

	inspection = InspectImage(declared.Registry, declared.Digest)
	if inspection.Status != ImagePresent {
		if inspection.Status == ImageError {
			return failedAcquisition(acquisitionFailureReason(inspection.Err), inspection.Err)
		}
		return failedAcquisition("runtime image identity mismatch", errors.New("runtime image identity mismatch"))
	}
	return AcquisitionResult{Outcome: AcquisitionDownloaded}
}

func InspectRuntime(declared modelpack.Runtime) ImageResult {
	inspection := InspectImage(declared.Image, declared.Digest)
	if inspection.Status != ImageMissing || declared.Image == declared.Digest {
		return inspection
	}
	return InspectImage(declared.Digest, declared.Digest)
}

func failedAcquisition(reason string, err error) AcquisitionResult {
	return AcquisitionResult{Outcome: AcquisitionFailed, Reason: reason, Err: err}
}

func acquisitionFailureReason(err error) string {
	if errors.Is(err, exec.ErrNotFound) {
		return "Docker unavailable"
	}
	message := strings.ToLower(err.Error())
	if strings.Contains(message, "cannot connect to the docker daemon") || strings.Contains(message, "docker daemon is not running") {
		return "Docker unavailable"
	}
	return "runtime image inspection failed"
}

type dockerProgressWriter struct {
	mu      sync.Mutex
	pending []byte
	lines   []string
	notify  func(PullProgress)
}

func newDockerProgressWriter(notify func(PullProgress)) *dockerProgressWriter {
	return &dockerProgressWriter{notify: notify}
}

func (writer *dockerProgressWriter) Write(data []byte) (int, error) {
	writer.mu.Lock()
	defer writer.mu.Unlock()
	for _, value := range data {
		if value == '\n' || value == '\r' {
			writer.emitLocked()
			continue
		}
		if len(writer.pending) < 4096 {
			writer.pending = append(writer.pending, value)
		}
	}
	return len(data), nil
}

func (writer *dockerProgressWriter) flush() {
	writer.mu.Lock()
	defer writer.mu.Unlock()
	writer.emitLocked()
}

func (writer *dockerProgressWriter) emitLocked() {
	line := safeDockerProgressLine(string(writer.pending))
	writer.pending = writer.pending[:0]
	if line == "" {
		return
	}
	writer.lines = append(writer.lines, line)
	if len(writer.lines) > 20 {
		writer.lines = append([]string(nil), writer.lines[len(writer.lines)-20:]...)
	}
	if writer.notify != nil {
		writer.notify(PullProgress{Status: "Pulling", Activity: dockerActivity(line), Latest: line})
	}
}

func (writer *dockerProgressWriter) summary() string {
	writer.mu.Lock()
	defer writer.mu.Unlock()
	return strings.Join(writer.lines, "\n")
}

func safeDockerProgressLine(value string) string {
	value = strings.TrimSpace(strings.Map(func(character rune) rune {
		if unicode.IsControl(character) {
			return -1
		}
		return character
	}, value))
	if value == "" {
		return ""
	}
	lower := strings.ToLower(value)
	for _, sensitive := range []string{"authorization", "credential", "password", "bearer ", "token="} {
		if strings.Contains(lower, sensitive) {
			return ""
		}
	}
	if len(value) > 240 {
		value = value[:240] + "…"
	}
	return value
}

func dockerActivity(line string) string {
	lower := strings.ToLower(line)
	switch {
	case strings.Contains(lower, "download") || strings.Contains(lower, "pulling fs layer"):
		return "Downloading layers"
	case strings.Contains(lower, "extract"):
		return "Extracting layers"
	case strings.Contains(lower, "verif"):
		return "Verifying layers"
	case strings.Contains(lower, "digest:") || strings.Contains(lower, "status:"):
		return "Finalizing image"
	default:
		return "Pulling image"
	}
}
