package runtime

import (
	"encoding/json"
	"errors"
	"fmt"
	"regexp"
	"strings"

	"b70ctl/internal/modelpack"
)

type ImageStatus uint8

const (
	ImageError ImageStatus = iota
	ImageMissing
	ImagePresent
)

type ImageResult struct {
	Status ImageStatus
	Err    error
}

var imageIDPattern = regexp.MustCompile(`^sha256:[0-9a-f]{64}$`)

func InspectImage(image, digest string) ImageResult {
	output, err := runDocker("image", "inspect", "--format", "{{json .Id}}", image)
	if err != nil {
		if strings.Contains(string(output), "No such image:") {
			return ImageResult{Status: ImageMissing}
		}
		return ImageResult{Status: ImageError, Err: fmt.Errorf("inspect Docker image %q: %w", image, err)}
	}

	var imageID string
	if err := json.Unmarshal(output, &imageID); err != nil || !imageIDPattern.MatchString(imageID) {
		return ImageResult{Status: ImageError, Err: errors.New("Docker image inspect returned an invalid image ID")}
	}
	if imageID != digest {
		return ImageResult{Status: ImageMissing}
	}
	return ImageResult{Status: ImagePresent}
}

func InspectRuntimes(runtimes []modelpack.Runtime) map[string]ImageResult {
	results := make(map[string]ImageResult, len(runtimes))
	for _, runtime := range runtimes {
		results[runtime.ID] = InspectRuntime(runtime)
	}
	return results
}
