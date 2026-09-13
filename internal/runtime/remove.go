package runtime

import (
	"errors"
	"strings"
)

type ImageRemovalStatus string

const (
	ImageRemovalMissing  ImageRemovalStatus = "missing"
	ImageRemovalRemoved  ImageRemovalStatus = "removed"
	ImageRemovalRetained ImageRemovalStatus = "retained"
)

type ImageRemovalResult struct {
	Status ImageRemovalStatus
	Reason string
}

func RemoveImage(image, digest string) (ImageRemovalResult, error) {
	inspection := InspectImage(image, digest)
	switch inspection.Status {
	case ImageMissing:
		return ImageRemovalResult{Status: ImageRemovalMissing}, nil
	case ImageError:
		return ImageRemovalResult{}, inspection.Err
	case ImagePresent:
	default:
		return ImageRemovalResult{}, errors.New("Docker image has an unknown inspection state")
	}

	output, err := runDocker("image", "rm", image)
	if err != nil {
		return ImageRemovalResult{
			Status: ImageRemovalRetained,
			Reason: dockerRemovalReason(output),
		}, nil
	}
	return ImageRemovalResult{Status: ImageRemovalRemoved}, nil
}

func dockerRemovalReason(output []byte) string {
	detail := strings.TrimSpace(string(output))
	lower := strings.ToLower(detail)
	if strings.Contains(lower, "is being used by") || strings.Contains(lower, "image is referenced") || strings.Contains(lower, "conflict: unable to remove") {
		return "Docker reports image is in use"
	}
	detail = strings.TrimPrefix(detail, "Error response from daemon: ")
	if line, _, found := strings.Cut(detail, "\n"); found {
		detail = line
	}
	if detail == "" {
		return "Docker refused image removal"
	}
	if len(detail) > 120 {
		detail = detail[:120] + "…"
	}
	return "Docker refused removal: " + detail
}
