package runtime

import (
	"fmt"

	"b70ctl/internal/modelpack"
	"b70ctl/internal/modelstore"
)

type ProfileAvailability struct {
	ProfileID string
	ModelID   string
	ModelName string
	Cards     int
	Context   int
	Mode      string
	Available bool
	Missing   []string
}

func Resolve(
	manifest *modelpack.Manifest,
	models []modelstore.Artifact,
	runtimes map[string]ImageResult,
	availableCards int,
) []ProfileAvailability {
	modelsByID := make(map[string]modelpack.Model, len(manifest.Models))
	for _, model := range manifest.Models {
		modelsByID[model.ID] = model
	}
	modesByID := make(map[string]modelpack.Mode, len(manifest.Modes))
	for _, mode := range manifest.Modes {
		modesByID[mode.ID] = mode
	}

	resolved := make([]ProfileAvailability, 0, len(manifest.Profiles))
	for _, profile := range manifest.Profiles {
		target := modelsByID[profile.ModelID]
		var missing []string
		switch modelstore.Assess(models, target).State {
		case modelstore.Missing:
			missing = append(missing, "target model")
		case modelstore.Incomplete:
			missing = append(missing, "target model incomplete")
		}
		for _, assistantID := range modesByID[profile.Mode].Assistants {
			assistant := modelsByID[assistantID]
			switch modelstore.Assess(models, assistant).State {
			case modelstore.Missing:
				missing = append(missing, "assistant "+assistantID)
			case modelstore.Incomplete:
				missing = append(missing, "assistant "+assistantID+" incomplete")
			}
		}
		if runtimes[profile.RuntimeID].Status != ImagePresent {
			missing = append(missing, "runtime "+profile.RuntimeID)
		}
		if profile.Cards > availableCards {
			missing = append(missing, fmt.Sprintf("requires %d cards", profile.Cards))
		}

		resolved = append(resolved, ProfileAvailability{
			ProfileID: profile.ID,
			ModelID:   target.ID,
			ModelName: target.Name,
			Cards:     profile.Cards,
			Context:   profile.Context,
			Mode:      profile.Mode,
			Available: len(missing) == 0,
			Missing:   missing,
		})
	}
	return resolved
}
