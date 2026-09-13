package tui

import (
	"fmt"
	"sort"

	"b70ctl/internal/modelpack"
)

func cardOptions(profiles []modelpack.Profile, modelID string) []int {
	values := map[int]struct{}{}
	for _, profile := range profiles {
		if profile.ModelID == modelID {
			values[profile.Cards] = struct{}{}
		}
	}
	return sortedInts(values)
}

func contextOptions(profiles []modelpack.Profile, modelID string, cards int) []int {
	values := map[int]struct{}{}
	for _, profile := range profiles {
		if profile.ModelID == modelID && profile.Cards == cards {
			values[profile.Context] = struct{}{}
		}
	}
	return sortedInts(values)
}

func modeOptions(manifest *modelpack.Manifest, profiles []modelpack.Profile, modelID string, cards, context int) []string {
	values := map[string]struct{}{}
	for _, profile := range profiles {
		if profile.ModelID == modelID && profile.Cards == cards && profile.Context == context {
			values[profile.Mode] = struct{}{}
		}
	}
	result := make([]string, 0, len(values))
	if manifest != nil {
		for _, mode := range manifest.Modes {
			if _, exists := values[mode.ID]; exists {
				result = append(result, mode.ID)
				delete(values, mode.ID)
			}
		}
	}
	remaining := make([]string, 0, len(values))
	for value := range values {
		remaining = append(remaining, value)
	}
	sort.Strings(remaining)
	result = append(result, remaining...)
	return result
}

func exactProfile(profiles []modelpack.Profile, modelID string, cards, context int, mode string) (modelpack.Profile, bool) {
	for _, profile := range profiles {
		if profile.ModelID == modelID && profile.Cards == cards && profile.Context == context && profile.Mode == mode {
			return profile, true
		}
	}
	return modelpack.Profile{}, false
}

func formatContext(context int) string {
	if context >= 1024 && context%1024 == 0 {
		return fmt.Sprintf("%dK", context/1024)
	}
	return fmt.Sprintf("%d", context)
}

func modeName(manifest *modelpack.Manifest, modeID string) string {
	if manifest != nil {
		for _, mode := range manifest.Modes {
			if mode.ID == modeID {
				return mode.DisplayName
			}
		}
	}
	return modeID
}

func sortedInts(values map[int]struct{}) []int {
	result := make([]int, 0, len(values))
	for value := range values {
		result = append(result, value)
	}
	sort.Ints(result)
	return result
}
