package modelpack

import (
	"errors"
	"fmt"
	"path"
	"path/filepath"
	"regexp"
	"strings"
)

var (
	packIDPattern   = regexp.MustCompile(`^[a-z0-9]+(?:[._-][a-z0-9]+)*$`)
	digestPattern   = regexp.MustCompile(`^sha256:[0-9a-f]{64}$`)
	registryPattern = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._:/-]*@sha256:[0-9a-f]{64}$`)
)

// The exact fixed v26 runtime device bind is the only pack mount Controller permits.
var fixedDeviceBind = [2]string{"--mount", "type=bind,source=/dev/dri/by-path,target=/dev/dri/by-path,readonly"}

func (manifest *Manifest) validate() error {
	if manifest.SchemaVersion != schemaVersion {
		return fmt.Errorf("schema_version must be %d", schemaVersion)
	}
	if !packIDPattern.MatchString(manifest.ID) {
		return errors.New("pack id is invalid")
	}
	if strings.TrimSpace(manifest.Name) == "" {
		return errors.New("pack name is required")
	}
	if strings.TrimSpace(manifest.Version) == "" {
		return errors.New("pack version is required")
	}
	models, err := validateModels(manifest.Models)
	if err != nil {
		return err
	}
	runtimes, err := validateRuntimes(manifest.Runtimes)
	if err != nil {
		return err
	}
	modes, err := validateModes(manifest.Modes, models)
	if err != nil {
		return err
	}
	if err := validateProfiles(manifest.Profiles, models, runtimes, modes); err != nil {
		return err
	}
	for _, profile := range manifest.Profiles {
		if _, err := Resolve(manifest, profile); err != nil {
			return fmt.Errorf("profile %q: %w", profile.ID, err)
		}
	}
	return nil
}

func validateModels(models []Model) (map[string]Model, error) {
	byID := make(map[string]Model, len(models))
	for _, model := range models {
		if model.ID == "" {
			return nil, errors.New("model id is required")
		}
		if _, exists := byID[model.ID]; exists {
			return nil, fmt.Errorf("duplicate model id %q", model.ID)
		}
		if strings.TrimSpace(model.Name) == "" {
			return nil, fmt.Errorf("model %q name is required", model.ID)
		}
		if model.Kind != "target" && model.Kind != "assistant" {
			return nil, fmt.Errorf("model %q kind is invalid", model.ID)
		}
		if strings.TrimSpace(model.Repo) == "" {
			return nil, fmt.Errorf("model %q repo is required", model.ID)
		}
		if strings.TrimSpace(model.Revision) == "" {
			return nil, fmt.Errorf("model %q revision is required", model.ID)
		}
		if len(model.Files) == 0 {
			return nil, fmt.Errorf("model %q files list must not be empty", model.ID)
		}
		paths := make(map[string]struct{}, len(model.Files))
		for _, file := range model.Files {
			clean, err := validateModelFilePath(file.Path)
			if err != nil {
				return nil, fmt.Errorf("model %q file path %q: %w", model.ID, file.Path, err)
			}
			if _, exists := paths[clean]; exists {
				return nil, fmt.Errorf("model %q has duplicate file path %q", model.ID, file.Path)
			}
			paths[clean] = struct{}{}
			if file.Size != nil && *file.Size < 0 {
				return nil, fmt.Errorf("model %q file %q size must not be negative", model.ID, file.Path)
			}
		}
		if !validMountPath(model.MountPath) {
			return nil, fmt.Errorf("model %q mount_path must be absolute and clean", model.ID)
		}
		if err := validateLaunchBlock(model.Launch, "model "+model.ID); err != nil {
			return nil, err
		}
		byID[model.ID] = model
	}
	return byID, nil
}

func validateModelFilePath(value string) (string, error) {
	if value == "" {
		return "", errors.New("must not be empty")
	}
	if filepath.IsAbs(value) {
		return "", errors.New("must be relative")
	}
	clean := filepath.Clean(filepath.FromSlash(value))
	if clean == "." {
		return "", errors.New("must name a file")
	}
	if clean == ".." || strings.HasPrefix(clean, ".."+string(filepath.Separator)) {
		return "", errors.New("must stay inside the model root")
	}
	return clean, nil
}

func validateRuntimes(runtimes []Runtime) (map[string]Runtime, error) {
	byID := make(map[string]Runtime, len(runtimes))
	for _, runtime := range runtimes {
		if runtime.ID == "" {
			return nil, errors.New("runtime id is required")
		}
		if _, exists := byID[runtime.ID]; exists {
			return nil, fmt.Errorf("duplicate runtime id %q", runtime.ID)
		}
		if strings.TrimSpace(runtime.Image) == "" {
			return nil, fmt.Errorf("runtime %q image is required", runtime.ID)
		}
		if !digestPattern.MatchString(runtime.Digest) {
			return nil, fmt.Errorf("runtime %q digest is invalid", runtime.ID)
		}
		if runtime.Registry != "" && (!registryPattern.MatchString(runtime.Registry) || strings.Contains(runtime.Registry, "://")) {
			return nil, fmt.Errorf("runtime %q registry must be an immutable @sha256 reference", runtime.ID)
		}
		if runtime.ContainerPort < 1 || runtime.ContainerPort > 65535 {
			return nil, fmt.Errorf("runtime %q container_port must be between 1 and 65535", runtime.ID)
		}
		if !strings.HasPrefix(runtime.HealthPath, "/") {
			return nil, fmt.Errorf("runtime %q health_path must start with /", runtime.ID)
		}
		if err := validateRuntimeLaunch(runtime.Launch, "runtime "+runtime.ID); err != nil {
			return nil, err
		}
		byID[runtime.ID] = runtime
	}
	return byID, nil
}

func validateModes(modes []Mode, models map[string]Model) (map[string]Mode, error) {
	byID := make(map[string]Mode, len(modes))
	for _, mode := range modes {
		if mode.ID == "" {
			return nil, errors.New("mode id is required")
		}
		if _, exists := byID[mode.ID]; exists {
			return nil, fmt.Errorf("duplicate mode id %q", mode.ID)
		}
		if strings.TrimSpace(mode.DisplayName) == "" {
			return nil, fmt.Errorf("mode %q display_name is required", mode.ID)
		}
		seen := map[string]struct{}{}
		for _, assistantID := range mode.Assistants {
			if _, duplicate := seen[assistantID]; duplicate {
				return nil, fmt.Errorf("mode %q has duplicate assistant %q", mode.ID, assistantID)
			}
			seen[assistantID] = struct{}{}
			assistant, exists := models[assistantID]
			if !exists {
				return nil, fmt.Errorf("mode %q references unknown model %q", mode.ID, assistantID)
			}
			if assistant.Kind != "assistant" {
				return nil, fmt.Errorf("mode %q model %q is not an assistant", mode.ID, assistantID)
			}
		}
		if err := validateLaunchBlock(mode.Launch, "mode "+mode.ID); err != nil {
			return nil, err
		}
		byID[mode.ID] = mode
	}
	return byID, nil
}

func validateProfiles(profiles []Profile, models map[string]Model, runtimes map[string]Runtime, modes map[string]Mode) error {
	ids := make(map[string]struct{}, len(profiles))
	qualifications := make(map[qualification]struct{}, len(profiles))
	for _, profile := range profiles {
		if profile.ID == "" {
			return errors.New("profile id is required")
		}
		if _, exists := ids[profile.ID]; exists {
			return fmt.Errorf("duplicate profile id %q", profile.ID)
		}
		ids[profile.ID] = struct{}{}
		target, exists := models[profile.ModelID]
		if !exists {
			return fmt.Errorf("unknown model %q", profile.ModelID)
		}
		if target.Kind != "target" {
			return fmt.Errorf("model %q is not a target", profile.ModelID)
		}
		if _, exists := runtimes[profile.RuntimeID]; !exists {
			return fmt.Errorf("unknown runtime %q", profile.RuntimeID)
		}
		if _, exists := modes[profile.Mode]; !exists {
			return fmt.Errorf("unknown mode %q", profile.Mode)
		}
		if profile.Cards != 1 && profile.Cards != 2 && profile.Cards != 4 {
			return errors.New("cards must be 1, 2, or 4")
		}
		if profile.TensorParallel != 1 && profile.TensorParallel != 2 && profile.TensorParallel != 4 {
			return errors.New("tensor_parallel must be 1, 2, or 4")
		}
		if profile.Context <= 0 {
			return errors.New("context must be positive")
		}
		if err := validateLaunchBlock(profile.Launch, "profile "+profile.ID); err != nil {
			return err
		}
		key := qualification{modelID: profile.ModelID, cards: profile.Cards, context: profile.Context, mode: profile.Mode}
		if _, exists := qualifications[key]; exists {
			return errors.New("duplicate qualified profile")
		}
		qualifications[key] = struct{}{}
	}
	return nil
}

type qualification struct {
	modelID string
	cards   int
	context int
	mode    string
}

func validateLaunchBlock(launch LaunchBlock, label string) error {
	return validateLaunchValues(launch.DockerArgs, launch.Environment, label)
}

func validateRuntimeLaunch(launch RuntimeLaunch, label string) error {
	return validateLaunchValues(launch.DockerArgs, launch.Environment, label)
}

func validateLaunchValues(dockerArgs []string, environment map[string]string, label string) error {
	for index, arg := range dockerArgs {
		if arg == fixedDeviceBind[0] && index+1 < len(dockerArgs) && dockerArgs[index+1] == fixedDeviceBind[1] {
			continue
		}
		if forbiddenDockerArg(arg) {
			return fmt.Errorf("%s docker argument %q is controlled by B70 LLM Controller", label, arg)
		}
	}
	for name := range environment {
		if strings.TrimSpace(name) == "" || strings.Contains(name, "=") {
			return fmt.Errorf("%s environment name %q is invalid", label, name)
		}
	}
	return nil
}

func forbiddenDockerArg(arg string) bool {
	for _, option := range []string{"--name", "--rm", "--publish", "--publish-all", "--network", "--volume", "--mount", "--env", "--env-file", "--label"} {
		if arg == option || strings.HasPrefix(arg, option+"=") {
			return true
		}
	}
	for _, option := range []string{"-p", "-P", "-v", "-e", "-l"} {
		if strings.HasPrefix(arg, option) {
			return true
		}
	}
	return false
}

func validMountPath(value string) bool {
	return path.IsAbs(value) && path.Clean(value) == value
}
