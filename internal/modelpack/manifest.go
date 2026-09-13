package modelpack

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

const schemaVersion = 1

type Manifest struct {
	SchemaVersion int       `json:"schema_version"`
	ID            string    `json:"id"`
	Name          string    `json:"name"`
	Version       string    `json:"version"`
	Models        []Model   `json:"models"`
	Runtimes      []Runtime `json:"runtimes"`
	Modes         []Mode    `json:"modes"`
	Profiles      []Profile `json:"profiles"`
}

type LaunchBlock struct {
	DockerArgs  []string          `json:"docker_args"`
	Environment map[string]string `json:"environment"`
	CommandArgs []string          `json:"command_args"`
}

type RuntimeLaunch struct {
	DockerArgs  []string          `json:"docker_args"`
	Environment map[string]string `json:"environment"`
	Command     []string          `json:"command"`
}

type Model struct {
	ID        string      `json:"id"`
	Name      string      `json:"name"`
	Kind      string      `json:"kind"`
	Repo      string      `json:"repo"`
	Revision  string      `json:"revision"`
	Gated     bool        `json:"gated"`
	Files     []ModelFile `json:"files"`
	MountPath string      `json:"mount_path"`
	Launch    LaunchBlock `json:"launch"`
}

type ModelFile struct {
	Path string `json:"path"`
	Size *int64 `json:"size,omitempty"`
}

type Runtime struct {
	ID            string        `json:"id"`
	Image         string        `json:"image"`
	Digest        string        `json:"digest"`
	Registry      string        `json:"registry,omitempty"`
	ContainerPort int           `json:"container_port"`
	HealthPath    string        `json:"health_path"`
	Launch        RuntimeLaunch `json:"launch"`
}

type Mode struct {
	ID          string      `json:"id"`
	DisplayName string      `json:"display_name"`
	Assistants  []string    `json:"assistants"`
	Launch      LaunchBlock `json:"launch"`
}

type Profile struct {
	ID             string      `json:"id"`
	ModelID        string      `json:"model_id"`
	RuntimeID      string      `json:"runtime_id"`
	Cards          int         `json:"cards"`
	TensorParallel int         `json:"tensor_parallel"`
	Context        int         `json:"context"`
	Mode           string      `json:"mode"`
	Launch         LaunchBlock `json:"launch"`
}

type manifestFile struct {
	SchemaVersion int            `json:"schema_version"`
	ID            string         `json:"id"`
	Name          string         `json:"name"`
	Version       string         `json:"version"`
	Models        *[]modelFile   `json:"models"`
	Runtimes      *[]runtimeFile `json:"runtimes"`
	Modes         *[]modeFile    `json:"modes"`
	Profiles      *[]profileFile `json:"profiles"`
}

type modelFile struct {
	ID        string           `json:"id"`
	Name      string           `json:"name"`
	Kind      string           `json:"kind"`
	Repo      string           `json:"repo"`
	Revision  string           `json:"revision"`
	Gated     *bool            `json:"gated"`
	Files     *[]ModelFile     `json:"files"`
	MountPath string           `json:"mount_path"`
	Launch    *launchBlockFile `json:"launch"`
}

type runtimeFile struct {
	ID            string             `json:"id"`
	Image         string             `json:"image"`
	Digest        string             `json:"digest"`
	Registry      *string            `json:"registry"`
	ContainerPort int                `json:"container_port"`
	HealthPath    string             `json:"health_path"`
	Launch        *runtimeLaunchFile `json:"launch"`
}

type modeFile struct {
	ID          string           `json:"id"`
	DisplayName string           `json:"display_name"`
	Assistants  *[]string        `json:"assistants"`
	Launch      *launchBlockFile `json:"launch"`
}

type profileFile struct {
	ID             string           `json:"id"`
	ModelID        string           `json:"model_id"`
	RuntimeID      string           `json:"runtime_id"`
	Cards          int              `json:"cards"`
	TensorParallel int              `json:"tensor_parallel"`
	Context        int              `json:"context"`
	Mode           string           `json:"mode"`
	Launch         *launchBlockFile `json:"launch"`
}

type launchBlockFile struct {
	DockerArgs  *[]string          `json:"docker_args"`
	Environment *map[string]string `json:"environment"`
	CommandArgs *[]string          `json:"command_args"`
}

type runtimeLaunchFile struct {
	DockerArgs  *[]string          `json:"docker_args"`
	Environment *map[string]string `json:"environment"`
	Command     *[]string          `json:"command"`
}

func Load(root string) (*Manifest, error) {
	file, err := os.Open(filepath.Join(root, "pack.json"))
	if err != nil {
		return nil, fmt.Errorf("open pack.json: %w", err)
	}
	defer file.Close()
	var source manifestFile
	decoder := json.NewDecoder(file)
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&source); err != nil {
		return nil, fmt.Errorf("decode pack.json: %w", err)
	}
	if err := ensureJSONEnd(decoder); err != nil {
		return nil, err
	}
	manifest, err := source.manifest()
	if err != nil {
		return nil, err
	}
	if err := manifest.validate(); err != nil {
		return nil, err
	}
	if err := requireRegularFile(filepath.Join(root, "README.md"), "pack README.md is required"); err != nil {
		return nil, err
	}
	return manifest, nil
}

func ensureJSONEnd(decoder *json.Decoder) error {
	var extra any
	if err := decoder.Decode(&extra); errors.Is(err, io.EOF) {
		return nil
	} else if err != nil {
		return fmt.Errorf("decode pack.json: %w", err)
	}
	return errors.New("pack.json must contain one JSON object")
}

func (source manifestFile) manifest() (*Manifest, error) {
	if source.Models == nil {
		return nil, errors.New("models list is required")
	}
	if source.Runtimes == nil {
		return nil, errors.New("runtimes list is required")
	}
	if source.Modes == nil {
		return nil, errors.New("modes list is required")
	}
	if source.Profiles == nil {
		return nil, errors.New("profiles list is required")
	}
	manifest := &Manifest{SchemaVersion: source.SchemaVersion, ID: source.ID, Name: source.Name, Version: source.Version}
	for _, item := range *source.Models {
		if item.Gated == nil {
			return nil, fmt.Errorf("model %q gated is required", item.ID)
		}
		if item.Files == nil {
			return nil, fmt.Errorf("model %q files list is required", item.ID)
		}
		launch, err := item.Launch.launchBlock("model " + item.ID)
		if err != nil {
			return nil, err
		}
		manifest.Models = append(manifest.Models, Model{ID: item.ID, Name: item.Name, Kind: item.Kind, Repo: item.Repo, Revision: item.Revision, Gated: *item.Gated, Files: *item.Files, MountPath: item.MountPath, Launch: launch})
	}
	for _, item := range *source.Runtimes {
		if item.Registry != nil && strings.TrimSpace(*item.Registry) == "" {
			return nil, fmt.Errorf("runtime %q registry is empty", item.ID)
		}
		launch, err := item.Launch.runtimeLaunch("runtime " + item.ID)
		if err != nil {
			return nil, err
		}
		value := Runtime{ID: item.ID, Image: item.Image, Digest: item.Digest, ContainerPort: item.ContainerPort, HealthPath: item.HealthPath, Launch: launch}
		if item.Registry != nil {
			value.Registry = *item.Registry
		}
		manifest.Runtimes = append(manifest.Runtimes, value)
	}
	for _, item := range *source.Modes {
		if item.Assistants == nil {
			return nil, fmt.Errorf("mode %q assistants list is required", item.ID)
		}
		launch, err := item.Launch.launchBlock("mode " + item.ID)
		if err != nil {
			return nil, err
		}
		manifest.Modes = append(manifest.Modes, Mode{ID: item.ID, DisplayName: item.DisplayName, Assistants: *item.Assistants, Launch: launch})
	}
	for _, item := range *source.Profiles {
		launch, err := item.Launch.launchBlock("profile " + item.ID)
		if err != nil {
			return nil, err
		}
		manifest.Profiles = append(manifest.Profiles, Profile{ID: item.ID, ModelID: item.ModelID, RuntimeID: item.RuntimeID, Cards: item.Cards, TensorParallel: item.TensorParallel, Context: item.Context, Mode: item.Mode, Launch: launch})
	}
	return manifest, nil
}

func (source *launchBlockFile) launchBlock(label string) (LaunchBlock, error) {
	if source == nil {
		return LaunchBlock{}, fmt.Errorf("%s launch is required", label)
	}
	if source.DockerArgs == nil {
		return LaunchBlock{}, fmt.Errorf("%s launch docker_args list is required", label)
	}
	if source.Environment == nil {
		return LaunchBlock{}, fmt.Errorf("%s launch environment object is required", label)
	}
	if source.CommandArgs == nil {
		return LaunchBlock{}, fmt.Errorf("%s launch command_args list is required", label)
	}
	return LaunchBlock{DockerArgs: *source.DockerArgs, Environment: *source.Environment, CommandArgs: *source.CommandArgs}, nil
}

func (source *runtimeLaunchFile) runtimeLaunch(label string) (RuntimeLaunch, error) {
	if source == nil {
		return RuntimeLaunch{}, fmt.Errorf("%s launch is required", label)
	}
	if source.DockerArgs == nil {
		return RuntimeLaunch{}, fmt.Errorf("%s launch docker_args list is required", label)
	}
	if source.Environment == nil {
		return RuntimeLaunch{}, fmt.Errorf("%s launch environment object is required", label)
	}
	if source.Command == nil {
		return RuntimeLaunch{}, fmt.Errorf("%s launch command list is required", label)
	}
	return RuntimeLaunch{DockerArgs: *source.DockerArgs, Environment: *source.Environment, Command: *source.Command}, nil
}

func requireRegularFile(filePath, missingMessage string) error {
	info, err := os.Stat(filePath)
	if err != nil {
		if os.IsNotExist(err) {
			return errors.New(missingMessage)
		}
		return err
	}
	if !info.Mode().IsRegular() {
		return errors.New(missingMessage)
	}
	return nil
}
