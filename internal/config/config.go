package config

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
)

const (
	AccessLocal = "local"
	AccessLAN   = "lan"
)

type Config struct {
	ModelDirectory string        `json:"model_directory"`
	DefaultAccess  string        `json:"default_access"`
	DefaultPort    int           `json:"default_port"`
	LastRun        *RunSelection `json:"last_run,omitempty"`
}

// RunSelection records the last model, cards, context, and mode chosen on the
// Run Model screen. Values are stable identifiers resolved against installed
// packs when the screen is opened again; unresolvable values fall back to the
// normal defaults, so they are deliberately not constrained by Validate.
type RunSelection struct {
	ModelID string `json:"model_id"`
	Cards   int    `json:"cards"`
	Context int    `json:"context"`
	Mode    string `json:"mode"`
}

type Paths struct {
	ConfigFile string
	DataRoot   string
	PacksRoot  string
}

func Default() (Config, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return Config{}, fmt.Errorf("find home directory: %w", err)
	}
	return Config{
		ModelDirectory: filepath.Join(home, "b70ctl-models"),
		DefaultAccess:  AccessLocal,
		DefaultPort:    8000,
	}, nil
}

func ResolvePaths() (Paths, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return Paths{}, fmt.Errorf("find home directory: %w", err)
	}
	configRoot := os.Getenv("XDG_CONFIG_HOME")
	if configRoot == "" {
		configRoot = filepath.Join(home, ".config")
	}
	dataRoot := os.Getenv("XDG_DATA_HOME")
	if dataRoot == "" {
		dataRoot = filepath.Join(home, ".local", "share")
	}
	appData := filepath.Join(dataRoot, "b70ctl")
	return Paths{
		ConfigFile: filepath.Join(configRoot, "b70ctl", "config.json"),
		DataRoot:   appData,
		PacksRoot:  filepath.Join(appData, "packs"),
	}, nil
}

func Load(path string) (Config, error) {
	defaults, err := Default()
	if err != nil {
		return Config{}, err
	}
	file, err := os.Open(path)
	if os.IsNotExist(err) {
		return defaults, nil
	}
	if err != nil {
		return Config{}, fmt.Errorf("open config: %w", err)
	}
	defer file.Close()

	var value Config
	decoder := json.NewDecoder(file)
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&value); err != nil {
		return Config{}, fmt.Errorf("decode config: %w", err)
	}
	var extra any
	if err := decoder.Decode(&extra); !errors.Is(err, io.EOF) {
		if err != nil {
			return Config{}, fmt.Errorf("decode config: %w", err)
		}
		return Config{}, errors.New("config must contain one JSON object")
	}
	if err := value.Validate(); err != nil {
		return Config{}, err
	}
	value.ModelDirectory = filepath.Clean(value.ModelDirectory)
	return value, nil
}

func Save(path string, value Config) error {
	if err := value.Validate(); err != nil {
		return err
	}
	value.ModelDirectory = filepath.Clean(value.ModelDirectory)
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Errorf("create config directory: %w", err)
	}
	file, err := os.CreateTemp(filepath.Dir(path), ".config-*.tmp")
	if err != nil {
		return fmt.Errorf("create temporary config: %w", err)
	}
	temporary := file.Name()
	defer os.Remove(temporary)

	if err := file.Chmod(0o644); err != nil {
		file.Close()
		return fmt.Errorf("set config permissions: %w", err)
	}
	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(value); err != nil {
		file.Close()
		return fmt.Errorf("write config: %w", err)
	}
	if err := file.Sync(); err != nil {
		file.Close()
		return fmt.Errorf("sync config: %w", err)
	}
	if err := file.Close(); err != nil {
		return fmt.Errorf("close config: %w", err)
	}
	if err := os.Rename(temporary, path); err != nil {
		return fmt.Errorf("save config: %w", err)
	}
	return nil
}

func (value Config) Validate() error {
	if !filepath.IsAbs(value.ModelDirectory) {
		return errors.New("model directory must be an absolute path")
	}
	if value.DefaultAccess != AccessLocal && value.DefaultAccess != AccessLAN {
		return errors.New("default access must be local or lan")
	}
	if value.DefaultPort < 1024 || value.DefaultPort > 65535 {
		return errors.New("default port must be between 1024 and 65535")
	}
	return nil
}
