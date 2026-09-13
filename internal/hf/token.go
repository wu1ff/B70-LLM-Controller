package hf

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

const (
	SourceEnvironment = "HF_TOKEN"
	SourceFile        = "Controller token file"
)

func Configured() (bool, string, error) {
	token, source, err := Get()
	return token != "", source, err
}

func Get() (string, string, error) {
	if token := strings.TrimSpace(os.Getenv("HF_TOKEN")); token != "" {
		return token, SourceEnvironment, nil
	}
	path, err := tokenPath()
	if err != nil {
		return "", "", err
	}
	contents, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return "", "", nil
	}
	if err != nil {
		return "", "", fmt.Errorf("read Hugging Face token: %w", err)
	}
	token := strings.TrimSpace(string(contents))
	if token == "" {
		return "", "", nil
	}
	return token, SourceFile, nil
}

func Set(value string) error {
	value = strings.TrimSpace(value)
	if value == "" {
		return errors.New("token is required")
	}
	if strings.ContainsAny(value, "\r\n") {
		return errors.New("token must be one line")
	}
	path, err := tokenPath()
	if err != nil {
		return err
	}
	directory := filepath.Dir(path)
	if err := os.MkdirAll(directory, 0o700); err != nil {
		return fmt.Errorf("create Controller config directory: %w", err)
	}
	file, err := os.CreateTemp(directory, ".hf-token-*.tmp")
	if err != nil {
		return fmt.Errorf("create temporary token file: %w", err)
	}
	temporary := file.Name()
	defer os.Remove(temporary)

	if err := file.Chmod(0o600); err != nil {
		file.Close()
		return fmt.Errorf("set token permissions: %w", err)
	}
	if _, err := file.WriteString(value + "\n"); err != nil {
		file.Close()
		return fmt.Errorf("write Hugging Face token: %w", err)
	}
	if err := file.Sync(); err != nil {
		file.Close()
		return fmt.Errorf("sync Hugging Face token: %w", err)
	}
	if err := file.Close(); err != nil {
		return fmt.Errorf("close Hugging Face token: %w", err)
	}
	if err := os.Rename(temporary, path); err != nil {
		return fmt.Errorf("save Hugging Face token: %w", err)
	}
	if err := os.Chmod(path, 0o600); err != nil {
		return fmt.Errorf("set token permissions: %w", err)
	}
	return nil
}

func Clear() error {
	path, err := tokenPath()
	if err != nil {
		return err
	}
	if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("clear Hugging Face token: %w", err)
	}
	return nil
}

func tokenPath() (string, error) {
	root := os.Getenv("XDG_CONFIG_HOME")
	if root == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			return "", fmt.Errorf("find home directory: %w", err)
		}
		root = filepath.Join(home, ".config")
	}
	return filepath.Join(root, "b70ctl", "hf_token"), nil
}
