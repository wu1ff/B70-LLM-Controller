package runtime

import (
	"errors"
	"fmt"
	"path/filepath"
	"sort"

	"b70ctl/internal/modelpack"
	"b70ctl/internal/modelstore"
)

const ManagedContainer = "b70ctl-runtime"

type Access string

const (
	AccessLocal Access = "local"
	AccessLAN   Access = "lan"
)

type Launch struct {
	DockerArgs    []string
	BindAddress   string
	HostPort      int
	ContainerPort int
	HealthPath    string
}

func BuildLaunch(manifest *modelpack.Manifest, profile modelpack.Profile, models []modelstore.Artifact, access Access, hostPort int) (Launch, error) {
	if manifest == nil {
		return Launch{}, errors.New("manifest is required")
	}
	bindAddress, err := bindAddress(access)
	if err != nil {
		return Launch{}, err
	}
	if err := validateHostPort(hostPort); err != nil {
		return Launch{}, err
	}
	resolved, err := modelpack.Resolve(manifest, profile)
	if err != nil {
		return Launch{}, err
	}
	modelsByID := make(map[string]modelpack.Model, len(manifest.Models))
	for _, model := range manifest.Models {
		modelsByID[model.ID] = model
	}
	args := []string{
		"run", "-d", "--name", ManagedContainer,
		"--publish", fmt.Sprintf("%s:%d:%d", bindAddress, hostPort, resolved.ContainerPort),
		"--label", "b70ctl.managed=true",
		"--label", "b70ctl.pack_id=" + manifest.ID,
		"--label", "b70ctl.pack_version=" + manifest.Version,
		"--label", "b70ctl.profile_id=" + profile.ID,
		"--label", "b70ctl.health_path=" + resolved.HealthPath,
	}
	args = append(args, resolved.DockerArgs...)
	for _, mount := range resolved.Mounts {
		model, exists := modelsByID[mount.ModelID]
		if !exists {
			return Launch{}, fmt.Errorf("model %q is not declared", mount.ModelID)
		}
		artifact, exists := modelstore.Find(models, model.Repo, model.Revision)
		if !exists {
			return Launch{}, fmt.Errorf("model %q is not discovered", mount.ModelID)
		}
		if !modelstore.Check(artifact.Path, model.Files).Complete {
			return Launch{}, fmt.Errorf("model %q is incomplete", mount.ModelID)
		}
		if !filepath.IsAbs(artifact.Path) {
			return Launch{}, fmt.Errorf("model %q path is not absolute", mount.ModelID)
		}
		args = append(args, "--mount", fmt.Sprintf("type=bind,source=%s,target=%s,readonly", artifact.Path, mount.ContainerPath))
	}
	environmentNames := make([]string, 0, len(resolved.Environment))
	for name := range resolved.Environment {
		environmentNames = append(environmentNames, name)
	}
	sort.Strings(environmentNames)
	for _, name := range environmentNames {
		args = append(args, "--env", name+"="+resolved.Environment[name])
	}
	target, exists := modelsByID[profile.ModelID]
	if !exists {
		return Launch{}, fmt.Errorf("model %q is not declared", profile.ModelID)
	}
	serveCommand, err := modelpack.ServeCommand(resolved, target.MountPath)
	if err != nil {
		return Launch{}, err
	}
	args = append(args, resolved.Digest)
	args = append(args, serveCommand...)
	return Launch{DockerArgs: args, BindAddress: bindAddress, HostPort: hostPort, ContainerPort: resolved.ContainerPort, HealthPath: resolved.HealthPath}, nil
}

func bindAddress(access Access) (string, error) {
	switch access {
	case AccessLocal:
		return "127.0.0.1", nil
	case AccessLAN:
		return "0.0.0.0", nil
	default:
		return "", fmt.Errorf("access must be %q or %q", AccessLocal, AccessLAN)
	}
}

func validateHostPort(port int) error {
	if port < 1024 || port > 65535 {
		return errors.New("host port must be between 1024 and 65535")
	}
	return nil
}
