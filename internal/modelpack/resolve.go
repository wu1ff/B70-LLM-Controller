package modelpack

import "fmt"

type Mount struct {
	ModelID       string `json:"model_id"`
	ContainerPath string `json:"container_path"`
}

type Resolved struct {
	ContainerPort int
	HealthPath    string
	DockerArgs    []string
	Environment   map[string]string
	Mounts        []Mount
	Image         string
	Digest        string
	Command       []string
}

func Resolve(manifest *Manifest, profile Profile) (Resolved, error) {
	models := make(map[string]Model, len(manifest.Models))
	for _, model := range manifest.Models {
		models[model.ID] = model
	}
	runtimes := make(map[string]Runtime, len(manifest.Runtimes))
	for _, runtime := range manifest.Runtimes {
		runtimes[runtime.ID] = runtime
	}
	modes := make(map[string]Mode, len(manifest.Modes))
	for _, mode := range manifest.Modes {
		modes[mode.ID] = mode
	}
	model, exists := models[profile.ModelID]
	if !exists {
		return Resolved{}, fmt.Errorf("model %q is not declared", profile.ModelID)
	}
	runtime, exists := runtimes[profile.RuntimeID]
	if !exists {
		return Resolved{}, fmt.Errorf("runtime %q is not declared", profile.RuntimeID)
	}
	mode, exists := modes[profile.Mode]
	if !exists {
		return Resolved{}, fmt.Errorf("mode %q is not declared", profile.Mode)
	}
	layers := []LaunchBlock{{DockerArgs: runtime.Launch.DockerArgs, Environment: runtime.Launch.Environment}, model.Launch, mode.Launch, profile.Launch}
	resolved := Resolved{ContainerPort: runtime.ContainerPort, HealthPath: runtime.HealthPath, Environment: map[string]string{}, Image: runtime.Image, Digest: runtime.Digest, Command: append([]string(nil), runtime.Launch.Command...)}
	for _, layer := range layers {
		resolved.DockerArgs = append(resolved.DockerArgs, layer.DockerArgs...)
		for name, value := range layer.Environment {
			if _, duplicate := resolved.Environment[name]; duplicate {
				return Resolved{}, fmt.Errorf("duplicate environment key %q", name)
			}
			resolved.Environment[name] = value
		}
	}
	resolved.Command = append(resolved.Command, model.Launch.CommandArgs...)
	resolved.Command = append(resolved.Command, mode.Launch.CommandArgs...)
	resolved.Command = append(resolved.Command, profile.Launch.CommandArgs...)
	resolved.Mounts = append(resolved.Mounts, Mount{ModelID: model.ID, ContainerPath: model.MountPath})
	for _, assistantID := range mode.Assistants {
		assistant, exists := models[assistantID]
		if !exists {
			return Resolved{}, fmt.Errorf("assistant %q is not declared", assistantID)
		}
		resolved.Mounts = append(resolved.Mounts, Mount{ModelID: assistant.ID, ContainerPath: assistant.MountPath})
	}
	paths := map[string]string{}
	for _, mount := range resolved.Mounts {
		if existing, conflict := paths[mount.ContainerPath]; conflict {
			return Resolved{}, fmt.Errorf("models %q and %q conflict at mount path %q", existing, mount.ModelID, mount.ContainerPath)
		}
		paths[mount.ContainerPath] = mount.ModelID
	}
	return resolved, nil
}

// ServeCommand adapts a resolved contract into the vLLM CLI grammar:
// vLLM accepts `vllm serve <model> [options...]` only, so the target
// model's mount path must be the first positional argument, directly
// after "serve", with every layer's option tokens following it.
func ServeCommand(resolved Resolved, modelPath string) ([]string, error) {
	command := resolved.Command
	if len(command) < 2 || command[0] != "vllm" || command[1] != "serve" {
		return nil, fmt.Errorf("resolved command must begin with %q", "vllm serve")
	}
	if modelPath == "" {
		return nil, fmt.Errorf("model path is required")
	}
	positional := -1
	for index := 2; index < len(command); index++ {
		if command[index] == modelPath {
			if positional >= 0 {
				return nil, fmt.Errorf("model path %q appears more than once", modelPath)
			}
			positional = index
		}
	}
	if positional < 0 {
		return nil, fmt.Errorf("model path %q is missing from the resolved command", modelPath)
	}
	serve := make([]string, 0, len(command))
	serve = append(serve, "vllm", "serve", modelPath)
	serve = append(serve, command[2:positional]...)
	serve = append(serve, command[positional+1:]...)
	return serve, nil
}
