package runtime

import (
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"net/http"
	"strconv"
	"strings"
	"syscall"
	"time"
)

type State string

const (
	StateStopped  State = "Stopped"
	StateStarting State = "Starting"
	StateRunning  State = "Running"
	StateFailed   State = "Failed"
)

type StatusResult struct {
	State       State
	ExitCode    *int
	PackID      string
	PackVersion string
	ProfileID   string
	// Image is the exact image identity the managed container runs, used to
	// keep destructive image actions away from the active runtime.
	Image         string
	BindAddress   string
	HostPort      int
	ContainerPort int
}

type containerInfo struct {
	Config struct {
		Labels map[string]string `json:"Labels"`
	} `json:"Config"`
	Image string `json:"Image"`
	State struct {
		Running  bool `json:"Running"`
		ExitCode int  `json:"ExitCode"`
	} `json:"State"`
	NetworkSettings struct {
		Ports map[string][]struct {
			HostIP   string `json:"HostIp"`
			HostPort string `json:"HostPort"`
		} `json:"Ports"`
	} `json:"NetworkSettings"`
}

func Start(launch Launch) error {
	if err := validateLaunch(launch); err != nil {
		return err
	}
	if err := ensurePortAvailable(launch.BindAddress, launch.HostPort); err != nil {
		return err
	}

	container, exists, err := inspectManagedContainer()
	if err != nil {
		return err
	}
	if exists {
		if !container.managed() {
			return fmt.Errorf("container %s is not managed by B70 LLM Controller", ManagedContainer)
		}
		if container.State.Running {
			return errors.New("a model is already running")
		}
		if output, err := runDocker("rm", ManagedContainer); err != nil {
			return dockerError("remove stopped runtime container", output, err)
		}
	}
	if output, err := runDocker(launch.DockerArgs...); err != nil {
		return dockerError("start runtime container", output, err)
	}
	return nil
}

func Stop() error {
	container, exists, err := inspectManagedContainer()
	if err != nil {
		return err
	}
	if !exists {
		return nil
	}
	if !container.managed() {
		return fmt.Errorf("container %s is not managed by B70 LLM Controller", ManagedContainer)
	}
	if !container.State.Running {
		return nil
	}
	if output, err := runDocker("stop", "--time", "30", ManagedContainer); err != nil {
		return dockerError("stop runtime container", output, err)
	}
	return nil
}

func Logs(tail int) (string, error) {
	if tail <= 0 {
		return "", errors.New("log tail must be positive")
	}
	output, err := runDocker("logs", "--tail", strconv.Itoa(tail), ManagedContainer)
	if err != nil {
		return "", dockerError("read runtime logs", output, err)
	}
	return string(output), nil
}

func Status() (StatusResult, error) {
	container, exists, err := inspectManagedContainer()
	if err != nil {
		return StatusResult{}, err
	}
	if !exists {
		return StatusResult{State: StateStopped}, nil
	}
	if !container.managed() {
		return StatusResult{}, fmt.Errorf("container %s is not managed by B70 LLM Controller", ManagedContainer)
	}

	result := StatusResult{
		PackID:      container.Config.Labels["b70ctl.pack_id"],
		PackVersion: container.Config.Labels["b70ctl.pack_version"],
		ProfileID:   container.Config.Labels["b70ctl.profile_id"],
		Image:       container.Image,
	}
	bindAddress, hostPort, containerPort, err := container.publishedBinding()
	if err != nil && container.State.Running {
		return StatusResult{}, err
	}
	if err == nil {
		result.BindAddress = bindAddress
		result.HostPort = hostPort
		result.ContainerPort = containerPort
	}
	if !container.State.Running {
		if container.State.ExitCode != 0 {
			exitCode := container.State.ExitCode
			result.State = StateFailed
			result.ExitCode = &exitCode
			return result, nil
		}
		result.State = StateStopped
		return result, nil
	}
	healthPath := container.Config.Labels["b70ctl.health_path"]
	if !strings.HasPrefix(healthPath, "/") {
		return StatusResult{}, errors.New("managed runtime has no valid health path")
	}
	if healthy(hostPort, healthPath) {
		result.State = StateRunning
	} else {
		result.State = StateStarting
	}
	return result, nil
}

func inspectManagedContainer() (containerInfo, bool, error) {
	output, err := runDocker("inspect", ManagedContainer)
	if err != nil {
		if containerMissing(output) {
			return containerInfo{}, false, nil
		}
		return containerInfo{}, false, dockerError("inspect runtime container", output, err)
	}

	var containers []containerInfo
	if err := json.Unmarshal(output, &containers); err != nil || len(containers) != 1 {
		return containerInfo{}, false, errors.New("Docker container inspect returned invalid JSON")
	}
	return containers[0], true, nil
}

func (container containerInfo) managed() bool {
	return container.Config.Labels["b70ctl.managed"] == "true"
}

func (container containerInfo) publishedBinding() (string, int, int, error) {
	if len(container.NetworkSettings.Ports) != 1 {
		return "", 0, 0, errors.New("managed runtime has no published host port")
	}
	for portKey, bindings := range container.NetworkSettings.Ports {
		if len(bindings) != 1 || !strings.HasSuffix(portKey, "/tcp") {
			return "", 0, 0, errors.New("managed runtime has no published host port")
		}
		containerPort, err := strconv.Atoi(strings.TrimSuffix(portKey, "/tcp"))
		if err != nil || containerPort < 1 || containerPort > 65535 {
			return "", 0, 0, errors.New("managed runtime has an invalid container port")
		}
		hostPort, err := strconv.Atoi(bindings[0].HostPort)
		if err != nil || hostPort < 1 || hostPort > 65535 {
			return "", 0, 0, errors.New("managed runtime has an invalid published host port")
		}
		if bindings[0].HostIP != "127.0.0.1" && bindings[0].HostIP != "0.0.0.0" {
			return "", 0, 0, errors.New("managed runtime has an invalid published host address")
		}
		return bindings[0].HostIP, hostPort, containerPort, nil
	}
	return "", 0, 0, errors.New("managed runtime has no published host port")
}

func healthy(hostPort int, healthPath string) bool {
	client := http.Client{Timeout: 2 * time.Second}
	response, err := client.Get(fmt.Sprintf("http://127.0.0.1:%d%s", hostPort, healthPath))
	if err != nil {
		return false
	}
	defer response.Body.Close()
	return response.StatusCode >= 200 && response.StatusCode < 300
}

func validateLaunch(launch Launch) error {
	if err := validateHostPort(launch.HostPort); err != nil {
		return err
	}
	if launch.BindAddress != "127.0.0.1" && launch.BindAddress != "0.0.0.0" {
		return errors.New("launch bind address is invalid")
	}
	if len(launch.DockerArgs) == 0 || launch.DockerArgs[0] != "run" {
		return errors.New("Docker launch arguments are required")
	}
	return nil
}

func ensurePortAvailable(host string, port int) error {
	listener, err := net.Listen("tcp", net.JoinHostPort(host, strconv.Itoa(port)))
	if err != nil {
		if errors.Is(err, syscall.EADDRINUSE) {
			return fmt.Errorf("port %d is already in use", port)
		}
		return fmt.Errorf("check port %d: %w", port, err)
	}
	return listener.Close()
}

func containerMissing(output []byte) bool {
	message := strings.ToLower(string(output))
	return strings.Contains(message, "no such object:") || strings.Contains(message, "no such container:")
}

func dockerError(action string, output []byte, err error) error {
	detail := strings.TrimSpace(string(output))
	if detail == "" {
		return fmt.Errorf("%s: %w", action, err)
	}
	return fmt.Errorf("%s: %s", action, detail)
}
