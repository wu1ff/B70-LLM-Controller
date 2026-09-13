package runtime

import (
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strconv"
	"strings"
	"testing"
)

func TestStartRefusesRunningManagedContainer(t *testing.T) {
	port := unusedPort(t)
	setDockerHook(t, func(args ...string) ([]byte, error) {
		if reflect.DeepEqual(args, []string{"inspect", ManagedContainer}) {
			return inspectJSON(true, 0, port), nil
		}
		t.Fatalf("unexpected Docker call: %v", args)
		return nil, nil
	})

	err := Start(testLaunch(port))
	if err == nil || err.Error() != "a model is already running" {
		t.Fatalf("Start() error = %v", err)
	}
}

func TestStartRemovesStoppedManagedContainerBeforeRun(t *testing.T) {
	port := unusedPort(t)
	var calls [][]string
	setDockerHook(t, func(args ...string) ([]byte, error) {
		calls = append(calls, append([]string(nil), args...))
		switch args[0] {
		case "inspect":
			return inspectJSON(false, 0, port), nil
		case "rm", "run":
			return []byte("ok"), nil
		default:
			t.Fatalf("unexpected Docker call: %v", args)
			return nil, nil
		}
	})

	if err := Start(testLaunch(port)); err != nil {
		t.Fatalf("Start() error = %v", err)
	}
	want := [][]string{
		{"inspect", ManagedContainer},
		{"rm", ManagedContainer},
		{"run", "-d", "--name", ManagedContainer, "image-id", "command"},
	}
	if !reflect.DeepEqual(calls, want) {
		t.Fatalf("Docker calls = %#v, want %#v", calls, want)
	}
}

func TestStartRejectsOccupiedPortBeforeDocker(t *testing.T) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer listener.Close()
	port := listener.Addr().(*net.TCPAddr).Port
	called := false
	setDockerHook(t, func(...string) ([]byte, error) {
		called = true
		return nil, nil
	})

	err = Start(testLaunch(port))
	if err == nil || err.Error() != fmt.Sprintf("port %d is already in use", port) {
		t.Fatalf("Start() error = %v", err)
	}
	if called {
		t.Fatal("Start() called Docker for an occupied port")
	}
}

func TestStopTargetsOnlyManagedContainer(t *testing.T) {
	var calls [][]string
	setDockerHook(t, func(args ...string) ([]byte, error) {
		calls = append(calls, append([]string(nil), args...))
		if args[0] == "inspect" {
			return inspectJSON(true, 0, 8000), nil
		}
		return []byte("stopped"), nil
	})

	if err := Stop(); err != nil {
		t.Fatalf("Stop() error = %v", err)
	}
	want := [][]string{
		{"inspect", ManagedContainer},
		{"stop", "--time", "30", ManagedContainer},
	}
	if !reflect.DeepEqual(calls, want) {
		t.Fatalf("Docker calls = %#v, want %#v", calls, want)
	}
}

func TestStopMissingContainerIsHarmless(t *testing.T) {
	setDockerHook(t, func(args ...string) ([]byte, error) {
		if !reflect.DeepEqual(args, []string{"inspect", ManagedContainer}) {
			t.Fatalf("unexpected Docker call: %v", args)
		}
		return []byte("[]\nerror: no such object: " + ManagedContainer), errors.New("exit status 1")
	})

	if err := Stop(); err != nil {
		t.Fatalf("Stop() error = %v", err)
	}
}

func TestLogsUsesRequestedTail(t *testing.T) {
	setDockerHook(t, func(args ...string) ([]byte, error) {
		want := []string{"logs", "--tail", "100", ManagedContainer}
		if !reflect.DeepEqual(args, want) {
			t.Fatalf("Docker call = %v, want %v", args, want)
		}
		return []byte("line one\nline two\n"), nil
	})

	logs, err := Logs(100)
	if err != nil || logs != "line one\nline two\n" {
		t.Fatalf("Logs() = %q, %v", logs, err)
	}
}

func TestStatusRunningHealthSemantics(t *testing.T) {
	tests := []struct {
		name       string
		statusCode int
		want       State
	}{
		{name: "healthy is running", statusCode: http.StatusNoContent, want: StateRunning},
		{name: "unhealthy is starting", statusCode: http.StatusServiceUnavailable, want: StateStarting},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
				if request.URL.Path != "/health" {
					t.Fatalf("health path = %q", request.URL.Path)
				}
				writer.WriteHeader(test.statusCode)
			}))
			defer server.Close()
			port := serverPort(t, server)
			setDockerHook(t, func(...string) ([]byte, error) {
				return inspectJSON(true, 0, port), nil
			})

			status, err := Status()
			if err != nil || status.State != test.want || status.BindAddress != "127.0.0.1" || status.HostPort != port || status.ContainerPort != 8000 {
				t.Fatalf("Status() = %#v, %v; want %s", status, err, test.want)
			}
		})
	}
}

func TestStatusUnavailableHealthEndpointIsStarting(t *testing.T) {
	port := unusedPort(t)
	setDockerHook(t, func(...string) ([]byte, error) {
		return inspectJSON(true, 0, port), nil
	})

	status, err := Status()
	if err != nil || status.State != StateStarting {
		t.Fatalf("Status() = %#v, %v", status, err)
	}
}

func TestStatusExitedAndMissingSemantics(t *testing.T) {
	t.Run("nonzero exit is failed", func(t *testing.T) {
		setDockerHook(t, func(...string) ([]byte, error) {
			return inspectJSON(false, 17, 8000), nil
		})
		status, err := Status()
		if err != nil || status.State != StateFailed || status.ExitCode == nil || *status.ExitCode != 17 {
			t.Fatalf("Status() = %#v, %v", status, err)
		}
	})

	t.Run("missing is stopped", func(t *testing.T) {
		setDockerHook(t, func(...string) ([]byte, error) {
			return []byte("Error: No such container: " + ManagedContainer), errors.New("exit status 1")
		})
		status, err := Status()
		if err != nil || status.State != StateStopped || status.ExitCode != nil {
			t.Fatalf("Status() = %#v, %v", status, err)
		}
	})
}

func setDockerHook(t *testing.T, hook func(...string) ([]byte, error)) {
	t.Helper()
	original := runDocker
	runDocker = hook
	t.Cleanup(func() { runDocker = original })
}

func testLaunch(port int) Launch {
	return Launch{
		DockerArgs:    []string{"run", "-d", "--name", ManagedContainer, "image-id", "command"},
		BindAddress:   "127.0.0.1",
		HostPort:      port,
		ContainerPort: 8000,
		HealthPath:    "/health",
	}
}

func inspectJSON(running bool, exitCode, hostPort int) []byte {
	return []byte(fmt.Sprintf(`[{
  "Config":{"Labels":{"b70ctl.managed":"true","b70ctl.pack_id":"pack","b70ctl.pack_version":"1.0.0","b70ctl.profile_id":"profile","b70ctl.health_path":"/health"}},
  "State":{"Running":%t,"ExitCode":%d},
  "NetworkSettings":{"Ports":{"8000/tcp":[{"HostIp":"127.0.0.1","HostPort":"%d"}]}}
}]`, running, exitCode, hostPort))
}

func unusedPort(t *testing.T) int {
	t.Helper()
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	port := listener.Addr().(*net.TCPAddr).Port
	if err := listener.Close(); err != nil {
		t.Fatal(err)
	}
	return port
}

func serverPort(t *testing.T, server *httptest.Server) int {
	t.Helper()
	hostPort := strings.TrimPrefix(server.URL, "http://")
	_, portText, err := net.SplitHostPort(hostPort)
	if err != nil {
		t.Fatal(err)
	}
	port, err := strconv.Atoi(portText)
	if err != nil {
		t.Fatal(err)
	}
	return port
}
