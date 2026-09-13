package runtime

import (
	"context"
	"errors"
	"os/exec"
	"reflect"
	"strings"
	"testing"

	"b70ctl/internal/modelpack"
)

func TestAcquireReusesExactImageWithoutPull(t *testing.T) {
	declared := testAcquisitionRuntime()
	setDockerHook(t, func(args ...string) ([]byte, error) {
		return []byte(`"` + declared.Digest + `"`), nil
	})
	pulled := false
	setDockerPullHook(t, func(context.Context, string, func(PullProgress)) ([]byte, error) {
		pulled = true
		return nil, nil
	})

	result := Acquire(context.Background(), declared, nil)
	if result.Outcome != AcquisitionReused || result.Err != nil || pulled {
		t.Fatalf("Acquire() = %#v, pulled = %v", result, pulled)
	}
}

func TestAcquirePullsExactImmutableReferenceAndVerifiesDownloadedImage(t *testing.T) {
	declared := testAcquisitionRuntime()
	var inspected []string
	setDockerHook(t, func(args ...string) ([]byte, error) {
		reference := args[len(args)-1]
		inspected = append(inspected, reference)
		if reference == declared.Registry {
			return []byte(`"` + declared.Digest + `"`), nil
		}
		return missingImage(reference)
	})
	var pulled string
	var progresses []PullProgress
	setDockerPullHook(t, func(_ context.Context, reference string, notify func(PullProgress)) ([]byte, error) {
		pulled = reference
		notify(PullProgress{Status: "Pulling", Activity: "Downloading layers", Latest: "layer: Download complete"})
		return nil, nil
	})

	result := Acquire(context.Background(), declared, func(progress PullProgress) {
		progresses = append(progresses, progress)
	})
	if result.Outcome != AcquisitionDownloaded || result.Err != nil {
		t.Fatalf("Acquire() = %#v", result)
	}
	if pulled != declared.Registry {
		t.Fatalf("pulled reference = %q", pulled)
	}
	if !reflect.DeepEqual(inspected, []string{declared.Image, declared.Digest, declared.Registry}) {
		t.Fatalf("inspected references = %#v", inspected)
	}
	if len(progresses) < 2 || progresses[len(progresses)-1].Latest == "" {
		t.Fatalf("progresses = %#v", progresses)
	}
}

func TestAcquireMismatchedLocalTagUsesImmutableRegistryAuthority(t *testing.T) {
	declared := testAcquisitionRuntime()
	setDockerHook(t, func(args ...string) ([]byte, error) {
		reference := args[len(args)-1]
		switch reference {
		case declared.Image:
			return []byte(`"sha256:` + strings.Repeat("b", 64) + `"`), nil
		case declared.Digest:
			return missingImage(reference)
		default:
			return []byte(`"` + declared.Digest + `"`), nil
		}
	})
	var pulled string
	setDockerPullHook(t, func(_ context.Context, reference string, _ func(PullProgress)) ([]byte, error) {
		pulled = reference
		return nil, nil
	})

	result := Acquire(context.Background(), declared, nil)
	if result.Outcome != AcquisitionDownloaded || pulled != declared.Registry {
		t.Fatalf("Acquire() = %#v, pulled = %q", result, pulled)
	}
}

func TestAcquireRejectsDownloadedIdentityMismatch(t *testing.T) {
	declared := testAcquisitionRuntime()
	setDockerHook(t, func(args ...string) ([]byte, error) {
		reference := args[len(args)-1]
		if reference == declared.Registry {
			return []byte(`"sha256:` + strings.Repeat("b", 64) + `"`), nil
		}
		return missingImage(reference)
	})
	setDockerPullHook(t, func(context.Context, string, func(PullProgress)) ([]byte, error) {
		return nil, nil
	})

	result := Acquire(context.Background(), declared, nil)
	if result.Outcome != AcquisitionFailed || result.Reason != "runtime image identity mismatch" {
		t.Fatalf("Acquire() = %#v", result)
	}
}

func TestAcquireReportsPullFailureAndDockerUnavailable(t *testing.T) {
	t.Run("pull failure", func(t *testing.T) {
		declared := testAcquisitionRuntime()
		setDockerHook(t, func(args ...string) ([]byte, error) {
			return missingImage(args[len(args)-1])
		})
		setDockerPullHook(t, func(context.Context, string, func(PullProgress)) ([]byte, error) {
			return []byte("registry unavailable"), errors.New("exit status 1")
		})
		result := Acquire(context.Background(), declared, nil)
		if result.Outcome != AcquisitionFailed || result.Reason != "Docker pull failed" {
			t.Fatalf("Acquire() = %#v", result)
		}
	})

	t.Run("Docker executable missing", func(t *testing.T) {
		declared := testAcquisitionRuntime()
		setDockerHook(t, func(...string) ([]byte, error) { return nil, exec.ErrNotFound })
		result := Acquire(context.Background(), declared, nil)
		if result.Outcome != AcquisitionFailed || result.Reason != "Docker unavailable" {
			t.Fatalf("Acquire() = %#v", result)
		}
	})
}

func TestAcquireNeverPullsMutableRegistryAuthority(t *testing.T) {
	declared := testAcquisitionRuntime()
	declared.Registry = "ghcr.io/example/runtime:latest"
	setDockerHook(t, func(args ...string) ([]byte, error) {
		return missingImage(args[len(args)-1])
	})
	pulled := false
	setDockerPullHook(t, func(context.Context, string, func(PullProgress)) ([]byte, error) {
		pulled = true
		return nil, nil
	})

	result := Acquire(context.Background(), declared, nil)
	if result.Outcome != AcquisitionFailed || pulled {
		t.Fatalf("Acquire() = %#v, pulled = %v", result, pulled)
	}
}

func TestDockerProgressWriterSanitizesAndClassifiesOutput(t *testing.T) {
	var progresses []PullProgress
	writer := newDockerProgressWriter(func(progress PullProgress) {
		progresses = append(progresses, progress)
	})
	_, _ = writer.Write([]byte("abc: Download complete\nAuthorization: secret\nxyz: Extracting\r"))
	writer.flush()
	if len(progresses) != 2 {
		t.Fatalf("progresses = %#v", progresses)
	}
	if progresses[0].Activity != "Downloading layers" || progresses[1].Activity != "Extracting layers" {
		t.Fatalf("activities = %#v", progresses)
	}
	if strings.Contains(strings.ToLower(writer.summary()), "secret") {
		t.Fatalf("summary exposed sensitive output: %q", writer.summary())
	}
}

func testAcquisitionRuntime() modelpack.Runtime {
	digest := "sha256:" + strings.Repeat("a", 64)
	return modelpack.Runtime{
		ID: "runtime", Image: "local/runtime:latest", Digest: digest,
		Registry: "ghcr.io/example/runtime@" + digest,
	}
}

func missingImage(reference string) ([]byte, error) {
	return []byte("Error response from daemon: No such image: " + reference), errors.New("exit status 1")
}

func setDockerPullHook(t *testing.T, hook func(context.Context, string, func(PullProgress)) ([]byte, error)) {
	t.Helper()
	original := runDockerPull
	runDockerPull = hook
	t.Cleanup(func() { runDockerPull = original })
}
