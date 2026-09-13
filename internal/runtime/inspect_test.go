package runtime

import (
	"errors"
	"strings"
	"testing"

	"b70ctl/internal/modelpack"
)

func TestInspectImage(t *testing.T) {
	digest := "sha256:" + strings.Repeat("a", 64)
	tests := []struct {
		name       string
		output     string
		err        error
		wantStatus ImageStatus
		wantError  bool
	}{
		{
			name:       "exact identity is present",
			output:     `"` + digest + `"`,
			wantStatus: ImagePresent,
		},
		{
			name:       "image not found is missing",
			output:     "Error response from daemon: No such image: example/runtime:latest\n",
			err:        errors.New("exit status 1"),
			wantStatus: ImageMissing,
		},
		{
			name:       "tag resolving to another identity is missing",
			output:     `"sha256:` + strings.Repeat("b", 64) + `"`,
			wantStatus: ImageMissing,
		},
		{
			name:       "unexpected command failure is an error",
			output:     "Cannot connect to the Docker daemon\n",
			err:        errors.New("exit status 1"),
			wantStatus: ImageError,
			wantError:  true,
		},
		{
			name:       "malformed inspect output is an error",
			output:     `"not-an-image-id"`,
			wantStatus: ImageError,
			wantError:  true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			original := runDocker
			runDocker = func(...string) ([]byte, error) {
				return []byte(test.output), test.err
			}
			t.Cleanup(func() { runDocker = original })

			result := InspectImage("example/runtime:latest", digest)
			if result.Status != test.wantStatus {
				t.Fatalf("InspectImage() status = %v, want %v", result.Status, test.wantStatus)
			}
			if (result.Err != nil) != test.wantError {
				t.Fatalf("InspectImage() error = %v, want error %v", result.Err, test.wantError)
			}
		})
	}
}

func TestInspectRuntimesInspectsEachDeclaredRuntimeOnce(t *testing.T) {
	digest := "sha256:" + strings.Repeat("a", 64)
	original := runDocker
	calls := map[string]int{}
	runDocker = func(args ...string) ([]byte, error) {
		image := args[len(args)-1]
		calls[image]++
		return []byte(`"` + digest + `"`), nil
	}
	t.Cleanup(func() { runDocker = original })

	runtimes := []modelpack.Runtime{
		{ID: "base", Image: "example/base:1", Digest: digest},
		{ID: "draft", Image: "example/draft:1", Digest: digest},
	}
	results := InspectRuntimes(runtimes)

	if len(results) != 2 || calls["example/base:1"] != 1 || calls["example/draft:1"] != 1 {
		t.Fatalf("InspectRuntimes() results = %#v, calls = %#v", results, calls)
	}
}
