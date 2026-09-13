package runtime

import (
	"errors"
	"reflect"
	"strings"
	"testing"
)

func TestRemoveImageVerifiesIdentityThenRemovesExactTag(t *testing.T) {
	digest := "sha256:" + strings.Repeat("a", 64)
	var calls [][]string
	setDockerHook(t, func(args ...string) ([]byte, error) {
		calls = append(calls, append([]string(nil), args...))
		if args[1] == "inspect" {
			return []byte(`"` + digest + `"`), nil
		}
		return []byte("Untagged: example/runtime:1\n"), nil
	})

	result, err := RemoveImage("example/runtime:1", digest)
	if err != nil || result.Status != ImageRemovalRemoved {
		t.Fatalf("RemoveImage() = %#v, %v", result, err)
	}
	want := [][]string{
		{"image", "inspect", "--format", "{{json .Id}}", "example/runtime:1"},
		{"image", "rm", "example/runtime:1"},
	}
	if !reflect.DeepEqual(calls, want) {
		t.Fatalf("Docker calls = %#v, want %#v", calls, want)
	}
}

func TestRemoveImageMissingOrMismatchedIsNoOp(t *testing.T) {
	digest := "sha256:" + strings.Repeat("a", 64)
	calls := 0
	setDockerHook(t, func(args ...string) ([]byte, error) {
		calls++
		return []byte(`"sha256:` + strings.Repeat("b", 64) + `"`), nil
	})

	result, err := RemoveImage("example/runtime:1", digest)
	if err != nil || result.Status != ImageRemovalMissing || calls != 1 {
		t.Fatalf("RemoveImage() = %#v, %v; calls = %d", result, err, calls)
	}
}

func TestRemoveImageRetainsDockerRefusalWithoutForce(t *testing.T) {
	digest := "sha256:" + strings.Repeat("a", 64)
	setDockerHook(t, func(args ...string) ([]byte, error) {
		if args[1] == "inspect" {
			return []byte(`"` + digest + `"`), nil
		}
		if !reflect.DeepEqual(args, []string{"image", "rm", "example/runtime:1"}) {
			t.Fatalf("unexpected Docker removal call: %v", args)
		}
		return []byte("Error response from daemon: conflict: unable to remove repository reference - container is using its referenced image"), errors.New("exit status 1")
	})

	result, err := RemoveImage("example/runtime:1", digest)
	if err != nil || result.Status != ImageRemovalRetained || result.Reason != "Docker reports image is in use" {
		t.Fatalf("RemoveImage() = %#v, %v", result, err)
	}
}
