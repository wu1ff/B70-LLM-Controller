package runtime

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"b70ctl/internal/modelpack"
)

func ownedDigest(char byte) string {
	return "sha256:" + strings.Repeat(string(char), 64)
}

// imageStateFake answers image inspect/rm for a fixed set of present images
// and an optional managed container, tracking removal attempts.
type imageStateFake struct {
	images    map[string]string // reference -> image ID
	removed   []string
	container string // JSON array body, "" = no container
	refuseRM  bool
}

func (fake *imageStateFake) install(t *testing.T) {
	t.Helper()
	setDockerHook(t, func(args ...string) ([]byte, error) {
		switch {
		case len(args) >= 1 && args[0] == "inspect":
			if fake.container == "" {
				return []byte("Error: No such object: b70ctl-runtime"), errFake
			}
			return []byte(fake.container), nil
		case len(args) >= 3 && args[0] == "image" && args[1] == "inspect":
			reference := args[len(args)-1]
			if id, present := fake.images[reference]; present {
				return []byte(`"` + id + `"`), nil
			}
			return missingImage(reference)
		case len(args) >= 3 && args[0] == "image" && args[1] == "rm":
			fake.removed = append(fake.removed, args[2])
			if fake.refuseRM {
				return []byte("Error response from daemon: conflict: unable to delete - image is being used by stopped container"), errFake
			}
			delete(fake.images, args[2])
			return []byte("Untagged: " + args[2]), nil
		}
		return []byte("unexpected docker invocation"), errFake
	})
}

var errFake = &fakeError{}

type fakeError struct{}

func (*fakeError) Error() string { return "exit status 1" }

func runningContainerJSON(imageID string) string {
	return `[{` +
		`"Config":{"Labels":{"b70ctl.managed":"true","b70ctl.pack_id":"pack","b70ctl.pack_version":"1.0.0","b70ctl.profile_id":"profile","b70ctl.health_path":"/health"}},` +
		`"Image":"` + imageID + `",` +
		`"State":{"Running":true,"ExitCode":0},` +
		`"NetworkSettings":{"Ports":{"8000/tcp":[{"HostIp":"127.0.0.1","HostPort":"65534"}]}}}]`
}

func TestOwnershipStoreLifecycle(t *testing.T) {
	root := t.TempDir()
	digest := ownedDigest('a')

	if records, err := LoadOwnership(root); err != nil || len(records) != 0 {
		t.Fatalf("missing store not empty: %#v, %v", records, err)
	}
	if err := RecordOwnership(root, digest, "example/runtime:1"); err != nil {
		t.Fatal(err)
	}
	records, err := LoadOwnership(root)
	if err != nil {
		t.Fatal(err)
	}
	record, exists := records[digest]
	if !exists || record.Digest != digest || record.Image != "example/runtime:1" || record.Acquired.IsZero() {
		t.Fatalf("record = %#v, %v", record, exists)
	}

	// A second record call is idempotent and preserves the original evidence.
	first := record.Acquired
	if err := RecordOwnership(root, digest, "other/ref:2"); err != nil {
		t.Fatal(err)
	}
	records, _ = LoadOwnership(root)
	if records[digest].Image != "example/runtime:1" || !records[digest].Acquired.Equal(first) {
		t.Fatalf("record was rewritten: %#v", records[digest])
	}

	if err := RemoveOwnership(root, digest); err != nil {
		t.Fatal(err)
	}
	if records, _ = LoadOwnership(root); len(records) != 0 {
		t.Fatalf("record not removed: %#v", records)
	}
	if err := RemoveOwnership(root, digest); err != nil {
		t.Fatalf("removing an absent record failed: %v", err)
	}
}

func TestOwnershipWriteIsAtomicAndResidueFree(t *testing.T) {
	root := t.TempDir()
	if err := RecordOwnership(root, ownedDigest('b'), "example/runtime:1"); err != nil {
		t.Fatal(err)
	}
	if err := RecordOwnership(root, ownedDigest('c'), "example/runtime:2"); err != nil {
		t.Fatal(err)
	}
	entries, err := os.ReadDir(root)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 1 || entries[0].Name() != "runtime-ownership.json" {
		names := make([]string, 0, len(entries))
		for _, entry := range entries {
			names = append(names, entry.Name())
		}
		t.Fatalf("data root holds unexpected files: %v", names)
	}
	data, err := os.ReadFile(filepath.Join(root, "runtime-ownership.json"))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(data), `"schema_version": 1`) {
		t.Fatalf("store missing schema version: %s", data)
	}
}

func TestMalformedOwnershipStoreFailsClosed(t *testing.T) {
	cases := map[string]string{
		"garbage":              "{not json",
		"wrong schema version": `{"schema_version":99,"runtimes":{}}`,
		"missing runtimes":     `{"schema_version":1}`,
		"null runtimes":        `{"schema_version":1,"runtimes":null}`,
		"key mismatch":         `{"schema_version":1,"runtimes":{"sha256:aaa":{"digest":"sha256:bbb","acquired":"2026-01-01T00:00:00Z"}}}`,
		"invalid digest":       `{"schema_version":1,"runtimes":{"not-a-digest":{"digest":"not-a-digest","acquired":"2026-01-01T00:00:00Z"}}}`,
		"unknown field":        `{"schema_version":1,"runtimes":{},"extra":true}`,
		"trailing object":      `{"schema_version":1,"runtimes":{}}{"schema_version":1}`,
	}
	for name, contents := range cases {
		t.Run(name, func(t *testing.T) {
			root := t.TempDir()
			path := filepath.Join(root, "runtime-ownership.json")
			if err := os.WriteFile(path, []byte(contents), 0o644); err != nil {
				t.Fatal(err)
			}
			if _, err := LoadOwnership(root); err == nil {
				t.Fatal("malformed store loaded successfully")
			}
			if err := RecordOwnership(root, ownedDigest('a'), "example/runtime:1"); err == nil {
				t.Fatal("recording overwrote a malformed store")
			}
			if err := RemoveOwnership(root, ownedDigest('a')); err == nil {
				t.Fatal("removal rewrote a malformed store")
			}
			after, err := os.ReadFile(path)
			if err != nil || string(after) != contents {
				t.Fatalf("malformed store was modified: %q, %v", after, err)
			}
		})
	}
}

func TestOwnershipPersistsAcrossLoad(t *testing.T) {
	root := t.TempDir()
	digest := ownedDigest('d')
	if err := RecordOwnership(root, digest, "example/runtime:1"); err != nil {
		t.Fatal(err)
	}
	// A fresh load stands in for a new process reading the same store.
	records, err := LoadOwnership(root)
	if err != nil {
		t.Fatal(err)
	}
	if _, exists := records[digest]; !exists {
		t.Fatalf("ownership did not persist: %#v", records)
	}
}

func TestAcquireOwnedRecordsOwnershipOnlyForSelfPulledImages(t *testing.T) {
	declared := testAcquisitionRuntime()

	t.Run("image already present is reused without claiming ownership", func(t *testing.T) {
		root := t.TempDir()
		fake := &imageStateFake{images: map[string]string{
			declared.Image: declared.Digest, declared.Digest: declared.Digest,
		}}
		fake.install(t)
		setDockerPullHook(t, func(context.Context, string, func(PullProgress)) ([]byte, error) {
			t.Fatal("present image triggered a pull")
			return nil, nil
		})

		result := AcquireOwned(context.Background(), root, declared, nil)
		if result.Outcome != AcquisitionReused {
			t.Fatalf("AcquireOwned() = %#v", result)
		}
		if records, err := LoadOwnership(root); err != nil || len(records) != 0 {
			t.Fatalf("reuse recorded ownership: %#v, %v", records, err)
		}
	})

	t.Run("successful pull and verification records ownership", func(t *testing.T) {
		root := t.TempDir()
		fake := &imageStateFake{images: map[string]string{}}
		fake.install(t)
		setDockerPullHook(t, func(_ context.Context, reference string, _ func(PullProgress)) ([]byte, error) {
			fake.images[reference] = declared.Digest
			return nil, nil
		})

		result := AcquireOwned(context.Background(), root, declared, nil)
		if result.Outcome != AcquisitionDownloaded {
			t.Fatalf("AcquireOwned() = %#v", result)
		}
		records, err := LoadOwnership(root)
		if err != nil {
			t.Fatal(err)
		}
		if record, exists := records[declared.Digest]; !exists || record.Image != declared.Image {
			t.Fatalf("ownership not recorded: %#v", records)
		}
	})

	t.Run("already owned image stays owned after reuse", func(t *testing.T) {
		root := t.TempDir()
		if err := RecordOwnership(root, declared.Digest, declared.Image); err != nil {
			t.Fatal(err)
		}
		fake := &imageStateFake{images: map[string]string{
			declared.Image: declared.Digest, declared.Digest: declared.Digest,
		}}
		fake.install(t)

		if result := AcquireOwned(context.Background(), root, declared, nil); result.Outcome != AcquisitionReused {
			t.Fatalf("AcquireOwned() = %#v", result)
		}
		if records, _ := LoadOwnership(root); len(records) != 1 {
			t.Fatalf("existing ownership lost: %#v", records)
		}
	})

	t.Run("pull failure records nothing", func(t *testing.T) {
		root := t.TempDir()
		(&imageStateFake{images: map[string]string{}}).install(t)
		setDockerPullHook(t, func(context.Context, string, func(PullProgress)) ([]byte, error) {
			return []byte("registry unavailable"), errFake
		})

		result := AcquireOwned(context.Background(), root, declared, nil)
		if result.Outcome != AcquisitionFailed {
			t.Fatalf("AcquireOwned() = %#v", result)
		}
		if records, err := LoadOwnership(root); err != nil || len(records) != 0 {
			t.Fatalf("failed pull recorded ownership: %#v, %v", records, err)
		}
	})

	t.Run("post-pull identity mismatch records nothing", func(t *testing.T) {
		root := t.TempDir()
		mismatch := "sha256:" + strings.Repeat("e", 64)
		(&imageStateFake{images: map[string]string{declared.Registry: mismatch}}).install(t)
		setDockerPullHook(t, func(context.Context, string, func(PullProgress)) ([]byte, error) {
			return nil, nil
		})

		result := AcquireOwned(context.Background(), root, declared, nil)
		if result.Outcome != AcquisitionFailed {
			t.Fatalf("AcquireOwned() = %#v", result)
		}
		if records, err := LoadOwnership(root); err != nil || len(records) != 0 {
			t.Fatalf("unverified pull recorded ownership: %#v, %v", records, err)
		}
	})

	t.Run("unpersistable ownership keeps the verified image usable", func(t *testing.T) {
		root := t.TempDir()
		if err := os.WriteFile(filepath.Join(root, "runtime-ownership.json"), []byte("corrupt"), 0o644); err != nil {
			t.Fatal(err)
		}
		fake := &imageStateFake{images: map[string]string{}}
		fake.install(t)
		setDockerPullHook(t, func(_ context.Context, reference string, _ func(PullProgress)) ([]byte, error) {
			fake.images[reference] = declared.Digest
			return nil, nil
		})

		result := AcquireOwned(context.Background(), root, declared, nil)
		if result.Outcome != AcquisitionDownloaded {
			t.Fatalf("acquisition failed on provenance write: %#v", result)
		}
		if data, err := os.ReadFile(filepath.Join(root, "runtime-ownership.json")); err != nil || string(data) != "corrupt" {
			t.Fatalf("corrupt store was touched: %q, %v", data, err)
		}
	})
}

func TestRemoveOwnedRuntimeRechecksEverythingImmediately(t *testing.T) {
	digest := ownedDigest('f')
	other := ownedDigest('5')

	t.Run("owned unreferenced stopped runtime is removed with its record", func(t *testing.T) {
		root := t.TempDir()
		if err := RecordOwnership(root, digest, "example/runtime:1"); err != nil {
			t.Fatal(err)
		}
		fake := &imageStateFake{images: map[string]string{"example/runtime:1": digest}}
		fake.install(t)

		result := RemoveOwnedRuntime(root, digest, map[string]struct{}{})
		if result.Status != ImageRemovalRemoved {
			t.Fatalf("RemoveOwnedRuntime() = %#v, removed = %v", result, fake.removed)
		}
		if len(fake.removed) != 1 || fake.removed[0] != "example/runtime:1" {
			t.Fatalf("removed references = %v", fake.removed)
		}
		if records, _ := LoadOwnership(root); len(records) != 0 {
			t.Fatalf("record survived removal: %#v", records)
		}
	})

	t.Run("missing store or unknown digest never deletes", func(t *testing.T) {
		root := t.TempDir()
		fake := &imageStateFake{images: map[string]string{"example/runtime:1": digest}}
		fake.install(t)

		if result := RemoveOwnedRuntime(root, digest, nil); result.Status != ImageRemovalRetained || fake.removed != nil {
			t.Fatalf("unowned removal = %#v, removed = %v", result, fake.removed)
		}
		if result := RemoveOwnedRuntime(t.TempDir(), other, nil); result.Status != ImageRemovalRetained {
			t.Fatalf("absent-store removal = %#v", result)
		}
	})

	t.Run("digest referenced by an installed pack is retained", func(t *testing.T) {
		root := t.TempDir()
		if err := RecordOwnership(root, digest, "example/runtime:1"); err != nil {
			t.Fatal(err)
		}
		fake := &imageStateFake{images: map[string]string{"example/runtime:1": digest}}
		fake.install(t)

		result := RemoveOwnedRuntime(root, digest, map[string]struct{}{digest: {}})
		if result.Status != ImageRemovalRetained || !strings.Contains(result.Reason, "referenced by an installed pack") {
			t.Fatalf("referenced removal = %#v", result)
		}
		if fake.removed != nil {
			t.Fatalf("referenced image removed: %v", fake.removed)
		}
	})

	t.Run("active starting runtime using the image blocks removal", func(t *testing.T) {
		root := t.TempDir()
		if err := RecordOwnership(root, digest, "example/runtime:1"); err != nil {
			t.Fatal(err)
		}
		fake := &imageStateFake{images: map[string]string{"example/runtime:1": digest}, container: runningContainerJSON(digest)}
		fake.install(t)

		status, err := Status()
		if err != nil || status.State != StateStarting || status.Image != digest {
			t.Fatalf("Status() = %#v, %v", status, err)
		}
		result := RemoveOwnedRuntime(root, digest, map[string]struct{}{})
		if result.Status != ImageRemovalRetained || !strings.Contains(result.Reason, "in use by the running model") {
			t.Fatalf("active removal = %#v", result)
		}
		if fake.removed != nil {
			t.Fatalf("active image removed: %v", fake.removed)
		}
	})

	t.Run("active runtime on a different image does not block removal", func(t *testing.T) {
		root := t.TempDir()
		if err := RecordOwnership(root, digest, "example/runtime:1"); err != nil {
			t.Fatal(err)
		}
		fake := &imageStateFake{images: map[string]string{"example/runtime:1": digest}, container: runningContainerJSON(other)}
		fake.install(t)

		result := RemoveOwnedRuntime(root, digest, map[string]struct{}{})
		if result.Status != ImageRemovalRemoved {
			t.Fatalf("removal = %#v", result)
		}
	})

	t.Run("unresolvable runtime state fails closed", func(t *testing.T) {
		root := t.TempDir()
		if err := RecordOwnership(root, digest, "example/runtime:1"); err != nil {
			t.Fatal(err)
		}
		fake := &imageStateFake{images: map[string]string{"example/runtime:1": digest}}
		fake.install(t)
		setDockerHook(t, func(args ...string) ([]byte, error) {
			return []byte("cannot connect to the docker daemon"), errFake
		})

		result := RemoveOwnedRuntime(root, digest, map[string]struct{}{})
		if result.Status != ImageRemovalRetained || !strings.Contains(result.Reason, "runtime state is unavailable") {
			t.Fatalf("unresolved-state removal = %#v", result)
		}
	})

	t.Run("docker refusal retains image and record", func(t *testing.T) {
		root := t.TempDir()
		if err := RecordOwnership(root, digest, "example/runtime:1"); err != nil {
			t.Fatal(err)
		}
		fake := &imageStateFake{images: map[string]string{"example/runtime:1": digest}, refuseRM: true}
		fake.install(t)

		result := RemoveOwnedRuntime(root, digest, map[string]struct{}{})
		if result.Status != ImageRemovalRetained {
			t.Fatalf("refused removal = %#v", result)
		}
		if records, _ := LoadOwnership(root); len(records) != 1 {
			t.Fatalf("record dropped after Docker refusal: %#v", records)
		}
	})

	t.Run("image missing on disk clears only stale metadata", func(t *testing.T) {
		root := t.TempDir()
		if err := RecordOwnership(root, digest, "example/runtime:1"); err != nil {
			t.Fatal(err)
		}
		fake := &imageStateFake{images: map[string]string{}}
		fake.install(t)

		result := RemoveOwnedRuntime(root, digest, map[string]struct{}{})
		if result.Status != ImageRemovalMissing || fake.removed != nil {
			t.Fatalf("stale removal = %#v, removed = %v", result, fake.removed)
		}
		if records, _ := LoadOwnership(root); len(records) != 0 {
			t.Fatalf("stale record kept: %#v", records)
		}
	})

	t.Run("only the exact digest is removed", func(t *testing.T) {
		root := t.TempDir()
		if err := RecordOwnership(root, digest, "example/runtime:1"); err != nil {
			t.Fatal(err)
		}
		if err := RecordOwnership(root, other, "example/unrelated:1"); err != nil {
			t.Fatal(err)
		}
		fake := &imageStateFake{images: map[string]string{"example/runtime:1": digest, "example/unrelated:1": other}}
		fake.install(t)

		if result := RemoveOwnedRuntime(root, digest, map[string]struct{}{}); result.Status != ImageRemovalRemoved {
			t.Fatalf("removal = %#v", result)
		}
		for _, reference := range fake.removed {
			if strings.Contains(reference, "unrelated") {
				t.Fatalf("unrelated image removed: %v", fake.removed)
			}
		}
		if _, err := fake.images["example/unrelated:1"]; !err {
			t.Fatal("unrelated image vanished from Docker state")
		}
		records, _ := LoadOwnership(root)
		if _, kept := records[other]; !kept {
			t.Fatalf("unrelated ownership record dropped: %#v", records)
		}
	})

	t.Run("retagged image falls back to the exact digest reference", func(t *testing.T) {
		root := t.TempDir()
		if err := RecordOwnership(root, digest, "example/runtime:1"); err != nil {
			t.Fatal(err)
		}
		moved := ownedDigest('6')
		fake := &imageStateFake{images: map[string]string{"example/runtime:1": moved, digest: digest}}
		fake.install(t)

		result := RemoveOwnedRuntime(root, digest, map[string]struct{}{})
		if result.Status != ImageRemovalRemoved {
			t.Fatalf("retagged removal = %#v, removed = %v", result, fake.removed)
		}
		if len(fake.removed) != 1 || fake.removed[0] != digest {
			t.Fatalf("removed references = %v", fake.removed)
		}
	})

	t.Run("malformed ownership store refuses deletion", func(t *testing.T) {
		root := t.TempDir()
		if err := os.WriteFile(filepath.Join(root, "runtime-ownership.json"), []byte("corrupt"), 0o644); err != nil {
			t.Fatal(err)
		}
		fake := &imageStateFake{images: map[string]string{"example/runtime:1": digest}}
		fake.install(t)

		result := RemoveOwnedRuntime(root, digest, map[string]struct{}{})
		if result.Status != ImageRemovalRetained || !strings.Contains(result.Reason, "could not be read") {
			t.Fatalf("corrupt-store removal = %#v", result)
		}
		if fake.removed != nil {
			t.Fatalf("image removed despite unreadable store: %v", fake.removed)
		}
	})
}

func TestReferencedDigestsUsesExactDigestIdentity(t *testing.T) {
	digest := ownedDigest('7')
	other := ownedDigest('j')
	manifests := []*modelpack.Manifest{
		{Runtimes: []modelpack.Runtime{{ID: "one", Image: "example/one:1", Digest: digest}, {ID: "two", Image: "example/two:1", Digest: digest}}},
		{Runtimes: []modelpack.Runtime{{ID: "three", Image: "example/runtime:1", Digest: other}}},
		nil,
	}
	referenced := ReferencedDigests(manifests)
	if len(referenced) != 2 {
		t.Fatalf("referenced = %#v", referenced)
	}
	if _, exists := referenced[digest]; !exists {
		t.Fatalf("digest not referenced: %#v", referenced)
	}
	if _, exists := referenced[other]; !exists {
		t.Fatalf("second digest not referenced: %#v", referenced)
	}
}
