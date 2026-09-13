package runtime

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"

	"b70ctl/internal/modelpack"
)

// B70 LLM Controller deletes only runtime images it can prove it introduced.
// An image is b70ctl-owned exclusively when Controller pulled the exact
// immutable digest itself and the downloaded image verified against the pack
// declaration; the persisted store below is that proof. Images already on the
// machine — including ones that merely look like b70ctl runtimes — have
// unknown ownership and are never deleted.

const ownershipSchemaVersion = 1

const ownershipFileName = "runtime-ownership.json"

// OwnershipRecord proves b70ctl acquired one exact runtime digest.
type OwnershipRecord struct {
	Digest   string    `json:"digest"`
	Image    string    `json:"image,omitempty"`
	Acquired time.Time `json:"acquired"`
}

type ownershipFile struct {
	SchemaVersion int                        `json:"schema_version"`
	Runtimes      map[string]OwnershipRecord `json:"runtimes"`
}

// ReferencedDigests returns the exact runtime digests referenced by all
// installed packs. Installed packs define persistent runtime need: an owned
// image is unused only when zero installed packs reference its digest.
func ReferencedDigests(manifests []*modelpack.Manifest) map[string]struct{} {
	referenced := map[string]struct{}{}
	for _, manifest := range manifests {
		if manifest == nil {
			continue
		}
		for _, declared := range manifest.Runtimes {
			referenced[declared.Digest] = struct{}{}
		}
	}
	return referenced
}

// LoadOwnership reads the ownership store. A missing store is empty; a
// malformed or unreadable store is an error so every deletion path can fail
// closed instead of guessing ownership.
func LoadOwnership(dataRoot string) (map[string]OwnershipRecord, error) {
	data, err := os.ReadFile(ownershipPath(dataRoot))
	if errors.Is(err, os.ErrNotExist) {
		return map[string]OwnershipRecord{}, nil
	}
	if err != nil {
		return nil, fmt.Errorf("read runtime ownership store: %w", err)
	}

	var file ownershipFile
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&file); err != nil {
		return nil, errors.New("runtime ownership store is malformed")
	}
	var extra any
	if err := decoder.Decode(&extra); !errors.Is(err, io.EOF) {
		return nil, errors.New("runtime ownership store is malformed")
	}
	if file.SchemaVersion != ownershipSchemaVersion || file.Runtimes == nil {
		return nil, errors.New("runtime ownership store is malformed")
	}
	for key, record := range file.Runtimes {
		if key != record.Digest || !imageIDPattern.MatchString(record.Digest) {
			return nil, errors.New("runtime ownership store is malformed")
		}
	}
	return file.Runtimes, nil
}

// RecordOwnership persists proof that b70ctl acquired digest itself. An
// existing record is preserved as-is, and a malformed store is never
// overwritten — both keep the file's evidence trustworthy.
func RecordOwnership(dataRoot, digest, image string) error {
	if !imageIDPattern.MatchString(digest) {
		return errors.New("runtime digest is invalid")
	}
	records, err := LoadOwnership(dataRoot)
	if err != nil {
		return err
	}
	if _, exists := records[digest]; exists {
		return nil
	}
	records[digest] = OwnershipRecord{Digest: digest, Image: image, Acquired: time.Now().UTC()}
	return writeOwnership(dataRoot, records)
}

// RemoveOwnership drops one record. Removing an absent record succeeds; a
// malformed store is refused.
func RemoveOwnership(dataRoot, digest string) error {
	records, err := LoadOwnership(dataRoot)
	if err != nil {
		return err
	}
	if _, exists := records[digest]; !exists {
		return nil
	}
	delete(records, digest)
	return writeOwnership(dataRoot, records)
}

func ownershipPath(dataRoot string) string {
	return filepath.Join(dataRoot, ownershipFileName)
}

func writeOwnership(dataRoot string, records map[string]OwnershipRecord) error {
	if err := os.MkdirAll(dataRoot, 0o755); err != nil {
		return fmt.Errorf("create controller data directory: %w", err)
	}
	file, err := os.CreateTemp(dataRoot, "."+ownershipFileName+"-*")
	if err != nil {
		return fmt.Errorf("create temporary ownership store: %w", err)
	}
	temporary := file.Name()
	defer os.Remove(temporary)

	if err := file.Chmod(0o644); err != nil {
		file.Close()
		return fmt.Errorf("set ownership store permissions: %w", err)
	}
	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(ownershipFile{SchemaVersion: ownershipSchemaVersion, Runtimes: records}); err != nil {
		file.Close()
		return fmt.Errorf("write ownership store: %w", err)
	}
	if err := file.Sync(); err != nil {
		file.Close()
		return fmt.Errorf("sync ownership store: %w", err)
	}
	if err := file.Close(); err != nil {
		return fmt.Errorf("close ownership store: %w", err)
	}
	if err := os.Rename(temporary, ownershipPath(dataRoot)); err != nil {
		return fmt.Errorf("save ownership store: %w", err)
	}
	return nil
}

// AcquireOwned acquires declared and records ownership provenance when — and
// only when — b70ctl performed the pull itself and the downloaded image
// verified against the exact pack digest. A reused pre-existing image never
// claims ownership. When provenance cannot be persisted the acquisition still
// succeeds: the verified image stays usable and simply remains not owned, so
// Controller will never delete it.
func AcquireOwned(ctx context.Context, dataRoot string, declared modelpack.Runtime, notify func(PullProgress)) AcquisitionResult {
	result := Acquire(ctx, declared, notify)
	if result.Outcome != AcquisitionDownloaded {
		return result
	}
	_ = RecordOwnership(dataRoot, declared.Digest, declared.Image)
	return result
}

// RemoveOwnedRuntime removes one b70ctl-owned runtime image after rechecking
// every safety condition immediately before removal: the ownership record
// still exists for the exact digest, no installed pack references the digest
// (referenced is the caller's freshly computed reference set), no active
// managed runtime runs the image, and Docker still resolves the expected
// exact identity. Removal is never forced; when Docker refuses, both the
// image and the ownership record are retained.
func RemoveOwnedRuntime(dataRoot, digest string, referenced map[string]struct{}) ImageRemovalResult {
	records, err := LoadOwnership(dataRoot)
	if err != nil {
		return retainedOwnership("ownership record could not be read: " + err.Error())
	}
	record, owned := records[digest]
	if !owned {
		return retainedOwnership("runtime was not acquired by b70ctl")
	}
	if _, used := referenced[digest]; used {
		return retainedOwnership("runtime is referenced by an installed pack")
	}
	status, err := Status()
	if err != nil {
		return retainedOwnership("runtime state is unavailable: " + err.Error())
	}
	if (status.State == StateStarting || status.State == StateRunning) && status.Image == digest {
		return retainedOwnership("runtime is in use by the running model")
	}

	image := record.Image
	if image == "" || InspectImage(image, digest).Status != ImagePresent {
		image = digest
	}
	result, err := RemoveImage(image, digest)
	if err != nil {
		return ImageRemovalResult{Status: ImageRemovalRetained, Reason: err.Error()}
	}
	switch result.Status {
	case ImageRemovalRemoved, ImageRemovalMissing:
		// The image is gone from Docker; Controller's own record is stale.
		_ = RemoveOwnership(dataRoot, digest)
	}
	return result
}

func retainedOwnership(reason string) ImageRemovalResult {
	return ImageRemovalResult{Status: ImageRemovalRetained, Reason: reason}
}
