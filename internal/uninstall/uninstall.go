package uninstall

import (
	"errors"
	"fmt"
	"path/filepath"

	"b70ctl/internal/modelpack"
	"b70ctl/internal/modelstore"
	"b70ctl/internal/packstore"
	"b70ctl/internal/runtime"
)

type Mode uint8

const (
	DeleteEverything Mode = iota
	KeepModels
)

type Item struct {
	Artifact string
	Outcome  string
}

type Result struct {
	PackRemoved bool
	Items       []Item
}

type modelPlan struct {
	model  modelpack.Model
	shared bool
}

type runtimePlan struct {
	runtime modelpack.Runtime
	shared  bool
}

func Run(storeRoot, dataRoot, modelRoot, packID, version string, mode Mode) (Result, error) {
	if mode != DeleteEverything && mode != KeepModels {
		return Result{}, errors.New("uninstall mode is invalid")
	}
	installed, err := loadInstalled(storeRoot)
	if err != nil {
		return Result{}, err
	}
	var target *modelpack.Manifest
	var others []*modelpack.Manifest
	for _, manifest := range installed {
		if manifest.ID == packID && manifest.Version == version {
			target = manifest
		} else {
			others = append(others, manifest)
		}
	}
	if target == nil {
		return Result{}, fmt.Errorf("pack %s %s is not installed", packID, version)
	}

	status, err := runtime.Status()
	if err != nil {
		return Result{}, err
	}
	if status.PackID == packID && status.PackVersion == version &&
		(status.State == runtime.StateStarting || status.State == runtime.StateRunning) {
		return Result{}, errors.New("Stop this model before uninstalling the pack.")
	}

	models, runtimes := buildPlan(target, others)
	if err := packstore.Remove(storeRoot, packID, version); err != nil {
		return Result{}, err
	}

	result := Result{PackRemoved: true}
	for _, planned := range models {
		item := Item{Artifact: planned.model.Name}
		switch {
		case mode == KeepModels:
			item.Outcome = "Retained — keeping models"
		case planned.shared:
			item.Outcome = "Retained — used by another installed pack"
		default:
			item.Outcome = removeModel(modelRoot, planned.model)
		}
		result.Items = append(result.Items, item)
	}
	for _, planned := range runtimes {
		artifact := "Runtime image"
		if len(runtimes) > 1 {
			artifact = planned.runtime.Image
		}
		item := Item{Artifact: artifact}
		if planned.shared {
			item.Outcome = "Retained — used by another installed pack"
		} else {
			item.Outcome = removeRuntime(dataRoot, planned.runtime)
		}
		result.Items = append(result.Items, item)
	}
	return result, nil
}

func loadInstalled(storeRoot string) ([]*modelpack.Manifest, error) {
	packs, err := packstore.List(storeRoot)
	if err != nil {
		return nil, err
	}
	result := make([]*modelpack.Manifest, 0, len(packs))
	for _, pack := range packs {
		manifest, err := modelpack.Load(filepath.Join(storeRoot, pack.ID, pack.Version))
		if err != nil {
			return nil, fmt.Errorf("load installed pack %s %s: %w", pack.ID, pack.Version, err)
		}
		result = append(result, manifest)
	}
	return result, nil
}

func buildPlan(target *modelpack.Manifest, others []*modelpack.Manifest) ([]modelPlan, []runtimePlan) {
	sharedModels := map[string]struct{}{}
	sharedRuntimes := map[string]struct{}{}
	for _, manifest := range others {
		for _, model := range manifest.Models {
			sharedModels[modelKey(model)] = struct{}{}
		}
		for _, image := range manifest.Runtimes {
			sharedRuntimes[image.Digest] = struct{}{}
		}
	}

	var models []modelPlan
	seenModels := map[string]struct{}{}
	for _, model := range target.Models {
		key := modelKey(model)
		if _, seen := seenModels[key]; seen {
			continue
		}
		seenModels[key] = struct{}{}
		_, shared := sharedModels[key]
		models = append(models, modelPlan{model: model, shared: shared})
	}
	var runtimes []runtimePlan
	seenRuntimes := map[string]struct{}{}
	for _, image := range target.Runtimes {
		if _, seen := seenRuntimes[image.Digest]; seen {
			continue
		}
		seenRuntimes[image.Digest] = struct{}{}
		_, shared := sharedRuntimes[image.Digest]
		runtimes = append(runtimes, runtimePlan{runtime: image, shared: shared})
	}
	return models, runtimes
}

func modelKey(model modelpack.Model) string {
	return model.Repo + "\x00" + model.Revision
}

func removeModel(root string, model modelpack.Model) string {
	status, err := modelstore.Remove(root, model.Repo, model.Revision)
	if err != nil {
		return "Failed — " + err.Error()
	}
	switch status {
	case modelstore.RemovalRemoved:
		return "Removed"
	case modelstore.RemovalExternalHFCache:
		return "Retained — external Hugging Face cache"
	case modelstore.RemovalMultipleCopies:
		return "Retained — multiple local copies found"
	default:
		return "Not found"
	}
}

// removeRuntime removes an unshared pack runtime image. Controller deletes
// only images it provably acquired itself; anything else is retained and
// reported as an outcome, never an uninstall failure.
func removeRuntime(dataRoot string, declared modelpack.Runtime) string {
	records, err := runtime.LoadOwnership(dataRoot)
	if err != nil {
		return "Retained — ownership could not be verified"
	}
	if _, owned := records[declared.Digest]; !owned {
		return "Retained — it was not acquired by b70ctl"
	}
	// A runtime pulled by its immutable registry reference never gains the
	// declared image tag locally, so resolve the removal reference the same
	// way acquisition does: fall back to the digest itself.
	reference := declared.Image
	if runtime.InspectImage(reference, declared.Digest).Status != runtime.ImagePresent {
		reference = declared.Digest
	}
	result, err := runtime.RemoveImage(reference, declared.Digest)
	if err != nil {
		return "Failed — " + err.Error()
	}
	switch result.Status {
	case runtime.ImageRemovalRemoved, runtime.ImageRemovalMissing:
		// The image is gone from Docker, so Controller's ownership record is
		// stale metadata about an image it can no longer manage.
		_ = runtime.RemoveOwnership(dataRoot, declared.Digest)
		if result.Status == runtime.ImageRemovalRemoved {
			return "Removed"
		}
		return "Not found"
	default:
		return "Retained — " + result.Reason
	}
}
