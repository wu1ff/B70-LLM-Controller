package install

import (
	"context"
	"errors"
	"fmt"

	"b70ctl/internal/hf"
	"b70ctl/internal/modelpack"
	"b70ctl/internal/modelstore"
	"b70ctl/internal/packstore"
	runtimebackend "b70ctl/internal/runtime"
)

type State string

const (
	Present    State = "Present"
	Missing    State = "Missing"
	Incomplete State = "Incomplete"
)

type Outcome string

const (
	Reused     Outcome = "Reused"
	Downloaded Outcome = "Downloaded"
	Skipped    Outcome = "Skipped"
	Failed     Outcome = "Failed"
)

type Artifact struct {
	ModelID    string
	Name       string
	Repo       string
	Revision   string
	Kind       string
	Gated      bool
	Files      []modelpack.ModelFile
	State      State
	SkipReason string
}

type Plan struct {
	SelectedTargetIDs []string
	Artifacts         []Artifact
	Runtimes          []modelpack.Runtime
}

type Counts struct {
	SelectedTargets int
	Existing        int
	Incomplete      int
	Downloads       int
	Skipped         int
}

type Item struct {
	Artifact Artifact
	Outcome  Outcome
	Reason   string
	Err      error
}

type Result struct {
	Pack         *packstore.InstalledPack
	Items        []Item
	RuntimeItems []RuntimeItem
	Inventory    []modelstore.Artifact
}

type Event struct {
	Artifact        Artifact
	Repository      *hf.Repository
	Progress        hf.Progress
	Runtime         *modelpack.Runtime
	RuntimeProgress runtimebackend.PullProgress
}

type RuntimeItem struct {
	Runtime modelpack.Runtime
	Outcome runtimebackend.AcquisitionOutcome
	Reason  string
	Err     error
}

func Targets(manifest *modelpack.Manifest) []modelpack.Model {
	var targets []modelpack.Model
	for _, model := range manifest.Models {
		if model.Kind == "target" {
			targets = append(targets, model)
		}
	}
	return targets
}

func BuildPlan(manifest *modelpack.Manifest, selectedTargetIDs []string, inventory []modelstore.Artifact) (Plan, error) {
	models := make(map[string]modelpack.Model, len(manifest.Models))
	for _, model := range manifest.Models {
		models[model.ID] = model
	}
	runtimes := make(map[string]modelpack.Runtime, len(manifest.Runtimes))
	for _, declared := range manifest.Runtimes {
		runtimes[declared.ID] = declared
	}
	modes := make(map[string]modelpack.Mode, len(manifest.Modes))
	for _, mode := range manifest.Modes {
		modes[mode.ID] = mode
	}

	selected := make(map[string]struct{}, len(selectedTargetIDs))
	for _, id := range selectedTargetIDs {
		model, found := models[id]
		if !found || model.Kind != "target" {
			return Plan{}, fmt.Errorf("model %q is not a target", id)
		}
		selected[id] = struct{}{}
	}

	plan := Plan{SelectedTargetIDs: make([]string, 0, len(selected))}
	seen := map[string]struct{}{}
	seenRuntimes := map[string]struct{}{}
	add := func(model modelpack.Model) {
		identity := model.Repo + "\x00" + model.Revision
		if _, exists := seen[identity]; exists {
			return
		}
		seen[identity] = struct{}{}
		state := State(modelstore.Assess(inventory, model).State)
		plan.Artifacts = append(plan.Artifacts, Artifact{
			ModelID: model.ID, Name: model.Name, Repo: model.Repo, Revision: model.Revision,
			Kind: model.Kind, Gated: model.Gated, Files: model.Files, State: state,
		})
	}

	for _, target := range manifest.Models {
		if target.Kind != "target" {
			continue
		}
		if _, chosen := selected[target.ID]; !chosen {
			continue
		}
		plan.SelectedTargetIDs = append(plan.SelectedTargetIDs, target.ID)
		add(target)
		for _, profile := range manifest.Profiles {
			if profile.ModelID != target.ID {
				continue
			}
			for _, assistantID := range modes[profile.Mode].Assistants {
				add(models[assistantID])
			}
			declared := runtimes[profile.RuntimeID]
			if _, exists := seenRuntimes[declared.Digest]; !exists {
				seenRuntimes[declared.Digest] = struct{}{}
				plan.Runtimes = append(plan.Runtimes, declared)
			}
		}
	}
	return plan, nil
}

func (plan Plan) Counts() Counts {
	counts := Counts{SelectedTargets: len(plan.SelectedTargetIDs)}
	for _, artifact := range plan.Artifacts {
		switch {
		case artifact.State == Present:
			counts.Existing++
		case artifact.State == Incomplete:
			counts.Incomplete++
		case artifact.SkipReason != "":
			counts.Skipped++
		default:
			counts.Downloads++
		}
	}
	return counts
}

func (plan Plan) HasMissingGated() bool {
	for _, artifact := range plan.Artifacts {
		if artifact.State == Missing && artifact.Gated && artifact.SkipReason == "" {
			return true
		}
	}
	return false
}

func (plan *Plan) SkipMissingGated(reason string) {
	for index := range plan.Artifacts {
		artifact := &plan.Artifacts[index]
		if artifact.State == Missing && artifact.Gated {
			artifact.SkipReason = reason
		}
	}
}

func Prepare(ctx context.Context, sourcePath, storeRoot, dataRoot, modelRoot string, source string, plan Plan, client *hf.Client, notify func(Event)) (Result, error) {
	return prepare(ctx, sourcePath, storeRoot, modelRoot, plan, operations{
		inspect: client.Inspect,
		install: client.Install,
		acquire: func(ctx context.Context, declared modelpack.Runtime, notify func(runtimebackend.PullProgress)) runtimebackend.AcquisitionResult {
			return runtimebackend.AcquireOwned(ctx, dataRoot, declared, notify)
		},
	}, source, notify)
}

type operations struct {
	inspect func(context.Context, string, string) (hf.Repository, error)
	install func(context.Context, string, hf.Repository, func(hf.Progress)) (hf.Result, error)
	acquire func(context.Context, modelpack.Runtime, func(runtimebackend.PullProgress)) runtimebackend.AcquisitionResult
}

func prepare(ctx context.Context, sourcePath, storeRoot, modelRoot string, plan Plan, ops operations, source string, notify func(Event)) (Result, error) {
	installed, err := packstore.Import(sourcePath, storeRoot, source)
	if err != nil {
		return Result{}, err
	}
	result := Result{Pack: installed, Items: make([]Item, 0, len(plan.Artifacts))}
	inventory, err := modelstore.Scan(modelRoot)
	if err != nil {
		return result, err
	}

	verified, err := verifyAccess(ctx, plan, inventory, ops.inspect, notify)
	if err != nil {
		return result, err
	}

	for _, artifact := range plan.Artifacts {
		readiness := modelstore.Assess(inventory, artifact.model())
		if readiness.State == modelstore.Present {
			result.Items = append(result.Items, Item{Artifact: artifact, Outcome: Reused})
			continue
		}
		if readiness.State == modelstore.Incomplete {
			result.Items = append(result.Items, Item{Artifact: artifact, Outcome: Failed, Reason: "local model is incomplete"})
			continue
		}
		if artifact.SkipReason != "" {
			result.Items = append(result.Items, Item{Artifact: artifact, Outcome: Skipped, Reason: artifact.SkipReason})
			continue
		}
		repository, verifiedHere := verified[artifact.Repo+"\x00"+artifact.Revision]
		if !verifiedHere {
			if notify != nil {
				notify(Event{Artifact: artifact})
			}
			repository, err = ops.inspect(ctx, artifact.Repo, artifact.Revision)
			if err != nil {
				result.Items = append(result.Items, failedItem(artifact, err))
				continue
			}
		}
		if notify != nil {
			notify(Event{Artifact: artifact, Repository: &repository, Progress: initialProgress(repository)})
		}
		download, err := ops.install(ctx, modelRoot, repository, func(progress hf.Progress) {
			if notify != nil {
				notify(Event{Artifact: artifact, Repository: &repository, Progress: progress})
			}
		})
		if err != nil {
			result.Items = append(result.Items, failedItem(artifact, err))
			continue
		}
		if !modelstore.Check(download.Path, artifact.Files).Complete {
			result.Items = append(result.Items, Item{Artifact: artifact, Outcome: Failed, Reason: "downloaded model is incomplete"})
			continue
		}
		outcome := Downloaded
		if download.AlreadyPresent {
			outcome = Reused
		}
		result.Items = append(result.Items, Item{Artifact: artifact, Outcome: outcome})
	}

	for _, declared := range plan.Runtimes {
		acquisition := ops.acquire(ctx, declared, func(progress runtimebackend.PullProgress) {
			if notify != nil {
				value := declared
				notify(Event{Runtime: &value, RuntimeProgress: progress})
			}
		})
		result.RuntimeItems = append(result.RuntimeItems, RuntimeItem{
			Runtime: declared, Outcome: acquisition.Outcome, Reason: acquisition.Reason, Err: acquisition.Err,
		})
	}

	result.Inventory, err = modelstore.Scan(modelRoot)
	if err != nil {
		return result, err
	}
	return result, nil
}

// verifyAccess resolves revision metadata for every artifact that still
// needs to be downloaded and stops at the first access failure before any
// bytes move, so an unreachable model surfaces the token guidance instead
// of a partial multi-gigabyte install. Non-access failures are left to the
// per-artifact flow, which also reuses the successful resolutions.
func verifyAccess(ctx context.Context, plan Plan, inventory []modelstore.Artifact, inspect func(context.Context, string, string) (hf.Repository, error), notify func(Event)) (map[string]hf.Repository, error) {
	verified := make(map[string]hf.Repository)
	for _, artifact := range plan.Artifacts {
		if artifact.SkipReason != "" {
			continue
		}
		if modelstore.Assess(inventory, artifact.model()).State != modelstore.Missing {
			continue
		}
		if notify != nil {
			notify(Event{Artifact: artifact})
		}
		repository, err := inspect(ctx, artifact.Repo, artifact.Revision)
		if err != nil {
			if hf.IsAccessFailure(err) {
				return verified, fmt.Errorf("%s: %w", artifact.Name, err)
			}
			continue
		}
		verified[artifact.Repo+"\x00"+artifact.Revision] = repository
	}
	return verified, nil
}

func (artifact Artifact) model() modelpack.Model {
	return modelpack.Model{
		ID: artifact.ModelID, Name: artifact.Name, Kind: artifact.Kind, Repo: artifact.Repo,
		Revision: artifact.Revision, Gated: artifact.Gated, Files: artifact.Files,
	}
}

func initialProgress(repository hf.Repository) hf.Progress {
	progress := hf.Progress{TotalFiles: len(repository.Files)}
	progress.TotalBytes, progress.TotalKnown = repository.TotalSize()
	if len(repository.Files) > 0 {
		progress.CurrentFile = repository.Files[0].Path
		progress.FileNumber = 1
	}
	return progress
}

func failedItem(artifact Artifact, err error) Item {
	return Item{Artifact: artifact, Outcome: Failed, Reason: FailureReason(err), Err: err}
}

func FailureReason(err error) string {
	switch {
	case errors.Is(err, hf.ErrAuthenticationRequired), errors.Is(err, hf.ErrAccessUncertain):
		return "Hugging Face token not configured"
	case errors.Is(err, hf.ErrAccessDenied):
		return "Hugging Face access denied"
	case errors.Is(err, hf.ErrRepositoryNotFound):
		return "repository not found"
	case errors.Is(err, hf.ErrRevisionNotFound):
		return "revision not found"
	case errors.Is(err, hf.ErrNetworkUnavailable):
		return "network unavailable"
	case errors.Is(err, hf.ErrDestinationExists):
		return "model directory already exists"
	default:
		return err.Error()
	}
}

func (result Result) HasAccessIssue() bool {
	for _, item := range result.Items {
		if hf.IsAccessFailure(item.Err) {
			return true
		}
	}
	return false
}

func (result Result) HasRuntimeFailure() bool {
	for _, item := range result.RuntimeItems {
		if item.Outcome == runtimebackend.AcquisitionFailed {
			return true
		}
	}
	return false
}
