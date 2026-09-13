package modelstore

import (
	"os"
	"path/filepath"
	"strings"

	"b70ctl/internal/modelpack"
)

type CheckResult struct {
	Complete       bool
	CheckedFiles   int
	MissingFiles   []string
	SizeMismatches []string
}

type Readiness string

const (
	Missing    Readiness = "Missing"
	Present    Readiness = "Present"
	Incomplete Readiness = "Incomplete"
)

type ReadinessResult struct {
	State    Readiness
	Artifact Artifact
	Check    CheckResult
}

func Assess(artifacts []Artifact, model modelpack.Model) ReadinessResult {
	artifact, found := Find(artifacts, model.Repo, model.Revision)
	if !found {
		return ReadinessResult{State: Missing}
	}
	check := Check(artifact.Path, model.Files)
	state := Incomplete
	if check.Complete {
		state = Present
	}
	return ReadinessResult{State: state, Artifact: artifact, Check: check}
}

func Check(artifactPath string, expected []modelpack.ModelFile) CheckResult {
	root, err := filepath.Abs(artifactPath)
	if err != nil {
		return CheckResult{MissingFiles: expectedPaths(expected)}
	}
	root = filepath.Clean(root)
	result := CheckResult{}
	for _, file := range expected {
		result.CheckedFiles++
		if file.Path == "" || filepath.IsAbs(file.Path) {
			result.MissingFiles = append(result.MissingFiles, file.Path)
			continue
		}
		path := filepath.Join(root, filepath.Clean(filepath.FromSlash(file.Path)))
		relative, err := filepath.Rel(root, path)
		if err != nil || relative == "." || relative == ".." || strings.HasPrefix(relative, ".."+string(filepath.Separator)) {
			result.MissingFiles = append(result.MissingFiles, file.Path)
			continue
		}
		info, err := os.Stat(path)
		if err != nil || !info.Mode().IsRegular() {
			result.MissingFiles = append(result.MissingFiles, file.Path)
			continue
		}
		if file.Size != nil && info.Size() != *file.Size {
			result.SizeMismatches = append(result.SizeMismatches, file.Path)
		}
	}
	result.Complete = len(result.MissingFiles) == 0 && len(result.SizeMismatches) == 0
	return result
}

func expectedPaths(expected []modelpack.ModelFile) []string {
	paths := make([]string, len(expected))
	for index, file := range expected {
		paths[index] = file.Path
	}
	return paths
}
