// flashnextpackcheck validates local pack distribution and resolves a launch.
// It never starts/stops a serving container, builds an image, or publishes.
package main

import (
	"archive/tar"
	"compress/gzip"
	"context"
	"crypto/sha1"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"syscall"
	"time"

	"b70ctl/internal/catalog"
	"b70ctl/internal/hardware"
	"b70ctl/internal/hf"
	"b70ctl/internal/modelpack"
	"b70ctl/internal/modelstore"
	"b70ctl/internal/packstore"
	"b70ctl/internal/runtime"
)

func main() {
	pack := flag.String("pack", "model-packs/Qwen3.8-Flash-Next", "source pack directory")
	out := flag.String("output", "", "new directory for archive, isolated import, reports and draft staging")
	target := flag.String("target", "", "existing accepted target checkpoint, read only")
	seed := flag.String("draft-seed", "", "existing pinned draft checkpoint; hash verified before reuse")
	inspect := flag.Bool("inspect-models", false, "inspect pinned HF inventories and fetch only missing small draft files")
	flag.Parse()
	if err := run(*pack, *out, *target, *seed, *inspect); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
func must(err error) {
	if err != nil {
		panic(err)
	}
}
func digest(path string) string {
	f, err := os.Open(path)
	must(err)
	defer f.Close()
	h := sha256.New()
	_, err = io.Copy(h, f)
	must(err)
	return fmt.Sprintf("sha256:%x", h.Sum(nil))
}
func save(path string, v any) {
	data, err := json.MarshalIndent(v, "", "  ")
	must(err)
	must(os.WriteFile(path, append(data, '\n'), 0644))
}
func names(root string) ([]string, error) {
	result := []string{"README.md", "RUNTIME_RECIPE.md", "pack.json"}
	err := filepath.WalkDir(filepath.Join(root, "patches"), func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			return nil
		}
		info, err := d.Info()
		if err != nil {
			return err
		}
		if !info.Mode().IsRegular() {
			return fmt.Errorf("nonregular pack asset: %s", path)
		}
		rel, err := filepath.Rel(root, path)
		if err != nil {
			return err
		}
		result = append(result, filepath.ToSlash(rel))
		return nil
	})
	sort.Strings(result)
	return result, err
}
func archive(root, destination string, files []string) error {
	output, err := os.Create(destination)
	if err != nil {
		return err
	}
	defer output.Close()
	gz := gzip.NewWriter(output)
	tw := tar.NewWriter(gz)
	for _, name := range files {
		data, err := os.ReadFile(filepath.Join(root, name))
		if err != nil {
			return err
		}
		header := &tar.Header{Name: name, Typeflag: tar.TypeReg, Mode: 0644, Size: int64(len(data)), ModTime: time.Unix(0, 0).UTC(), Format: tar.FormatGNU}
		if err := tw.WriteHeader(header); err != nil {
			return err
		}
		if _, err := tw.Write(data); err != nil {
			return err
		}
	}
	if err := tw.Close(); err != nil {
		return err
	}
	return gz.Close()
}
func sameInventory(model modelpack.Model, repo hf.Repository) bool {
	if model.Repo != repo.Repo || model.Revision != repo.Revision || len(model.Files) != len(repo.Files) {
		return false
	}
	expected := map[string]int64{}
	for _, f := range model.Files {
		if f.Size == nil {
			return false
		}
		expected[f.Path] = *f.Size
	}
	for _, f := range repo.Files {
		n, exists := expected[f.Path]
		if !exists || !f.SizeKnown || n != f.Size {
			return false
		}
	}
	return true
}
func run(pack, out, target, seed string, inspect bool) error {
	if out == "" {
		return fmt.Errorf("--output is required")
	}
	absolute, err := filepath.Abs(out)
	if err != nil {
		return err
	}
	out = absolute
	packRoot, err := filepath.Abs(pack)
	if err != nil {
		return err
	}
	if rel, err := filepath.Rel(packRoot, out); err != nil || rel == "." || (rel != ".." && !strings.HasPrefix(rel, ".."+string(filepath.Separator))) {
		return fmt.Errorf("output must be outside the source pack")
	}
	if _, err := os.Lstat(out); !os.IsNotExist(err) {
		return fmt.Errorf("output must not exist")
	}
	if err := os.MkdirAll(out, 0755); err != nil {
		return err
	}
	manifest, err := modelpack.Load(pack)
	if err != nil {
		return err
	}
	if len(manifest.Models) != 2 || len(manifest.Runtimes) != 1 || manifest.Runtimes[0].Registry != "" {
		return fmt.Errorf("expected local-only Flash-Next target/assistant pack")
	}
	files, err := names(pack)
	if err != nil {
		return err
	}
	archivePath := filepath.Join(out, manifest.ID+"-"+manifest.Version+".tar.gz")
	if err := archive(pack, archivePath, files); err != nil {
		return err
	}
	repeat := filepath.Join(out, "repeat.tar.gz")
	if err := archive(pack, repeat, files); err != nil {
		return err
	}
	if digest(archivePath) != digest(repeat) {
		return fmt.Errorf("archive is not deterministic")
	}
	must(os.Remove(repeat))
	server := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { http.ServeFile(w, r, archivePath) }))
	defer server.Close()
	entry := catalog.Entry{ID: manifest.ID, Name: manifest.Name, Version: manifest.Version, ArchiveURL: server.URL + "/pack.tar.gz", SHA256: digest(archivePath)}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()
	client := catalog.Client{HTTPClient: server.Client()}
	acquired, err := client.Acquire(ctx, filepath.Join(out, "acquired"), entry)
	if err != nil {
		return err
	}
	got, err := names(acquired.Path)
	if err != nil {
		return err
	}
	if !reflect.DeepEqual(files, got) {
		return fmt.Errorf("extracted file set differs")
	}
	for _, name := range files {
		if digest(filepath.Join(pack, name)) != digest(filepath.Join(acquired.Path, name)) {
			return fmt.Errorf("extracted hash differs: %s", name)
		}
	}
	// Retain the relocated acquired pack as evidence, not an application download cache.
	relocated := filepath.Join(out, "extracted-pack")
	must(os.Rename(acquired.Path, relocated))
	must(acquired.Close())
	installed, err := packstore.Import(relocated, filepath.Join(out, "installed-packs"), packstore.SourceLocal)
	if err != nil {
		return err
	}
	installedRoot := filepath.Join(out, "installed-packs", installed.ID, installed.Version)
	loaded, err := modelpack.Load(installedRoot)
	if err != nil {
		return err
	}
	entries, err := os.ReadDir(installedRoot)
	if err != nil {
		return err
	}
	installedFiles := []string{}
	for _, f := range entries {
		installedFiles = append(installedFiles, f.Name())
	}
	report := map[string]any{"manifest": "PASS", "archive": archivePath, "archive_sha256": entry.SHA256, "archive_files": len(files), "deterministic_repeat": true, "local_tls_acquisition": "PASS; exact extracted file set and hashes", "extracted_pack": relocated, "imported_pack": installed, "imported_files": installedFiles, "serving_container_started": false, "public_install_test": false, "publication": false}
	if !inspect {
		save(filepath.Join(out, "results.json"), report)
		return nil
	}
	if target == "" || seed == "" {
		return fmt.Errorf("--target and --draft-seed are required with --inspect-models")
	}
	hub := hf.NewClient("")
	repos := []hf.Repository{}
	for _, model := range loaded.Models {
		repo, err := hub.Inspect(ctx, model.Repo, model.Revision)
		if err != nil {
			return err
		}
		if !sameInventory(model, repo) {
			return fmt.Errorf("pinned HF inventory differs: %s", model.Repo)
		}
		repos = append(repos, repo)
	}
	report["hf_pinned_inventories"] = "PASS: target 29 files, assistant 6 files"
	draft := loaded.Models[1]
	modelRoot := filepath.Join(out, "models")
	staging, err := modelstore.StagingPath(modelRoot, draft.Repo, draft.Revision)
	if err != nil {
		return err
	}
	must(os.MkdirAll(staging, 0755))
	save(filepath.Join(staging, ".b70-staging.json"), map[string]any{"schema": 1, "repo": draft.Repo, "revision": draft.Revision})
	weight := filepath.Join(seed, "model.safetensors")
	if digest(weight) != "sha256:d6723f2a359eaf33ec0c95f22ae8edda896774020c552296a85e1913b4d3186f" {
		return fmt.Errorf("seed weights differ from pinned LFS hash")
	}
	// Reuse only known exact pinned files. Hardlinks avoid copying weights and are
	// self-contained files in the later draft bind, unlike external symlinks.
	for _, name := range []string{"model.safetensors", "README.md", "config.json"} {
		from, to := filepath.Join(seed, name), filepath.Join(staging, name)
		if err := os.Link(from, to); errors.Is(err, syscall.EXDEV) {
			input, err := os.Open(from)
			if err != nil {
				return err
			}
			output, err := os.OpenFile(to, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0644)
			if err != nil {
				input.Close()
				return err
			}
			_, copyErr := io.Copy(output, input)
			input.Close()
			closeErr := output.Close()
			if copyErr != nil {
				return copyErr
			}
			if closeErr != nil {
				return closeErr
			}
		} else if err != nil {
			return err
		}
	}
	last := hf.Progress{}
	download, err := hub.Install(ctx, modelRoot, repos[1], func(p hf.Progress) { last = p })
	if err != nil {
		return err
	}
	report["draft_acquisition"] = download
	report["draft_progress"] = last
	var pinned struct {
		Siblings []struct {
			Path string `json:"rfilename"`
			Blob string `json:"blobId"`
			LFS  *struct {
				SHA string `json:"sha256"`
			} `json:"lfs"`
		} `json:"siblings"`
	}
	data, err := os.ReadFile(filepath.Join(relocated, "patches/production/manifests/draft-pinned-api.json"))
	if err != nil {
		return err
	}
	if err := json.Unmarshal(data, &pinned); err != nil {
		return err
	}
	for _, file := range pinned.Siblings {
		path := filepath.Join(download.Path, file.Path)
		if file.LFS != nil {
			if digest(path) != "sha256:"+file.LFS.SHA {
				return fmt.Errorf("draft LFS hash differs: %s", file.Path)
			}
		} else {
			contents, err := os.ReadFile(path)
			if err != nil {
				return err
			}
			blob := append([]byte(fmt.Sprintf("blob %d\x00", len(contents))), contents...)
			if fmt.Sprintf("%x", sha1.Sum(blob)) != file.Blob {
				return fmt.Errorf("draft Git blob differs: %s", file.Path)
			}
		}
	}
	report["draft_strong_hashes"] = "PASS: all six files match pinned LFS SHA256 or Git blob identities"
	artifacts, err := modelstore.Scan(modelRoot)
	if err != nil {
		return err
	}
	report["discovered_draft"] = artifacts
	targetAbs, err := filepath.Abs(target)
	if err != nil {
		return err
	}
	artifacts = append(artifacts, modelstore.Artifact{Repo: loaded.Models[0].Repo, Revision: loaded.Models[0].Revision, Path: targetAbs})
	report["target_binding"] = "Existing external checkpoint supplied explicitly from accepted evidence; not installed into the user's Controller model store"
	checks := map[string]modelstore.CheckResult{}
	for _, m := range loaded.Models {
		a, ok := modelstore.Find(artifacts, m.Repo, m.Revision)
		if !ok {
			return fmt.Errorf("model not found")
		}
		checks[m.ID] = modelstore.Check(a.Path, m.Files)
		if !checks[m.ID].Complete {
			return fmt.Errorf("incomplete model: %s", m.ID)
		}
	}
	report["model_checks"] = checks
	images := runtime.InspectRuntimes(loaded.Runtimes)
	if images[loaded.Runtimes[0].ID].Status != runtime.ImagePresent {
		return fmt.Errorf("accepted local image missing; no pull attempted")
	}
	acquisition := runtime.Acquire(ctx, loaded.Runtimes[0], nil)
	if acquisition.Outcome != runtime.AcquisitionReused {
		return fmt.Errorf("local runtime reuse failed: %s", acquisition.Reason)
	}
	report["runtime_acquisition"] = acquisition.Outcome
	cards, err := hardware.DetectB70()
	if err != nil {
		return err
	}
	report["b70_devices"] = cards
	availability := runtime.Resolve(loaded, artifacts, images, len(cards))
	report["profile_availability"] = availability
	if len(availability) != 1 || !availability[0].Available {
		return fmt.Errorf("local profile unavailable")
	}
	launch, err := runtime.BuildLaunch(loaded, loaded.Profiles[0], artifacts, runtime.AccessLocal, 18080)
	if err != nil {
		return err
	}
	report["dry_launch"] = launch
	missingAssistant := runtime.Resolve(loaded, artifacts[len(artifacts)-1:], images, len(cards))
	if missingAssistant[0].Available || !strings.Contains(strings.Join(missingAssistant[0].Missing, " "), "assistant") {
		return fmt.Errorf("assistant absence did not block readiness")
	}
	report["missing_assistant_guard"] = "PASS"
	fewer := runtime.Resolve(loaded, artifacts, images, 2)
	if fewer[0].Available {
		return fmt.Errorf("two-card host incorrectly accepted")
	}
	report["four_card_guard"] = "PASS"
	save(filepath.Join(out, "results.json"), report)
	fmt.Println("PASS: archive, local acquisition/import, pinned models, local runtime reuse and dry profile launch; no server started")
	return nil
}
