package hf

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"b70ctl/internal/modelstore"
)

const otherRevision = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"

// fakeHub serves repository files from a table, counts requests per file,
// and can inject failures, gated responses, and hanging partial bodies.
type fakeHub struct {
	mu       sync.Mutex
	requests map[string]int
	contents map[string]string
	failures map[string]int
	gated    map[string]chan struct{}
	hanging  map[string]bool
}

func newFakeHub(contents map[string]string) *fakeHub {
	return &fakeHub{
		requests: map[string]int{},
		contents: contents,
		failures: map[string]int{},
		gated:    map[string]chan struct{}{},
		hanging:  map[string]bool{},
	}
}

func (hub *fakeHub) fail(path string, attempts int) {
	hub.mu.Lock()
	defer hub.mu.Unlock()
	hub.failures[path] = attempts
}

func (hub *fakeHub) gate(path string) chan struct{} {
	release := make(chan struct{})
	hub.mu.Lock()
	defer hub.mu.Unlock()
	hub.gated[path] = release
	return release
}

func (hub *fakeHub) hang(path string) {
	hub.mu.Lock()
	defer hub.mu.Unlock()
	hub.hanging[path] = true
}

func (hub *fakeHub) requestCount(path string) int {
	hub.mu.Lock()
	defer hub.mu.Unlock()
	return hub.requests[path]
}

func (hub *fakeHub) handler(t *testing.T) http.HandlerFunc {
	return func(writer http.ResponseWriter, request *http.Request) {
		t.Helper()
		path := strings.TrimPrefix(request.URL.Path, "/example/model/resolve/"+testRevision+"/")
		hub.mu.Lock()
		hub.requests[path]++
		fail := hub.failures[path]
		if fail > 0 {
			hub.failures[path] = fail - 1
		}
		release, gated := hub.gated[path]
		hanging := hub.hanging[path]
		contents, found := hub.contents[path]
		hub.mu.Unlock()
		if gated {
			<-release
		}
		if fail > 0 {
			writer.WriteHeader(http.StatusBadGateway)
			return
		}
		if !found {
			writer.WriteHeader(http.StatusNotFound)
			return
		}
		writer.Header().Set("X-Repo-Commit", testRevision)
		if hanging {
			io.WriteString(writer, contents[:len(contents)/2])
			if flusher, ok := writer.(http.Flusher); ok {
				flusher.Flush()
			}
			<-request.Context().Done()
			hub.mu.Lock()
			delete(hub.hanging, path)
			hub.mu.Unlock()
			return
		}
		io.WriteString(writer, contents)
	}
}

func (hub *fakeHub) server(t *testing.T) *httptest.Server {
	t.Helper()
	server := httptest.NewServer(hub.handler(t))
	t.Cleanup(server.Close)
	return server
}

func stageTestDirectory(t *testing.T, root, repo, revision string) string {
	t.Helper()
	directory, err := modelstore.StagingPath(root, repo, revision)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(directory, 0o755); err != nil {
		t.Fatal(err)
	}
	return directory
}

func writeStagingTestMarker(t *testing.T, directory, repo, revision string) {
	t.Helper()
	contents := fmt.Sprintf(`{"schema":%d,"repo":%q,"revision":%q}`, stagingMarkerSchema, repo, revision)
	if err := os.WriteFile(filepath.Join(directory, stagingMarkerName), []byte(contents), 0o644); err != nil {
		t.Fatal(err)
	}
}

func threeFileRepository() Repository {
	return Repository{Repo: "example/model", Revision: testRevision, Files: []File{
		{Path: "a.bin", Size: 6, SizeKnown: true},
		{Path: "b.bin", Size: 9, SizeKnown: true},
		{Path: "c.bin", Size: 4, SizeKnown: true},
	}}
}

func stagingDirectory(t *testing.T, root string, repository Repository) string {
	t.Helper()
	directory, err := modelstore.StagingPath(root, repository.Repo, repository.Revision)
	if err != nil {
		t.Fatal(err)
	}
	return directory
}

func TestInstallInterruptedLeavesValidStagingAndEarlierFiles(t *testing.T) {
	root := t.TempDir()
	hub := newFakeHub(map[string]string{"a.bin": "config", "b.bin": "tokenizer", "c.bin": "brew"})
	hub.fail("b.bin", 1)
	repository := threeFileRepository()

	_, err := testClient(hub.server(t), "").Install(context.Background(), root, repository, nil)
	if !errors.Is(err, ErrDownloadFailed) {
		t.Fatalf("Install() error = %v", err)
	}
	staging := stagingDirectory(t, root, repository)
	if contents, err := os.ReadFile(filepath.Join(staging, "a.bin")); err != nil || string(contents) != "config" {
		t.Fatalf("completed staged file a.bin = %q, %v", contents, err)
	}
	if _, err := os.Lstat(filepath.Join(staging, "b.bin")); !os.IsNotExist(err) {
		t.Fatalf("failed file b.bin left a complete-looking artifact: %v", err)
	}
	if _, err := os.Lstat(filepath.Join(staging, "c.bin")); !os.IsNotExist(err) {
		t.Fatalf("file c.bin started before its turn: %v", err)
	}
	if _, err := os.Stat(filepath.Join(staging, stagingMarkerName)); err != nil {
		t.Fatalf("staging marker missing after interruption: %v", err)
	}
	if _, err := os.Lstat(filepath.Join(root, "example__model")); !os.IsNotExist(err) {
		t.Fatalf("final model directory appeared after interruption: %v", err)
	}
}

func TestInstallRetryReusesCompletedFilesAndFetchesOnlyMissing(t *testing.T) {
	root := t.TempDir()
	hub := newFakeHub(map[string]string{"a.bin": "config", "b.bin": "tokenizer", "c.bin": "brew"})
	hub.fail("b.bin", 1)
	repository := threeFileRepository()

	if _, err := testClient(hub.server(t), "").Install(context.Background(), root, repository, nil); !errors.Is(err, ErrDownloadFailed) {
		t.Fatalf("first Install() error = %v", err)
	}
	result, err := testClient(hub.server(t), "").Install(context.Background(), root, repository, nil)
	if err != nil {
		t.Fatalf("retry Install() error = %v", err)
	}
	if result.AlreadyPresent || result.Path != filepath.Join(root, "example__model") {
		t.Fatalf("retry Install() = %#v", result)
	}
	if hub.requestCount("a.bin") != 1 {
		t.Fatalf("a.bin requested %d times, want exactly once across both attempts", hub.requestCount("a.bin"))
	}
	if hub.requestCount("b.bin") != 2 || hub.requestCount("c.bin") != 1 {
		t.Fatalf("request counts b=%d c=%d, want b=2 c=1", hub.requestCount("b.bin"), hub.requestCount("c.bin"))
	}
	if result.BytesDownloaded != 13 {
		t.Fatalf("retry received %d bytes, want only missing files (13)", result.BytesDownloaded)
	}
	artifacts, err := modelstore.Scan(root)
	if err != nil {
		t.Fatal(err)
	}
	if artifact, found := modelstore.Find(artifacts, repository.Repo, repository.Revision); !found || artifact.Path != result.Path {
		t.Fatalf("completed model was not discovered: %#v", artifacts)
	}
	if _, err := os.Lstat(stagingDirectory(t, root, repository)); !os.IsNotExist(err) {
		t.Fatalf("staging directory survived a successful promotion: %v", err)
	}
	t.Logf("request counts after retry: a.bin=%d b.bin=%d c.bin=%d (a.bin reused, not re-fetched)", hub.requestCount("a.bin"), hub.requestCount("b.bin"), hub.requestCount("c.bin"))
}

func TestInstallRetryReplacesWrongSizeStagedFile(t *testing.T) {
	root := t.TempDir()
	repository := threeFileRepository()
	staging := stageTestDirectory(t, root, repository.Repo, repository.Revision)
	writeStagingTestMarker(t, staging, repository.Repo, repository.Revision)
	if err := os.WriteFile(filepath.Join(staging, "a.bin"), []byte("config"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(staging, "b.bin"), []byte("short"), 0o644); err != nil {
		t.Fatal(err)
	}
	hub := newFakeHub(map[string]string{"a.bin": "config", "b.bin": "tokenizer", "c.bin": "brew"})

	result, err := testClient(hub.server(t), "").Install(context.Background(), root, repository, nil)
	if err != nil {
		t.Fatalf("Install() error = %v", err)
	}
	if hub.requestCount("a.bin") != 0 {
		t.Fatalf("correct staged a.bin was re-fetched: %d requests", hub.requestCount("a.bin"))
	}
	if hub.requestCount("b.bin") != 1 || hub.requestCount("c.bin") != 1 {
		t.Fatalf("request counts b=%d c=%d, want b=1 c=1", hub.requestCount("b.bin"), hub.requestCount("c.bin"))
	}
	if contents, err := os.ReadFile(filepath.Join(result.Path, "b.bin")); err != nil || string(contents) != "tokenizer" {
		t.Fatalf("wrong-size b.bin was not replaced: %q, %v", contents, err)
	}
}

func TestInstallUnknownSizeStagedFileIsRedownloaded(t *testing.T) {
	root := t.TempDir()
	repository := Repository{Repo: "example/model", Revision: testRevision, Files: []File{{Path: "a.bin"}}}
	staging := stageTestDirectory(t, root, repository.Repo, repository.Revision)
	writeStagingTestMarker(t, staging, repository.Repo, repository.Revision)
	if err := os.WriteFile(filepath.Join(staging, "a.bin"), []byte("anything"), 0o644); err != nil {
		t.Fatal(err)
	}
	hub := newFakeHub(map[string]string{"a.bin": "fresh"})

	if _, err := testClient(hub.server(t), "").Install(context.Background(), root, repository, nil); err != nil {
		t.Fatalf("Install() error = %v", err)
	}
	if hub.requestCount("a.bin") != 1 {
		t.Fatalf("unverifiable staged file was reused: %d requests", hub.requestCount("a.bin"))
	}
	if contents, err := os.ReadFile(filepath.Join(root, "example__model", "a.bin")); err != nil || string(contents) != "fresh" {
		t.Fatalf("final a.bin = %q, %v", contents, err)
	}
}

func TestInstallPartFileNeverTreatedAsComplete(t *testing.T) {
	root := t.TempDir()
	repository := threeFileRepository()
	staging := stageTestDirectory(t, root, repository.Repo, repository.Revision)
	writeStagingTestMarker(t, staging, repository.Repo, repository.Revision)
	if err := os.WriteFile(filepath.Join(staging, "b.bin"), []byte("config"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(staging, "c.bin"+partSuffix), []byte("par"), 0o644); err != nil {
		t.Fatal(err)
	}
	hub := newFakeHub(map[string]string{"a.bin": "config", "b.bin": "tokenizer", "c.bin": "brew"})

	result, err := testClient(hub.server(t), "").Install(context.Background(), root, repository, nil)
	if err != nil {
		t.Fatalf("Install() error = %v", err)
	}
	if hub.requestCount("c.bin") != 1 {
		t.Fatalf("c.bin with only a .part present was treated as complete: %d requests", hub.requestCount("c.bin"))
	}
	entries, err := os.ReadDir(result.Path)
	if err != nil {
		t.Fatal(err)
	}
	for _, entry := range entries {
		if strings.HasSuffix(entry.Name(), partSuffix) {
			t.Fatalf("partial artifact %s reached the final model", entry.Name())
		}
	}
}

func TestInstallReusedFileCleansStalePartSibling(t *testing.T) {
	root := t.TempDir()
	repository := Repository{Repo: "example/model", Revision: testRevision, Files: []File{{Path: "a.bin", Size: 6, SizeKnown: true}}}
	staging := stageTestDirectory(t, root, repository.Repo, repository.Revision)
	writeStagingTestMarker(t, staging, repository.Repo, repository.Revision)
	if err := os.WriteFile(filepath.Join(staging, "a.bin"), []byte("config"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(staging, "a.bin"+partSuffix), []byte("conf"), 0o644); err != nil {
		t.Fatal(err)
	}
	hub := newFakeHub(map[string]string{"a.bin": "config"})

	result, err := testClient(hub.server(t), "").Install(context.Background(), root, repository, nil)
	if err != nil {
		t.Fatalf("Install() error = %v", err)
	}
	if hub.requestCount("a.bin") != 0 {
		t.Fatalf("complete a.bin was re-fetched: %d requests", hub.requestCount("a.bin"))
	}
	if _, err := os.Lstat(filepath.Join(result.Path, "a.bin"+partSuffix)); !os.IsNotExist(err) {
		t.Fatalf("stale partial sibling reached the final model: %v", err)
	}
}

func TestInstallSuccessConsumesStagingAndKeepsOnlyFinalModel(t *testing.T) {
	root := t.TempDir()
	hub := newFakeHub(map[string]string{"a.bin": "config", "b.bin": "tokenizer", "c.bin": "brew"})

	result, err := testClient(hub.server(t), "").Install(context.Background(), root, threeFileRepository(), nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := os.Lstat(filepath.Join(root, modelstore.StagingDirName)); !os.IsNotExist(err) {
		t.Fatalf("staging root survived promotion: %v", err)
	}
	if _, err := os.Lstat(filepath.Join(result.Path, stagingMarkerName)); !os.IsNotExist(err) {
		t.Fatalf("staging marker leaked into the final model: %v", err)
	}
	if _, err := os.Stat(filepath.Join(result.Path, ".b70-model.json")); err != nil {
		t.Fatalf("final model marker missing: %v", err)
	}
	entries, err := os.ReadDir(root)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 1 || entries[0].Name() != "example__model" {
		t.Fatalf("model root after success = %#v", entries)
	}
}

func TestInstallAlreadyPresentDoesNoNetworkOrStagingWork(t *testing.T) {
	root := t.TempDir()
	destination := filepath.Join(root, "example__model")
	writeTestMarker(t, destination, "example/model", testRevision)
	for path, contents := range map[string]string{"a.bin": "config", "b.bin": "tokenizer", "c.bin": "brew"} {
		if err := os.WriteFile(filepath.Join(destination, path), []byte(contents), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	hub := newFakeHub(map[string]string{"a.bin": "config", "b.bin": "tokenizer", "c.bin": "brew"})

	result, err := testClient(hub.server(t), "").Install(context.Background(), root, threeFileRepository(), nil)
	if err != nil {
		t.Fatal(err)
	}
	if !result.AlreadyPresent || result.Path != destination {
		t.Fatalf("Install() = %#v", result)
	}
	if total := hub.requestCount("a.bin") + hub.requestCount("b.bin") + hub.requestCount("c.bin"); total != 0 {
		t.Fatalf("AlreadyPresent performed %d network requests", total)
	}
	if _, err := os.Lstat(filepath.Join(root, modelstore.StagingDirName)); !os.IsNotExist(err) {
		t.Fatalf("AlreadyPresent created staging state: %v", err)
	}
}

func TestInstallRefusesStagingWithMismatchedProvenance(t *testing.T) {
	tests := []struct {
		name     string
		repo     string
		revision string
	}{
		{name: "same repo other revision", repo: "example/model", revision: otherRevision},
		{name: "other repo same revision", repo: "other/model", revision: testRevision},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			root := t.TempDir()
			repository := threeFileRepository()
			staging := stageTestDirectory(t, root, repository.Repo, repository.Revision)
			writeStagingTestMarker(t, staging, test.repo, test.revision)
			if err := os.WriteFile(filepath.Join(staging, "a.bin"), []byte("foreign"), 0o644); err != nil {
				t.Fatal(err)
			}
			hub := newFakeHub(map[string]string{"a.bin": "config", "b.bin": "tokenizer", "c.bin": "brew"})

			_, err := testClient(hub.server(t), "").Install(context.Background(), root, repository, nil)
			if err == nil || !strings.Contains(err.Error(), "not the requested model") {
				t.Fatalf("Install() error = %v, want provenance refusal", err)
			}
			if contents, readErr := os.ReadFile(filepath.Join(staging, "a.bin")); readErr != nil || string(contents) != "foreign" {
				t.Fatalf("refused staging content was deleted: %q, %v", contents, readErr)
			}
			if _, statErr := os.Lstat(filepath.Join(root, "example__model")); !os.IsNotExist(statErr) {
				t.Fatalf("final model appeared despite refusal: %v", statErr)
			}
		})
	}
}

func TestInstallRefusesMalformedStagingMarker(t *testing.T) {
	tests := []struct {
		name    string
		marker  string
		message string
	}{
		{name: "truncated json", marker: `{"schema":1,`, message: "invalid"},
		{name: "unsupported schema", marker: `{"schema":99,"repo":"example/model","revision":"` + testRevision + `"}`, message: "not supported"},
		{name: "unknown field", marker: `{"schema":1,"repo":"example/model","revision":"` + testRevision + `","evil":true}`, message: "invalid"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			root := t.TempDir()
			repository := threeFileRepository()
			staging := stageTestDirectory(t, root, repository.Repo, repository.Revision)
			if err := os.WriteFile(filepath.Join(staging, stagingMarkerName), []byte(test.marker), 0o644); err != nil {
				t.Fatal(err)
			}
			hub := newFakeHub(map[string]string{"a.bin": "config"})

			_, err := testClient(hub.server(t), "").Install(context.Background(), root, repository, nil)
			if err == nil || !strings.Contains(err.Error(), test.message) {
				t.Fatalf("Install() error = %v, want %q refusal", err, test.message)
			}
			if _, statErr := os.Stat(filepath.Join(staging, stagingMarkerName)); statErr != nil {
				t.Fatalf("refused marker was deleted: %v", statErr)
			}
		})
	}
}

func TestInstallRefusesUnmarkedStagingWithContent(t *testing.T) {
	root := t.TempDir()
	repository := threeFileRepository()
	staging := stageTestDirectory(t, root, repository.Repo, repository.Revision)
	if err := os.WriteFile(filepath.Join(staging, "mystery.bin"), []byte("unknown"), 0o644); err != nil {
		t.Fatal(err)
	}
	hub := newFakeHub(map[string]string{"a.bin": "config"})

	_, err := testClient(hub.server(t), "").Install(context.Background(), root, repository, nil)
	if err == nil || !strings.Contains(err.Error(), "no valid staging marker") {
		t.Fatalf("Install() error = %v, want unmarked-staging refusal", err)
	}
	if _, statErr := os.Stat(filepath.Join(staging, "mystery.bin")); statErr != nil {
		t.Fatalf("unmarked staging content was deleted: %v", statErr)
	}
}

func TestInstallRecoversPromotionCrashWindow(t *testing.T) {
	root := t.TempDir()
	repository := threeFileRepository()
	staging := stageTestDirectory(t, root, repository.Repo, repository.Revision)
	writeStagingTestMarker(t, staging, repository.Repo, repository.Revision)
	// Crash between writing the final model marker and the promotion
	// rename: every file is complete and the final marker already exists.
	for path, contents := range map[string]string{"a.bin": "config", "b.bin": "tokenizer", "c.bin": "brew"} {
		if err := os.WriteFile(filepath.Join(staging, path), []byte(contents), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.WriteFile(filepath.Join(staging, ".b70-model.json"), []byte(`{"repo":"example/model","revision":"`+testRevision+`"}`), 0o644); err != nil {
		t.Fatal(err)
	}
	hub := newFakeHub(map[string]string{"a.bin": "config", "b.bin": "tokenizer", "c.bin": "brew"})

	result, err := testClient(hub.server(t), "").Install(context.Background(), root, repository, nil)
	if err != nil {
		t.Fatalf("Install() error = %v", err)
	}
	if total := hub.requestCount("a.bin") + hub.requestCount("b.bin") + hub.requestCount("c.bin"); total != 0 {
		t.Fatalf("complete crashed staging was re-fetched: %d requests", total)
	}
	if result.Path != filepath.Join(root, "example__model") || result.BytesDownloaded != 0 {
		t.Fatalf("Install() = %#v", result)
	}
	if _, err := os.Stat(filepath.Join(result.Path, "b.bin")); err != nil {
		t.Fatalf("crashed promotion did not complete: %v", err)
	}
}

func TestInstallRecoversFreshStagingCrashWindows(t *testing.T) {
	tests := []struct {
		name  string
		build func(t *testing.T, staging string)
	}{
		{name: "empty staging directory", build: func(t *testing.T, staging string) {}},
		{name: "interrupted marker write", build: func(t *testing.T, staging string) {
			if err := os.WriteFile(filepath.Join(staging, stagingMarkerPartial), []byte(`{"schema"`), 0o644); err != nil {
				t.Fatal(err)
			}
		}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			root := t.TempDir()
			repository := threeFileRepository()
			staging := stageTestDirectory(t, root, repository.Repo, repository.Revision)
			test.build(t, staging)
			hub := newFakeHub(map[string]string{"a.bin": "config", "b.bin": "tokenizer", "c.bin": "brew"})

			result, err := testClient(hub.server(t), "").Install(context.Background(), root, repository, nil)
			if err != nil {
				t.Fatalf("Install() error = %v", err)
			}
			if _, statErr := os.Stat(filepath.Join(result.Path, "a.bin")); statErr != nil {
				t.Fatalf("recovered staging did not complete: %v", statErr)
			}
		})
	}
}

func TestInstallRefusesStagingSymlinkEscape(t *testing.T) {
	escapeTest := func(t *testing.T, place func(t *testing.T, root, outside string) (Repository, string)) {
		root := t.TempDir()
		outside := filepath.Join(root, "outside")
		if err := os.MkdirAll(outside, 0o755); err != nil {
			t.Fatal(err)
		}
		repository, target := place(t, root, outside)
		hub := newFakeHub(map[string]string{"a.bin": "config", "b.bin": "tokenizer", "c.bin": "brew", "nested/a.bin": "config"})

		_, err := testClient(hub.server(t), "").Install(context.Background(), root, repository, nil)
		if err == nil {
			t.Fatal("Install() accepted a staging symlink escape")
		}
		if _, statErr := os.Stat(target); statErr != nil {
			t.Fatalf("symlink target was modified: %v", statErr)
		}
		if _, statErr := os.Lstat(filepath.Join(root, "example__model")); !os.IsNotExist(statErr) {
			t.Fatalf("final model appeared despite refusal: %v", statErr)
		}
	}
	t.Run("staging directory itself", func(t *testing.T) {
		escapeTest(t, func(t *testing.T, root, outside string) (Repository, string) {
			repository := threeFileRepository()
			stagingParent := filepath.Join(root, modelstore.StagingDirName)
			if err := os.MkdirAll(stagingParent, 0o755); err != nil {
				t.Fatal(err)
			}
			staging := stagingDirectory(t, root, repository)
			if err := os.Symlink(outside, staging); err != nil {
				t.Fatal(err)
			}
			sentinel := filepath.Join(outside, "sentinel")
			if err := os.WriteFile(sentinel, []byte("keep"), 0o644); err != nil {
				t.Fatal(err)
			}
			return repository, sentinel
		})
	})
	t.Run("nested parent directory", func(t *testing.T) {
		escapeTest(t, func(t *testing.T, root, outside string) (Repository, string) {
			repository := Repository{Repo: "example/model", Revision: testRevision, Files: []File{
				{Path: "nested/a.bin", Size: 6, SizeKnown: true},
			}}
			staging := stageTestDirectory(t, root, repository.Repo, repository.Revision)
			writeStagingTestMarker(t, staging, repository.Repo, repository.Revision)
			if err := os.Symlink(outside, filepath.Join(staging, "nested")); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(filepath.Join(outside, "sentinel"), []byte("keep"), 0o644); err != nil {
				t.Fatal(err)
			}
			return repository, filepath.Join(outside, "sentinel")
		})
	})
	t.Run("staged file", func(t *testing.T) {
		escapeTest(t, func(t *testing.T, root, outside string) (Repository, string) {
			repository := threeFileRepository()
			staging := stageTestDirectory(t, root, repository.Repo, repository.Revision)
			writeStagingTestMarker(t, staging, repository.Repo, repository.Revision)
			target := filepath.Join(outside, "model.bin")
			if err := os.WriteFile(target, []byte("precious"), 0o644); err != nil {
				t.Fatal(err)
			}
			if err := os.Symlink(target, filepath.Join(staging, "a.bin")); err != nil {
				t.Fatal(err)
			}
			return repository, target
		})
	})
}

func TestInstallConcurrentAttemptsAreExclusive(t *testing.T) {
	root := t.TempDir()
	hub := newFakeHub(map[string]string{"a.bin": "config", "b.bin": "tokenizer", "c.bin": "brew"})
	release := hub.gate("a.bin")
	repository := threeFileRepository()
	type outcome struct {
		result Result
		err    error
	}
	finished := make(chan outcome, 1)

	go func() {
		result, err := testClient(hub.server(t), "").Install(context.Background(), root, repository, nil)
		finished <- outcome{result: result, err: err}
	}()
	deadline := time.Now().Add(5 * time.Second)
	for hub.requestCount("a.bin") == 0 && time.Now().Before(deadline) {
		time.Sleep(5 * time.Millisecond)
	}
	if hub.requestCount("a.bin") == 0 {
		t.Fatal("first attempt never reached the gated file")
	}

	_, err := testClient(hub.server(t), "").Install(context.Background(), root, repository, nil)
	if !errors.Is(err, ErrDownloadInProgress) {
		t.Fatalf("concurrent Install() error = %v, want ErrDownloadInProgress", err)
	}
	close(release)
	first := <-finished
	if first.err != nil {
		t.Fatalf("first attempt failed: %v", first.err)
	}
	if _, err := os.Stat(filepath.Join(first.result.Path, "c.bin")); err != nil {
		t.Fatalf("first attempt did not complete after release: %v", err)
	}
}

func TestAcquireStagingAreaLockIsExclusiveUntilReleased(t *testing.T) {
	root := t.TempDir()
	repository := threeFileRepository()
	first, err := acquireStagingArea(root, repository)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := acquireStagingArea(root, repository); !errors.Is(err, ErrDownloadInProgress) {
		t.Fatalf("second acquire error = %v, want ErrDownloadInProgress", err)
	}
	first.release()
	second, err := acquireStagingArea(root, repository)
	if err != nil {
		t.Fatalf("acquire after release error = %v", err)
	}
	second.release()
}

func TestInstallCancellationPreservesReusuableStagedState(t *testing.T) {
	root := t.TempDir()
	hub := newFakeHub(map[string]string{"a.bin": "config", "b.bin": "tokenizer", "c.bin": "brew"})
	hub.hang("b.bin")
	repository := threeFileRepository()

	ctx, cancel := context.WithTimeout(context.Background(), 300*time.Millisecond)
	defer cancel()
	if _, err := testClient(hub.server(t), "").Install(ctx, root, repository, nil); err == nil {
		t.Fatal("cancelled Install() reported success")
	}
	staging := stagingDirectory(t, root, repository)
	if contents, err := os.ReadFile(filepath.Join(staging, "a.bin")); err != nil || string(contents) != "config" {
		t.Fatalf("completed staged file lost after cancellation: %q, %v", contents, err)
	}
	if _, err := os.Lstat(filepath.Join(staging, "b.bin")); !os.IsNotExist(err) {
		t.Fatalf("interrupted b.bin looks complete: %v", err)
	}

	result, err := testClient(hub.server(t), "").Install(context.Background(), root, repository, nil)
	if err != nil {
		t.Fatalf("retry after cancellation error = %v", err)
	}
	if hub.requestCount("a.bin") != 1 {
		t.Fatalf("a.bin requested %d times across cancellation and retry, want once", hub.requestCount("a.bin"))
	}
	if _, err := os.Stat(filepath.Join(result.Path, "b.bin")); err != nil {
		t.Fatalf("retry did not complete b.bin: %v", err)
	}
}

func TestReplaceFlowResumesThroughSameDownloader(t *testing.T) {
	root := t.TempDir()
	hub := newFakeHub(map[string]string{"a.bin": "config", "b.bin": "tokenizer", "c.bin": "brew"})
	hub.fail("b.bin", 1)
	repository := threeFileRepository()

	if _, err := testClient(hub.server(t), "").Install(context.Background(), root, repository, nil); !errors.Is(err, ErrDownloadFailed) {
		t.Fatalf("first Install() error = %v", err)
	}
	destination := filepath.Join(root, "example__model")
	writeTestMarker(t, destination, repository.Repo, repository.Revision)
	if err := os.WriteFile(filepath.Join(destination, "b.bin"), []byte("corrupt"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := testClient(hub.server(t), "").Install(context.Background(), root, repository, nil); !errors.Is(err, ErrDownloadedIncomplete) {
		t.Fatalf("Install() over incomplete final model error = %v", err)
	}

	status, err := modelstore.Remove(root, repository.Repo, repository.Revision)
	if err != nil || status != modelstore.RemovalRemoved {
		t.Fatalf("replace clear step = %q, %v", status, err)
	}
	result, err := testClient(hub.server(t), "").Install(context.Background(), root, repository, nil)
	if err != nil {
		t.Fatalf("replace Install() error = %v", err)
	}
	if hub.requestCount("a.bin") != 1 {
		t.Fatalf("a.bin requested %d times, want once across the whole replace flow", hub.requestCount("a.bin"))
	}
	if _, err := os.Stat(filepath.Join(result.Path, "c.bin")); err != nil {
		t.Fatalf("replace did not complete the model: %v", err)
	}
}

func TestInstallProgressWithReusedFilesStaysWithinTotal(t *testing.T) {
	root := t.TempDir()
	hub := newFakeHub(map[string]string{"a.bin": "config", "b.bin": "tokenizer", "c.bin": "brew"})
	hub.fail("b.bin", 1)
	repository := threeFileRepository()
	if _, err := testClient(hub.server(t), "").Install(context.Background(), root, repository, nil); !errors.Is(err, ErrDownloadFailed) {
		t.Fatalf("first Install() error = %v", err)
	}

	var progress []Progress
	if _, err := testClient(hub.server(t), "").Install(context.Background(), root, repository, func(value Progress) {
		progress = append(progress, value)
	}); err != nil {
		t.Fatal(err)
	}
	if len(progress) == 0 {
		t.Fatal("retry emitted no progress")
	}
	previous := int64(-1)
	for _, value := range progress {
		if value.BytesDownloaded > value.TotalBytes {
			t.Fatalf("progress %#v exceeds the total", value)
		}
		if value.BytesDownloaded < previous {
			t.Fatalf("progress went backwards: %d after %d", value.BytesDownloaded, previous)
		}
		previous = value.BytesDownloaded
	}
	last := progress[len(progress)-1]
	if last.BytesDownloaded != 19 || last.TotalBytes != 19 || last.FileNumber != 3 || last.ReusedFiles != 1 {
		t.Fatalf("final progress = %#v", last)
	}
}

func TestInstallRetryReachesPresentWithoutFurtherNetwork(t *testing.T) {
	root := t.TempDir()
	hub := newFakeHub(map[string]string{"a.bin": "config", "b.bin": "tokenizer", "c.bin": "brew"})
	hub.fail("b.bin", 1)
	repository := threeFileRepository()
	if _, err := testClient(hub.server(t), "").Install(context.Background(), root, repository, nil); !errors.Is(err, ErrDownloadFailed) {
		t.Fatalf("first Install() error = %v", err)
	}
	if _, err := testClient(hub.server(t), "").Install(context.Background(), root, repository, nil); err != nil {
		t.Fatalf("retry Install() error = %v", err)
	}
	result, err := testClient(hub.server(t), "").Install(context.Background(), root, repository, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !result.AlreadyPresent || result.Path != filepath.Join(root, "example__model") {
		t.Fatalf("post-retry Install() = %#v, want AlreadyPresent", result)
	}
	if hub.requestCount("a.bin") != 1 || hub.requestCount("b.bin") != 2 || hub.requestCount("c.bin") != 1 {
		t.Fatalf("unexpected request totals: a=%d b=%d c=%d", hub.requestCount("a.bin"), hub.requestCount("b.bin"), hub.requestCount("c.bin"))
	}
}
