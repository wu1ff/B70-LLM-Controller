package hf

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"

	"b70ctl/internal/modelpack"
	"b70ctl/internal/modelstore"
)

const testRevision = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"

func TestInspectExactRevisionAndAuthorization(t *testing.T) {
	var authorization string
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		authorization = request.Header.Get("Authorization")
		if request.URL.Path != "/api/models/example/model/revision/"+testRevision || request.URL.Query().Get("blobs") != "true" {
			t.Fatalf("request URL = %s", request.URL.String())
		}
		fmt.Fprintf(writer, `{"sha":%q,"siblings":[{"rfilename":"config.json","size":12},{"rfilename":"nested/tokenizer.json"}]}`, testRevision)
	}))
	defer server.Close()

	client := testClient(server, "secret-token")
	repository, err := client.Inspect(context.Background(), "example/model", testRevision)
	if err != nil {
		t.Fatal(err)
	}
	want := Repository{Repo: "example/model", Revision: testRevision, Files: []File{
		{Path: "config.json", Size: 12, SizeKnown: true},
		{Path: "nested/tokenizer.json"},
	}}
	if !reflect.DeepEqual(repository, want) {
		t.Fatalf("Inspect() = %#v, want %#v", repository, want)
	}
	if authorization != "Bearer secret-token" {
		t.Fatalf("Authorization = %q", authorization)
	}
	if _, known := repository.TotalSize(); known {
		t.Fatal("TotalSize() reported an unknown total as known")
	}
}

func TestInspectPublicRequestHasNoAuthorization(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		if value := request.Header.Get("Authorization"); value != "" {
			t.Fatalf("Authorization = %q", value)
		}
		fmt.Fprintf(writer, `{"sha":%q,"siblings":[]}`, testRevision)
	}))
	defer server.Close()
	if _, err := testClient(server, "").Inspect(context.Background(), "example/model", testRevision); err != nil {
		t.Fatal(err)
	}
}

func TestInspectRepositoryAndRevisionFailures(t *testing.T) {
	tests := []struct {
		name      string
		errorCode string
		token     string
		want      error
	}{
		{name: "repository with token", errorCode: "RepoNotFound", token: "token", want: ErrRepositoryNotFound},
		{name: "revision with token", errorCode: "RevisionNotFound", token: "token", want: ErrRevisionNotFound},
		{name: "revision without token", errorCode: "RevisionNotFound", want: ErrAccessUncertain},
		{name: "repository without token", errorCode: "RepoNotFound", want: ErrAccessUncertain},
		{name: "bare not-found without token", want: ErrAccessUncertain},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
				if test.errorCode != "" {
					writer.Header().Set("X-Error-Code", test.errorCode)
				}
				writer.WriteHeader(http.StatusNotFound)
			}))
			defer server.Close()
			_, err := testClient(server, test.token).Inspect(context.Background(), "example/model", testRevision)
			if !errors.Is(err, test.want) {
				t.Fatalf("Inspect() error = %v, want %v", err, test.want)
			}
		})
	}
}

func TestInspectErrorsNeverContainTheToken(t *testing.T) {
	const token = "hf_secret-token-value_DoNotLeak"
	statuses := []int{http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound, http.StatusInternalServerError}
	calls := 0
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		status := statuses[calls%len(statuses)]
		calls++
		if status == http.StatusNotFound {
			writer.Header().Set("X-Error-Code", "RevisionNotFound")
		}
		writer.WriteHeader(status)
		fmt.Fprint(writer, "server body")
	}))
	defer server.Close()
	client := testClient(server, token)
	for _, status := range statuses {
		_, err := client.Inspect(context.Background(), "example/model", testRevision)
		if err == nil {
			t.Fatalf("HTTP %d: Inspect() succeeded unexpectedly", status)
		}
		if message := err.Error(); strings.Contains(message, token) {
			t.Fatalf("HTTP %d: error leaks the token: %q", status, message)
		}
	}
}

func TestInspectAccessFailures(t *testing.T) {
	tests := []struct {
		name   string
		status int
		token  string
		want   error
	}{
		{name: "401 anonymous", status: http.StatusUnauthorized, want: ErrAuthenticationRequired},
		{name: "401 token", status: http.StatusUnauthorized, token: "bad-token", want: ErrAccessDenied},
		{name: "403", status: http.StatusForbidden, token: "token", want: ErrAccessDenied},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
				writer.WriteHeader(test.status)
				fmt.Fprint(writer, "body must not be exposed")
			}))
			defer server.Close()
			_, err := testClient(server, test.token).Inspect(context.Background(), "example/model", testRevision)
			if !errors.Is(err, test.want) || strings.Contains(fmt.Sprint(err), "body must not be exposed") {
				t.Fatalf("Inspect() error = %v, want %v", err, test.want)
			}
		})
	}
}

func TestInspectRejectsResolvedRevisionMismatch(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		fmt.Fprint(writer, `{"sha":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","siblings":[]}`)
	}))
	defer server.Close()
	_, err := testClient(server, "").Inspect(context.Background(), "example/model", testRevision)
	if !errors.Is(err, ErrRevisionNotFound) {
		t.Fatalf("Inspect() error = %v", err)
	}
}

func TestInspectRejectsUnsafeRemotePath(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		fmt.Fprintf(writer, `{"sha":%q,"siblings":[{"rfilename":"../escape","size":4}]}`, testRevision)
	}))
	defer server.Close()
	_, err := testClient(server, "").Inspect(context.Background(), "example/model", testRevision)
	if err == nil || !strings.Contains(err.Error(), "unsafe") {
		t.Fatalf("Inspect() error = %v", err)
	}
}

func TestInstallWritesNestedFilesMarkerLastAndReportsProgress(t *testing.T) {
	root := t.TempDir()
	files := map[string]string{
		"config.json":           "config",
		"nested/tokenizer.json": "tokenizer",
	}
	var markerSeenDuringDownload bool
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		stagingRoot := filepath.Join(root, ".b70ctl-staging")
		entries, _ := os.ReadDir(stagingRoot)
		for _, entry := range entries {
			if !strings.HasSuffix(entry.Name(), "-"+testRevision) {
				continue
			}
			if _, err := os.Stat(filepath.Join(stagingRoot, entry.Name(), ".b70-model.json")); err == nil {
				markerSeenDuringDownload = true
			}
		}
		prefix := "/example/model/resolve/" + testRevision + "/"
		contents, found := files[strings.TrimPrefix(request.URL.Path, prefix)]
		if !found {
			writer.WriteHeader(http.StatusNotFound)
			return
		}
		writer.Header().Set("X-Repo-Commit", testRevision)
		fmt.Fprint(writer, contents)
	}))
	defer server.Close()
	repository := Repository{Repo: "example/model", Revision: testRevision, Files: []File{
		{Path: "config.json", Size: 6, SizeKnown: true},
		{Path: "nested/tokenizer.json", Size: 9, SizeKnown: true},
	}}
	var progress []Progress
	result, err := testClient(server, "").Install(context.Background(), root, repository, func(value Progress) {
		progress = append(progress, value)
	})
	if err != nil {
		t.Fatal(err)
	}
	if markerSeenDuringDownload {
		t.Fatal("model marker existed before downloads completed")
	}
	if result.Path != filepath.Join(root, "example__model") || result.BytesDownloaded != 15 || result.AlreadyPresent {
		t.Fatalf("Install() = %#v", result)
	}
	for path, want := range files {
		contents, err := os.ReadFile(filepath.Join(result.Path, filepath.FromSlash(path)))
		if err != nil || string(contents) != want {
			t.Fatalf("downloaded %s = %q, %v", path, contents, err)
		}
	}
	markerContents, err := os.ReadFile(filepath.Join(result.Path, ".b70-model.json"))
	if err != nil {
		t.Fatal(err)
	}
	var marker struct {
		Repo     string `json:"repo"`
		Revision string `json:"revision"`
	}
	if err := json.Unmarshal(markerContents, &marker); err != nil || marker.Repo != repository.Repo || marker.Revision != repository.Revision {
		t.Fatalf("marker = %#v, %v", marker, err)
	}
	artifacts, err := modelstore.Scan(root)
	if err != nil {
		t.Fatal(err)
	}
	if artifact, found := modelstore.Find(artifacts, repository.Repo, repository.Revision); !found || artifact.Path != result.Path {
		t.Fatalf("downloaded model was not rediscovered: %#v", artifacts)
	}
	last := progress[len(progress)-1]
	if last.CurrentFile != "nested/tokenizer.json" || last.FileNumber != 2 || last.TotalFiles != 2 || last.BytesDownloaded != 15 || last.TotalBytes != 15 || !last.TotalKnown {
		t.Fatalf("last progress = %#v", last)
	}
	var firstFile []int64
	var secondFileStart int64 = -1
	for _, value := range progress {
		if value.CurrentFile == "config.json" {
			firstFile = append(firstFile, value.BytesDownloaded)
		}
		if value.CurrentFile == "nested/tokenizer.json" && secondFileStart < 0 {
			secondFileStart = value.BytesDownloaded
		}
	}
	if len(firstFile) < 2 || firstFile[0] != 0 || firstFile[len(firstFile)-1] != 6 {
		t.Fatalf("first-file byte progress = %v", firstFile)
	}
	if secondFileStart != 6 {
		t.Fatalf("second file started at %d bytes, want prior-file total 6", secondFileStart)
	}
}

func TestInstallSendsAuthorization(t *testing.T) {
	root := t.TempDir()
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		if value := request.Header.Get("Authorization"); value != "Bearer secret-token" {
			t.Fatalf("Authorization = %q", value)
		}
		writer.Header().Set("X-Repo-Commit", testRevision)
		fmt.Fprint(writer, "model")
	}))
	defer server.Close()
	repository := Repository{Repo: "example/model", Revision: testRevision, Files: []File{{Path: "model.bin", Size: 5, SizeKnown: true}}}
	if _, err := testClient(server, "secret-token").Install(context.Background(), root, repository, nil); err != nil {
		t.Fatal(err)
	}
}

func TestFailedDownloadLeavesNoFinalModelButRetainsStaging(t *testing.T) {
	root := t.TempDir()
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		writer.WriteHeader(http.StatusBadGateway)
	}))
	defer server.Close()
	repository := Repository{Repo: "example/model", Revision: testRevision, Files: []File{{Path: "model.bin"}}}
	_, err := testClient(server, "").Install(context.Background(), root, repository, nil)
	if !errors.Is(err, ErrDownloadFailed) {
		t.Fatalf("Install() error = %v", err)
	}
	if _, err := os.Lstat(filepath.Join(root, "example__model")); !os.IsNotExist(err) {
		t.Fatalf("final model appeared after failure: %v", err)
	}
	staging, err := modelstore.StagingPath(root, repository.Repo, repository.Revision)
	if err != nil {
		t.Fatal(err)
	}
	if info, err := os.Lstat(staging); err != nil || !info.IsDir() {
		t.Fatalf("resumable staging was not retained: %v", err)
	}
	if _, err := os.Stat(filepath.Join(staging, ".b70-staging.json")); err != nil {
		t.Fatalf("staging marker was not retained: %v", err)
	}
}

func TestExistingExactModelIsReused(t *testing.T) {
	root := t.TempDir()
	destination := filepath.Join(root, "existing")
	writeTestMarker(t, destination, "example/model", testRevision)
	if err := os.WriteFile(filepath.Join(destination, "model.bin"), []byte("model"), 0o644); err != nil {
		t.Fatal(err)
	}
	repository := Repository{Repo: "example/model", Revision: testRevision, Files: []File{{Path: "model.bin"}}}
	result, err := (&Client{}).Install(context.Background(), root, repository, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !result.AlreadyPresent || result.Path != destination {
		t.Fatalf("Install() = %#v", result)
	}
}

func TestExistingExactIncompleteModelIsRejected(t *testing.T) {
	root := t.TempDir()
	writeTestMarker(t, filepath.Join(root, "existing"), "example/model", testRevision)
	repository := Repository{Repo: "example/model", Revision: testRevision, Files: []File{{Path: "model.bin"}}}
	if _, err := (&Client{}).Install(context.Background(), root, repository, nil); !errors.Is(err, ErrDownloadedIncomplete) {
		t.Fatalf("Install() error = %v", err)
	}
}

func TestMismatchingDestinationIsPreserved(t *testing.T) {
	root := t.TempDir()
	destination := filepath.Join(root, "example__model")
	if err := os.MkdirAll(destination, 0o755); err != nil {
		t.Fatal(err)
	}
	keep := filepath.Join(destination, "keep.txt")
	if err := os.WriteFile(keep, []byte("untouched"), 0o644); err != nil {
		t.Fatal(err)
	}
	repository := Repository{Repo: "example/model", Revision: testRevision}
	_, err := (&Client{}).Install(context.Background(), root, repository, nil)
	if !errors.Is(err, ErrDestinationExists) {
		t.Fatalf("Install() error = %v", err)
	}
	contents, readErr := os.ReadFile(keep)
	if readErr != nil || string(contents) != "untouched" {
		t.Fatalf("preserved file = %q, %v", contents, readErr)
	}
}

func TestLiveTinyPublicDownload(t *testing.T) {
	if os.Getenv("B70_LIVE_HF_TEST") == "" {
		t.Skip("set B70_LIVE_HF_TEST=1 to run the live Hub smoke test")
	}
	const repo = "hf-internal-testing/tiny-random-gpt2"
	const revision = "71034c5d8bde858ff824298bdedc65515b97d2b9"
	client := NewClient("")
	repository, err := client.Inspect(context.Background(), repo, revision)
	if err != nil {
		t.Fatal(err)
	}
	root := t.TempDir()
	result, err := client.Install(context.Background(), root, repository, nil)
	if err != nil {
		t.Fatal(err)
	}
	artifacts, err := modelstore.Scan(root)
	if err != nil {
		t.Fatal(err)
	}
	if _, found := modelstore.Find(artifacts, repo, revision); !found {
		t.Fatal("live downloaded model was not discovered")
	}
	t.Logf("downloaded %s at %s: %d bytes", repo, revision, result.BytesDownloaded)
}

func TestLiveCurrentQwenPackInventoryMatchesHF(t *testing.T) {
	if os.Getenv("B70_LIVE_HF_INVENTORY_TEST") == "" {
		t.Skip("set B70_LIVE_HF_INVENTORY_TEST=1 to run the exact-revision inventory comparison")
	}
	manifest, err := modelpack.Load(filepath.Join("..", "..", "model-packs", "Qwen3.8-27B"))
	if err != nil {
		t.Fatal(err)
	}
	client := NewClient("")
	for _, model := range manifest.Models {
		repository, err := client.Inspect(context.Background(), model.Repo, model.Revision)
		if err != nil {
			t.Fatal(err)
		}
		remote := repositoryExpectedFiles(repository)
		frozen := append([]modelpack.ModelFile(nil), model.Files...)
		if !sort.SliceIsSorted(frozen, func(i, j int) bool { return frozen[i].Path < frozen[j].Path }) {
			t.Errorf("%s frozen inventory is not in lexical path order", model.Repo)
		}
		sort.Slice(remote, func(i, j int) bool { return remote[i].Path < remote[j].Path })
		sort.Slice(frozen, func(i, j int) bool { return frozen[i].Path < frozen[j].Path })
		if !reflect.DeepEqual(remote, frozen) {
			t.Errorf("%s frozen inventory does not match exact-revision metadata", model.Repo)
		}
		t.Logf("%s files=%d known_bytes=%d exact_match=%t", model.Repo, len(frozen), knownBytes(frozen), reflect.DeepEqual(remote, frozen))
	}
}

func knownBytes(files []modelpack.ModelFile) int64 {
	var total int64
	for _, file := range files {
		if file.Size != nil {
			total += *file.Size
		}
	}
	return total
}

func testClient(server *httptest.Server, token string) *Client {
	return &Client{baseURL: server.URL, http: server.Client(), token: token}
}

func writeTestMarker(t *testing.T, directory, repo, revision string) {
	t.Helper()
	if err := os.MkdirAll(directory, 0o755); err != nil {
		t.Fatal(err)
	}
	contents := fmt.Sprintf(`{"repo":%q,"revision":%q}`, repo, revision)
	if err := os.WriteFile(filepath.Join(directory, ".b70-model.json"), []byte(contents), 0o644); err != nil {
		t.Fatal(err)
	}
}
