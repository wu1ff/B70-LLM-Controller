package hf

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	pathpkg "path"
	"path/filepath"
	"strings"
	"time"

	"b70ctl/internal/modelpack"
	"b70ctl/internal/modelstore"
)

var (
	ErrRepositoryNotFound     = errors.New("repository not found")
	ErrRevisionNotFound       = errors.New("revision not found")
	ErrAuthenticationRequired = errors.New("authentication required")
	ErrAccessDenied           = errors.New("access denied")
	ErrAccessUncertain        = errors.New("model access could not be verified without a Hugging Face token")
	ErrDownloadFailed         = errors.New("download failed")
	ErrDestinationExists      = errors.New("destination already exists")
	ErrNetworkUnavailable     = errors.New("network unavailable")
	ErrDownloadedIncomplete   = errors.New("downloaded model is incomplete")
	ErrDownloadInProgress     = errors.New("model download already in progress")
)

// IsAccessFailure reports whether an error means Hugging Face refused or
// could not verify access to a repository: no token is configured, the
// configured token was rejected, or an unauthenticated request received a
// not-found-shaped answer that Hugging Face also gives for gated, private,
// and hidden repositories.
func IsAccessFailure(err error) bool {
	return errors.Is(err, ErrAuthenticationRequired) ||
		errors.Is(err, ErrAccessDenied) ||
		errors.Is(err, ErrAccessUncertain)
}

// LegacyDownloadPrefix is the throwaway temporary-directory prefix used by
// older Controller downloads. Current downloads stage resumably under
// modelstore.StagingDirName instead; startup cleanup sweeps leftovers with
// this prefix.
const LegacyDownloadPrefix = ".b70ctl-download-"

type File struct {
	Path      string
	Size      int64
	SizeKnown bool
}

type Repository struct {
	Repo     string
	Revision string
	Files    []File
}

type Progress struct {
	CurrentFile     string
	FileNumber      int
	TotalFiles      int
	BytesDownloaded int64
	TotalBytes      int64
	TotalKnown      bool
	ReusedFiles     int
}

type Result struct {
	Path            string
	AlreadyPresent  bool
	BytesDownloaded int64
}

type Client struct {
	baseURL string
	http    *http.Client
	token   string
}

func NewClient(token string) *Client {
	const baseURL = "https://huggingface.co"
	return &Client{baseURL: baseURL, http: hubHTTPClient(baseURL), token: strings.TrimSpace(token)}
}

func (repository Repository) TotalSize() (int64, bool) {
	var total int64
	for _, file := range repository.Files {
		if !file.SizeKnown {
			return 0, false
		}
		total += file.Size
	}
	return total, true
}

func (client *Client) Inspect(ctx context.Context, repo, revision string) (Repository, error) {
	escapedRepo, err := escapeRepo(repo)
	if err != nil {
		return Repository{}, err
	}
	if !fullRevision(revision) {
		return Repository{}, errors.New("revision must be a full commit SHA")
	}
	requestURL := client.baseURL + "/api/models/" + escapedRepo + "/revision/" + url.PathEscape(revision) + "?blobs=true"
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, requestURL, nil)
	if err != nil {
		return Repository{}, fmt.Errorf("inspect repository: %w", err)
	}
	client.authorize(request)
	response, err := client.http.Do(request)
	if err != nil {
		return Repository{}, fmt.Errorf("%w: %v", ErrNetworkUnavailable, err)
	}
	defer response.Body.Close()
	if err := client.metadataStatus(response); err != nil {
		return Repository{}, err
	}

	var metadata struct {
		SHA      string `json:"sha"`
		Siblings []struct {
			Path string `json:"rfilename"`
			Size *int64 `json:"size"`
		} `json:"siblings"`
	}
	if err := json.NewDecoder(response.Body).Decode(&metadata); err != nil {
		return Repository{}, fmt.Errorf("inspect repository: %w", err)
	}
	if metadata.SHA != revision {
		return Repository{}, fmt.Errorf("%w: resolved commit does not match the required revision", ErrRevisionNotFound)
	}
	repository := Repository{Repo: repo, Revision: revision, Files: make([]File, 0, len(metadata.Siblings))}
	for _, remote := range metadata.Siblings {
		if err := validateRemotePath(remote.Path); err != nil {
			return Repository{}, err
		}
		file := File{Path: remote.Path}
		if remote.Size != nil && *remote.Size >= 0 {
			file.Size = *remote.Size
			file.SizeKnown = true
		}
		repository.Files = append(repository.Files, file)
	}
	return repository, nil
}

func (client *Client) Install(ctx context.Context, modelRoot string, repository Repository, progress func(Progress)) (Result, error) {
	if _, err := escapeRepo(repository.Repo); err != nil {
		return Result{}, err
	}
	if !fullRevision(repository.Revision) {
		return Result{}, errors.New("revision must be a full commit SHA")
	}
	for _, file := range repository.Files {
		if err := validateRemotePath(file.Path); err != nil {
			return Result{}, err
		}
	}
	root, err := filepath.Abs(modelRoot)
	if err != nil {
		return Result{}, fmt.Errorf("resolve model root: %w", err)
	}
	root = filepath.Clean(root)
	if result, found, err := presentResult(root, repository); err != nil {
		return Result{}, err
	} else if found {
		return result, nil
	}
	if err := os.MkdirAll(root, 0o755); err != nil {
		return Result{}, fmt.Errorf("create model directory: %w", err)
	}
	destination := filepath.Join(root, modelstore.DestinationName(repository.Repo))
	if _, err := os.Lstat(destination); err == nil {
		return Result{}, ErrDestinationExists
	} else if !os.IsNotExist(err) {
		return Result{}, fmt.Errorf("check model destination: %w", err)
	}

	staging, err := acquireStagingArea(root, repository)
	if err != nil {
		return Result{}, err
	}
	defer staging.release()

	// Another attempt may have finished this exact model while we were
	// reaching the lock; re-check before transferring anything.
	if result, found, err := presentResult(root, repository); err != nil {
		return Result{}, err
	} else if found {
		return result, nil
	}

	total, totalKnown := repository.TotalSize()
	var downloaded, reusedBytes int64
	reusedFiles := 0
	for index, file := range repository.Files {
		target, err := stagingFileTarget(staging.dir, file.Path)
		if err != nil {
			return Result{}, err
		}
		if err := ensureStagingParents(staging.dir, target); err != nil {
			return Result{}, err
		}
		state := Progress{
			CurrentFile: file.Path, FileNumber: index + 1, TotalFiles: len(repository.Files),
			BytesDownloaded: reusedBytes + downloaded, TotalBytes: total, TotalKnown: totalKnown,
			ReusedFiles: reusedFiles,
		}
		reuse := false
		if info, err := os.Lstat(target); err == nil {
			if !info.Mode().IsRegular() {
				return Result{}, fmt.Errorf("%w: staged %s is not a regular file", ErrDownloadFailed, file.Path)
			}
			if file.SizeKnown && info.Size() == file.Size {
				reuse = true
			} else if err := os.Remove(target); err != nil {
				return Result{}, fmt.Errorf("%w: replace staged %s: %v", ErrDownloadFailed, file.Path, err)
			}
		} else if !os.IsNotExist(err) {
			return Result{}, fmt.Errorf("%w: check staged %s: %v", ErrDownloadFailed, file.Path, err)
		}
		if reuse {
			if file.SizeKnown {
				reusedBytes += file.Size
			}
			reusedFiles++
			if err := clearStalePartFile(target + partSuffix); err != nil {
				return Result{}, err
			}
			state.BytesDownloaded = reusedBytes + downloaded
			state.ReusedFiles = reusedFiles
			if progress != nil {
				progress(state)
			}
			continue
		}
		part := target + partSuffix
		if err := clearStalePartFile(part); err != nil {
			return Result{}, err
		}
		if progress != nil {
			progress(state)
		}
		written, err := client.downloadFile(ctx, repository, file, part, func(count int64) {
			downloaded += count
			state.BytesDownloaded = reusedBytes + downloaded
			if progress != nil {
				progress(state)
			}
		})
		if err != nil {
			return Result{}, err
		}
		if file.SizeKnown && written != file.Size {
			return Result{}, fmt.Errorf("%w: %s has an unexpected size", ErrDownloadFailed, file.Path)
		}
		if err := os.Rename(part, target); err != nil {
			return Result{}, fmt.Errorf("%w: finalize %s: %v", ErrDownloadFailed, file.Path, err)
		}
	}
	if !modelstore.Check(staging.dir, repositoryExpectedFiles(repository)).Complete {
		return Result{}, ErrDownloadedIncomplete
	}
	if err := writeMarker(filepath.Join(staging.dir, ".b70-model.json"), repository); err != nil {
		return Result{}, err
	}
	if _, err := os.Lstat(destination); err == nil {
		return Result{}, ErrDestinationExists
	} else if !os.IsNotExist(err) {
		return Result{}, fmt.Errorf("check model destination: %w", err)
	}
	if err := os.Rename(staging.dir, destination); err != nil {
		return Result{}, fmt.Errorf("install model: %w", err)
	}
	staging.consumed()
	_ = os.Remove(filepath.Join(destination, stagingMarkerName))
	artifacts, err := modelstore.Scan(root)
	if err != nil {
		os.RemoveAll(destination)
		return Result{}, fmt.Errorf("verify installed model: %w", err)
	}
	artifact, found := modelstore.Find(artifacts, repository.Repo, repository.Revision)
	if !found || artifact.Path != destination {
		os.RemoveAll(destination)
		return Result{}, errors.New("verify installed model: downloaded model was not discovered")
	}
	return Result{Path: destination, BytesDownloaded: downloaded}, nil
}

// presentResult reports a complete final model for the exact repository
// identity, the incomplete-final-model error, or nothing.
func presentResult(root string, repository Repository) (Result, bool, error) {
	artifacts, err := modelstore.Scan(root)
	if err != nil {
		return Result{}, false, err
	}
	artifact, found := modelstore.Find(artifacts, repository.Repo, repository.Revision)
	if !found {
		return Result{}, false, nil
	}
	if !modelstore.Check(artifact.Path, repositoryExpectedFiles(repository)).Complete {
		return Result{}, false, ErrDownloadedIncomplete
	}
	return Result{Path: artifact.Path, AlreadyPresent: true}, true, nil
}

func repositoryExpectedFiles(repository Repository) []modelpack.ModelFile {
	expected := make([]modelpack.ModelFile, len(repository.Files))
	for index, file := range repository.Files {
		expected[index].Path = file.Path
		if file.SizeKnown {
			size := file.Size
			expected[index].Size = &size
		}
	}
	return expected
}

func (client *Client) downloadFile(ctx context.Context, repository Repository, file File, destination string, count func(int64)) (int64, error) {
	escapedRepo, _ := escapeRepo(repository.Repo)
	requestURL := client.baseURL + "/" + escapedRepo + "/resolve/" + url.PathEscape(repository.Revision) + "/" + escapeRemotePath(file.Path)
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, requestURL, nil)
	if err != nil {
		return 0, fmt.Errorf("%w: %s", ErrDownloadFailed, file.Path)
	}
	client.authorize(request)
	response, err := client.http.Do(request)
	if err != nil {
		return 0, fmt.Errorf("%w: %v", ErrNetworkUnavailable, err)
	}
	defer response.Body.Close()
	if err := client.downloadStatus(response, file.Path); err != nil {
		return 0, err
	}
	if commit := response.Header.Get("X-Repo-Commit"); commit != "" && commit != repository.Revision {
		return 0, fmt.Errorf("%w: resolved commit does not match the required revision", ErrDownloadFailed)
	}
	if err := os.MkdirAll(filepath.Dir(destination), 0o755); err != nil {
		return 0, fmt.Errorf("%w: create model directory: %v", ErrDownloadFailed, err)
	}
	output, err := os.OpenFile(destination, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o644)
	if err != nil {
		return 0, fmt.Errorf("%w: create %s: %v", ErrDownloadFailed, file.Path, err)
	}
	written, copyErr := io.Copy(&countingWriter{writer: output, count: count}, response.Body)
	closeErr := output.Close()
	if copyErr != nil {
		return written, fmt.Errorf("%w: %s: %v", ErrDownloadFailed, file.Path, copyErr)
	}
	if closeErr != nil {
		return written, fmt.Errorf("%w: %s: %v", ErrDownloadFailed, file.Path, closeErr)
	}
	return written, nil
}

func (client *Client) authorize(request *http.Request) {
	if client.token != "" {
		request.Header.Set("Authorization", "Bearer "+client.token)
	}
}

func (client *Client) metadataStatus(response *http.Response) error {
	switch response.StatusCode {
	case http.StatusOK:
		return nil
	case http.StatusUnauthorized:
		if client.token == "" {
			return ErrAuthenticationRequired
		}
		return ErrAccessDenied
	case http.StatusForbidden:
		return ErrAccessDenied
	case http.StatusNotFound:
		if client.token == "" {
			// Hugging Face answers unauthenticated requests for gated,
			// private, and hidden repositories exactly like requests for
			// absent ones, so without a token the difference cannot be
			// established and a missing revision must not be claimed.
			return ErrAccessUncertain
		}
		if response.Header.Get("X-Error-Code") == "RevisionNotFound" {
			return ErrRevisionNotFound
		}
		return ErrRepositoryNotFound
	default:
		return fmt.Errorf("inspect repository: HTTP %d", response.StatusCode)
	}
}

func (client *Client) downloadStatus(response *http.Response, file string) error {
	switch response.StatusCode {
	case http.StatusOK:
		return nil
	case http.StatusUnauthorized:
		if client.token == "" {
			return ErrAuthenticationRequired
		}
		return ErrAccessDenied
	case http.StatusForbidden:
		return ErrAccessDenied
	default:
		return fmt.Errorf("%w: %s returned HTTP %d", ErrDownloadFailed, file, response.StatusCode)
	}
}

type countingWriter struct {
	writer io.Writer
	count  func(int64)
}

func (writer *countingWriter) Write(value []byte) (int, error) {
	written, err := writer.writer.Write(value)
	if written > 0 {
		writer.count(int64(written))
	}
	return written, err
}

func writeMarker(path string, repository Repository) error {
	file, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o644)
	if errors.Is(err, os.ErrExist) {
		// A previous attempt can crash between writing the marker and the
		// promotion rename; an identical marker is still valid.
		if markerMatches(path, repository) {
			return nil
		}
		return fmt.Errorf("write model marker: unexpected marker already exists")
	}
	if err != nil {
		return fmt.Errorf("write model marker: %w", err)
	}
	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	err = encoder.Encode(struct {
		Repo     string `json:"repo"`
		Revision string `json:"revision"`
	}{Repo: repository.Repo, Revision: repository.Revision})
	if err != nil {
		file.Close()
		return fmt.Errorf("write model marker: %w", err)
	}
	if err := file.Close(); err != nil {
		return fmt.Errorf("write model marker: %w", err)
	}
	return nil
}

func markerMatches(path string, repository Repository) bool {
	file, err := os.Open(path)
	if err != nil {
		return false
	}
	defer file.Close()
	var identity struct {
		Repo     string `json:"repo"`
		Revision string `json:"revision"`
	}
	if err := json.NewDecoder(file).Decode(&identity); err != nil {
		return false
	}
	return identity.Repo == repository.Repo && identity.Revision == repository.Revision
}

func validateRemotePath(value string) error {
	if value == "" || strings.HasPrefix(value, "/") {
		return errors.New("repository contains an unsafe file path")
	}
	clean := pathpkg.Clean(value)
	if clean == "." || clean == ".." || strings.HasPrefix(clean, "../") || clean != value {
		return errors.New("repository contains an unsafe file path")
	}
	return nil
}

func escapeRemotePath(value string) string {
	parts := strings.Split(value, "/")
	for index := range parts {
		parts[index] = url.PathEscape(parts[index])
	}
	return strings.Join(parts, "/")
}

func escapeRepo(repo string) (string, error) {
	parts := strings.Split(repo, "/")
	if len(parts) != 2 || parts[0] == "" || parts[1] == "" || parts[0] == "." || parts[0] == ".." || parts[1] == "." || parts[1] == ".." {
		return "", errors.New("repository must be namespace/name")
	}
	return url.PathEscape(parts[0]) + "/" + url.PathEscape(parts[1]), nil
}

func fullRevision(value string) bool {
	if len(value) != 40 {
		return false
	}
	for _, character := range value {
		if (character < '0' || character > '9') && (character < 'a' || character > 'f') {
			return false
		}
	}
	return true
}

func hubHTTPClient(baseURL string) *http.Client {
	base, _ := url.Parse(baseURL)
	dialer := &net.Dialer{Timeout: 30 * time.Second, KeepAlive: 30 * time.Second}
	return &http.Client{
		Transport: &http.Transport{
			Proxy:                 http.ProxyFromEnvironment,
			DialContext:           dialer.DialContext,
			ForceAttemptHTTP2:     true,
			MaxIdleConns:          10,
			IdleConnTimeout:       90 * time.Second,
			TLSHandshakeTimeout:   30 * time.Second,
			ResponseHeaderTimeout: 30 * time.Second,
			ExpectContinueTimeout: time.Second,
		},
		CheckRedirect: func(request *http.Request, via []*http.Request) error {
			if len(via) >= 10 {
				return errors.New("too many redirects")
			}
			if request.URL.Host != base.Host || request.URL.Scheme != base.Scheme {
				request.Header.Del("Authorization")
			}
			return nil
		},
	}
}
