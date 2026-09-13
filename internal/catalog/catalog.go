package catalog

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
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"time"
)

const DefaultURL = "https://raw.githubusercontent.com/wu1ff/B70-LLM-Controller/main/model-packs/index.json"

const (
	schemaVersion  = 1
	maximumCatalog = 2 << 20
)

var (
	packIDPattern = regexp.MustCompile(`^[a-z0-9]+(?:[._-][a-z0-9]+)*$`)
	shaPattern    = regexp.MustCompile(`^sha256:[0-9a-f]{64}$`)
)

type Catalog struct {
	SchemaVersion int     `json:"schema_version"`
	Packs         []Entry `json:"packs"`
}

type Entry struct {
	ID         string `json:"id"`
	Name       string `json:"name"`
	Version    string `json:"version"`
	ArchiveURL string `json:"archive_url"`
	SHA256     string `json:"sha256"`
}

type Result struct {
	Catalog Catalog
	Cached  bool
}

type Client struct {
	URL        string
	HTTPClient *http.Client
}

func NewClient() *Client {
	transport := &http.Transport{
		Proxy:                 http.ProxyFromEnvironment,
		DialContext:           (&net.Dialer{Timeout: 10 * time.Second, KeepAlive: 30 * time.Second}).DialContext,
		TLSHandshakeTimeout:   10 * time.Second,
		ResponseHeaderTimeout: 15 * time.Second,
		ExpectContinueTimeout: time.Second,
	}
	return &Client{
		URL: DefaultURL,
		HTTPClient: &http.Client{
			Transport: transport,
			Timeout:   30 * time.Second,
		},
	}
}

func Parse(reader io.Reader) (Catalog, error) {
	var value Catalog
	decoder := json.NewDecoder(reader)
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&value); err != nil {
		return Catalog{}, fmt.Errorf("decode catalog: %w", err)
	}
	var extra any
	if err := decoder.Decode(&extra); !errors.Is(err, io.EOF) {
		if err != nil {
			return Catalog{}, fmt.Errorf("decode catalog: %w", err)
		}
		return Catalog{}, errors.New("catalog must contain one JSON object")
	}
	if err := value.validate(); err != nil {
		return Catalog{}, err
	}
	sort.Slice(value.Packs, func(i, j int) bool {
		if value.Packs[i].ID == value.Packs[j].ID {
			return value.Packs[i].Version < value.Packs[j].Version
		}
		return value.Packs[i].ID < value.Packs[j].ID
	})
	return value, nil
}

func (value Catalog) validate() error {
	if value.SchemaVersion != schemaVersion {
		return fmt.Errorf("schema_version must be %d", schemaVersion)
	}
	if value.Packs == nil {
		return errors.New("packs list is required")
	}
	seen := make(map[string]struct{}, len(value.Packs))
	for _, entry := range value.Packs {
		if !packIDPattern.MatchString(entry.ID) {
			return fmt.Errorf("catalog pack id %q is invalid", entry.ID)
		}
		if strings.TrimSpace(entry.Name) == "" {
			return fmt.Errorf("catalog pack %q name is required", entry.ID)
		}
		if strings.TrimSpace(entry.Version) == "" {
			return fmt.Errorf("catalog pack %q version is required", entry.ID)
		}
		archiveURL, err := url.Parse(entry.ArchiveURL)
		if err != nil || archiveURL.Scheme != "https" || archiveURL.Host == "" {
			return fmt.Errorf("catalog pack %q archive_url must use HTTPS", entry.ID)
		}
		if !shaPattern.MatchString(entry.SHA256) {
			return fmt.Errorf("catalog pack %q sha256 is invalid", entry.ID)
		}
		key := entry.ID + "\x00" + entry.Version
		if _, exists := seen[key]; exists {
			return fmt.Errorf("duplicate catalog pack %s %s", entry.ID, entry.Version)
		}
		seen[key] = struct{}{}
	}
	return nil
}

func (client *Client) Refresh(ctx context.Context, dataRoot string) (Result, error) {
	remote, remoteErr := client.fetch(ctx)
	if remoteErr == nil {
		if err := cache(dataRoot, remote); err != nil {
			return Result{}, err
		}
		return Result{Catalog: remote}, nil
	}

	cached, cacheErr := loadCache(dataRoot)
	if cacheErr == nil {
		return Result{Catalog: cached, Cached: true}, nil
	}
	return Result{}, fmt.Errorf("model pack catalog is unavailable: %w", remoteErr)
}

func (client *Client) fetch(ctx context.Context) (Catalog, error) {
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, client.URL, nil)
	if err != nil {
		return Catalog{}, fmt.Errorf("create catalog request: %w", err)
	}
	response, err := client.HTTPClient.Do(request)
	if err != nil {
		return Catalog{}, fmt.Errorf("fetch catalog: %w", err)
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		return Catalog{}, fmt.Errorf("fetch catalog: HTTP %s", response.Status)
	}
	return parseLimited(response.Body)
}

func cache(dataRoot string, value Catalog) error {
	directory := filepath.Join(dataRoot, "catalog")
	if err := os.MkdirAll(directory, 0o755); err != nil {
		return fmt.Errorf("create catalog cache: %w", err)
	}
	file, err := os.CreateTemp(directory, ".index-*.tmp")
	if err != nil {
		return fmt.Errorf("create catalog cache: %w", err)
	}
	temporary := file.Name()
	defer os.Remove(temporary)
	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(value); err != nil {
		file.Close()
		return fmt.Errorf("write catalog cache: %w", err)
	}
	if err := file.Sync(); err != nil {
		file.Close()
		return fmt.Errorf("sync catalog cache: %w", err)
	}
	if err := file.Close(); err != nil {
		return fmt.Errorf("close catalog cache: %w", err)
	}
	if err := os.Rename(temporary, filepath.Join(directory, "index.json")); err != nil {
		return fmt.Errorf("save catalog cache: %w", err)
	}
	return nil
}

func loadCache(dataRoot string) (Catalog, error) {
	file, err := os.Open(filepath.Join(dataRoot, "catalog", "index.json"))
	if err != nil {
		return Catalog{}, err
	}
	defer file.Close()
	return parseLimited(file)
}

func parseLimited(reader io.Reader) (Catalog, error) {
	data, err := io.ReadAll(io.LimitReader(reader, maximumCatalog+1))
	if err != nil {
		return Catalog{}, fmt.Errorf("read catalog: %w", err)
	}
	if len(data) > maximumCatalog {
		return Catalog{}, errors.New("catalog is too large")
	}
	return Parse(strings.NewReader(string(data)))
}
