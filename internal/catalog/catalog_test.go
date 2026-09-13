package catalog

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

const validSHA = "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"

func TestDefaultURLUsesPublicControllerRepository(t *testing.T) {
	want := "https://raw.githubusercontent.com/wu1ff/B70-LLM-Controller/main/model-packs/index.json"
	if DefaultURL != want {
		t.Fatalf("DefaultURL = %q, want %q", DefaultURL, want)
	}
	if got := NewClient().URL; got != want {
		t.Fatalf("NewClient().URL = %q, want %q", got, want)
	}
}

func TestParseValidCatalogAndSortsDeterministically(t *testing.T) {
	value, err := Parse(strings.NewReader(`{
		"schema_version":1,
		"packs":[
			{"id":"zeta","name":"Zeta","version":"1.0.0","archive_url":"https://example.invalid/zeta.tar.gz","sha256":"` + validSHA + `"},
			{"id":"alpha","name":"Alpha Two","version":"2.0.0","archive_url":"https://example.invalid/a2.tar.gz","sha256":"` + validSHA + `"},
			{"id":"alpha","name":"Alpha One","version":"1.0.0","archive_url":"https://example.invalid/a1.tar.gz","sha256":"` + validSHA + `"}
		]
	}`))
	if err != nil {
		t.Fatal(err)
	}
	got := make([]string, 0, len(value.Packs))
	for _, entry := range value.Packs {
		got = append(got, entry.ID+"@"+entry.Version)
	}
	want := []string{"alpha@1.0.0", "alpha@2.0.0", "zeta@1.0.0"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("order = %v, want %v", got, want)
	}
}

func TestParseRejectsInvalidCatalogs(t *testing.T) {
	tests := map[string]string{
		"unknown field":      `{"schema_version":1,"packs":[],"description":"no"}`,
		"unknown pack field": `{"schema_version":1,"packs":[{"id":"pack","name":"Pack","version":"1","archive_url":"https://example.invalid/a","sha256":"` + validSHA + `","description":"no"}]}`,
		"trailing JSON":      `{"schema_version":1,"packs":[]} {}`,
		"wrong schema":       `{"schema_version":2,"packs":[]}`,
		"missing packs":      `{"schema_version":1}`,
		"bad id":             `{"schema_version":1,"packs":[{"id":"../bad","name":"Bad","version":"1","archive_url":"https://example.invalid/a","sha256":"` + validSHA + `"}]}`,
		"duplicate":          `{"schema_version":1,"packs":[{"id":"same","name":"Same","version":"1","archive_url":"https://example.invalid/a","sha256":"` + validSHA + `"},{"id":"same","name":"Same","version":"1","archive_url":"https://example.invalid/b","sha256":"` + validSHA + `"}]}`,
		"http URL":           `{"schema_version":1,"packs":[{"id":"pack","name":"Pack","version":"1","archive_url":"http://example.invalid/a","sha256":"` + validSHA + `"}]}`,
		"bad sha":            `{"schema_version":1,"packs":[{"id":"pack","name":"Pack","version":"1","archive_url":"https://example.invalid/a","sha256":"sha256:ABC"}]}`,
	}
	for name, input := range tests {
		t.Run(name, func(t *testing.T) {
			if _, err := Parse(strings.NewReader(input)); err == nil {
				t.Fatal("Parse() error = nil")
			}
		})
	}
}

func TestRefreshCachesRemoteAndFallsBackToValidCache(t *testing.T) {
	available := true
	server := httptest.NewTLSServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		if !available {
			writer.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		fmt.Fprintf(writer, `{"schema_version":1,"packs":[{"id":"pack","name":"Pack","version":"1.0.0","archive_url":"https://example.invalid/pack.tar.gz","sha256":"%s"}]}`, validSHA)
	}))
	defer server.Close()
	client := &Client{URL: server.URL, HTTPClient: server.Client()}
	root := t.TempDir()

	result, err := client.Refresh(context.Background(), root)
	if err != nil || result.Cached || len(result.Catalog.Packs) != 1 {
		t.Fatalf("remote result = %#v, %v", result, err)
	}
	if _, err := os.Stat(filepath.Join(root, "catalog", "index.json")); err != nil {
		t.Fatalf("cache missing: %v", err)
	}

	available = false
	result, err = client.Refresh(context.Background(), root)
	if err != nil || !result.Cached || len(result.Catalog.Packs) != 1 {
		t.Fatalf("cached result = %#v, %v", result, err)
	}
}

func TestRefreshFailsCleanlyWithoutValidCache(t *testing.T) {
	server := httptest.NewTLSServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		writer.WriteHeader(http.StatusNotFound)
	}))
	defer server.Close()
	client := &Client{URL: server.URL, HTTPClient: server.Client()}

	if _, err := client.Refresh(context.Background(), t.TempDir()); err == nil || !strings.Contains(err.Error(), "catalog is unavailable") {
		t.Fatalf("missing cache error = %v", err)
	}
	root := t.TempDir()
	if err := os.MkdirAll(filepath.Join(root, "catalog"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "catalog", "index.json"), []byte(`{"schema_version":2,"packs":[]}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := client.Refresh(context.Background(), root); err == nil {
		t.Fatal("invalid cache was accepted")
	}
}
