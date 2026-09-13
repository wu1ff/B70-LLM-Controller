package catalog

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

type archiveEntry struct {
	name     string
	typeflag byte
	linkname string
	data     []byte
}

func TestAcquireDownloadsVerifiesExtractsAndCleans(t *testing.T) {
	archive := validPackArchive(t, "remote-pack", "Remote Pack", "1.0.0")
	server := archiveServer(t, archive)
	defer server.Close()
	client := &Client{HTTPClient: server.Client()}
	root := t.TempDir()
	entry := catalogEntry(server.URL, archive, "remote-pack", "Remote Pack", "1.0.0")

	pack, err := client.Acquire(context.Background(), root, entry)
	if err != nil {
		t.Fatal(err)
	}
	if pack.Manifest.ID != entry.ID {
		t.Fatalf("manifest = %#v", pack.Manifest)
	}
	temporary := filepath.Dir(pack.Path)
	if _, err := os.Stat(filepath.Join(pack.Path, "pack.json")); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(temporary, "pack.tar.gz")); !os.IsNotExist(err) {
		t.Fatalf("verified archive was not removed: %v", err)
	}
	if err := pack.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(temporary); !os.IsNotExist(err) {
		t.Fatalf("temporary state remains: %v", err)
	}
}

func TestAcquireRejectsChecksumBeforeExtractionAndCleans(t *testing.T) {
	archive := makeArchive(t, []archiveEntry{{name: "../escape", typeflag: tar.TypeReg, data: []byte("bad")}})
	server := archiveServer(t, archive)
	defer server.Close()
	client := &Client{HTTPClient: server.Client()}
	root := t.TempDir()
	entry := catalogEntry(server.URL, archive, "pack", "Pack", "1")
	entry.SHA256 = validSHA

	if _, err := client.Acquire(context.Background(), root, entry); err == nil || err.Error() != "pack archive checksum mismatch" {
		t.Fatalf("Acquire() error = %v", err)
	}
	assertNoTemporaryState(t, root)
}

func TestAcquireRejectsUnsafeAndMalformedArchives(t *testing.T) {
	tests := map[string][]byte{
		"traversal":      makeArchive(t, []archiveEntry{{name: "../escape", typeflag: tar.TypeReg, data: []byte("bad")}}),
		"absolute":       makeArchive(t, []archiveEntry{{name: "/escape", typeflag: tar.TypeReg, data: []byte("bad")}}),
		"symlink":        makeArchive(t, []archiveEntry{{name: "link", typeflag: tar.TypeSymlink, linkname: "pack.json"}}),
		"hardlink":       makeArchive(t, []archiveEntry{{name: "link", typeflag: tar.TypeLink, linkname: "pack.json"}}),
		"malformed gzip": []byte("not gzip"),
		"malformed tar":  gzipBytes(t, []byte("not tar")),
	}
	for name, archive := range tests {
		t.Run(name, func(t *testing.T) {
			server := archiveServer(t, archive)
			defer server.Close()
			client := &Client{HTTPClient: server.Client()}
			root := t.TempDir()
			entry := catalogEntry(server.URL, archive, "pack", "Pack", "1")
			if _, err := client.Acquire(context.Background(), root, entry); err == nil {
				t.Fatal("Acquire() error = nil")
			}
			assertNoTemporaryState(t, root)
		})
	}
}

func TestAcquireRequiresValidPackAndMatchingCatalogIdentity(t *testing.T) {
	invalid := makeArchive(t, []archiveEntry{{name: "README.md", typeflag: tar.TypeReg, data: []byte("# Missing manifest\n")}})
	tests := []struct {
		name    string
		archive []byte
		entry   func(string, []byte) Entry
		match   string
	}{
		{name: "invalid pack", archive: invalid, entry: func(url string, data []byte) Entry {
			return catalogEntry(url, data, "pack", "Pack", "1")
		}, match: "load downloaded model pack"},
		{name: "id mismatch", archive: validPackArchive(t, "actual", "Pack", "1"), entry: func(url string, data []byte) Entry {
			return catalogEntry(url, data, "expected", "Pack", "1")
		}, match: "catalog entry does not match downloaded pack"},
		{name: "name mismatch", archive: validPackArchive(t, "pack", "Actual", "1"), entry: func(url string, data []byte) Entry {
			return catalogEntry(url, data, "pack", "Expected", "1")
		}, match: "catalog entry does not match downloaded pack"},
		{name: "version mismatch", archive: validPackArchive(t, "pack", "Pack", "2"), entry: func(url string, data []byte) Entry {
			return catalogEntry(url, data, "pack", "Pack", "1")
		}, match: "catalog entry does not match downloaded pack"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			server := archiveServer(t, test.archive)
			defer server.Close()
			client := &Client{HTTPClient: server.Client()}
			root := t.TempDir()
			if _, err := client.Acquire(context.Background(), root, test.entry(server.URL, test.archive)); err == nil || !strings.Contains(err.Error(), test.match) {
				t.Fatalf("Acquire() error = %v", err)
			}
			assertNoTemporaryState(t, root)
		})
	}
}

func validPackArchive(t *testing.T, id, name, version string) []byte {
	t.Helper()
	digest := "sha256:" + strings.Repeat("b", 64)
	manifest := map[string]any{
		"schema_version": 1, "id": id, "name": name, "version": version,
		"models":   []map[string]any{{"id": "target", "name": "Target", "kind": "target", "repo": "example/target", "revision": strings.Repeat("1", 40), "gated": false, "files": []map[string]any{{"path": "model.bin", "size": 5}}, "mount_path": "/models/target", "launch": map[string]any{"docker_args": []string{}, "environment": map[string]string{}, "command_args": []string{"/models/target"}}}},
		"runtimes": []map[string]any{{"id": "runtime", "image": "example/runtime:1", "digest": digest, "container_port": 8000, "health_path": "/health", "launch": map[string]any{"docker_args": []string{}, "environment": map[string]string{}, "command": []string{"serve"}}}},
		"modes":    []map[string]any{{"id": "base", "display_name": "Base", "assistants": []string{}, "launch": map[string]any{"docker_args": []string{}, "environment": map[string]string{}, "command_args": []string{}}}},
		"profiles": []map[string]any{{"id": "base", "model_id": "target", "runtime_id": "runtime", "cards": 1, "tensor_parallel": 1, "context": 4096, "mode": "base", "launch": map[string]any{"docker_args": []string{}, "environment": map[string]string{}, "command_args": []string{}}}},
	}
	manifestJSON, err := json.Marshal(manifest)
	if err != nil {
		t.Fatal(err)
	}
	return makeArchive(t, []archiveEntry{
		{name: "README.md", typeflag: tar.TypeReg, data: []byte("# Pack\n")},
		{name: "pack.json", typeflag: tar.TypeReg, data: manifestJSON},
		{name: "notes.txt", typeflag: tar.TypeReg, data: []byte("harmless extra")},
	})
}

func makeArchive(t *testing.T, entries []archiveEntry) []byte {
	t.Helper()
	var buffer bytes.Buffer
	gzipWriter := gzip.NewWriter(&buffer)
	tarWriter := tar.NewWriter(gzipWriter)
	for _, entry := range entries {
		header := &tar.Header{Name: entry.name, Typeflag: entry.typeflag, Linkname: entry.linkname, Mode: 0o644, Size: int64(len(entry.data))}
		if entry.typeflag == tar.TypeDir {
			header.Mode = 0o755
			header.Size = 0
		}
		if err := tarWriter.WriteHeader(header); err != nil {
			t.Fatal(err)
		}
		if len(entry.data) > 0 {
			if _, err := tarWriter.Write(entry.data); err != nil {
				t.Fatal(err)
			}
		}
	}
	if err := tarWriter.Close(); err != nil {
		t.Fatal(err)
	}
	if err := gzipWriter.Close(); err != nil {
		t.Fatal(err)
	}
	return buffer.Bytes()
}

func gzipBytes(t *testing.T, data []byte) []byte {
	t.Helper()
	var buffer bytes.Buffer
	writer := gzip.NewWriter(&buffer)
	if _, err := writer.Write(data); err != nil {
		t.Fatal(err)
	}
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}
	return buffer.Bytes()
}

func archiveServer(t *testing.T, archive []byte) *httptest.Server {
	t.Helper()
	return httptest.NewTLSServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		writer.Write(archive)
	}))
}

func catalogEntry(url string, archive []byte, id, name, version string) Entry {
	hash := sha256.Sum256(archive)
	return Entry{ID: id, Name: name, Version: version, ArchiveURL: url, SHA256: "sha256:" + hex.EncodeToString(hash[:])}
}

func assertNoTemporaryState(t *testing.T, root string) {
	t.Helper()
	entries, err := os.ReadDir(root)
	if err != nil {
		t.Fatal(err)
	}
	for _, entry := range entries {
		if strings.HasPrefix(entry.Name(), ".pack-download-") {
			t.Fatalf("temporary state remains: %s", entry.Name())
		}
	}
}
