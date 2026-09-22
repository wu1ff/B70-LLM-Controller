package catalog

import (
	"archive/tar"
	"compress/gzip"
	"context"
	"crypto/sha256"
	"crypto/tls"
	"encoding/hex"
	"encoding/json"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"b70ctl/internal/modelpack"
)

const (
	publicPackDirectory = "Qwen3.8-27B"
	publicArchiveName   = "qwen38-27b-b70-1.0.3.tar.gz"
	// Fixture URL for the locally served archive; generated catalogs never
	// reference the repository itself because generated archives are not
	// tracked there.
	publicArchiveURL = "https://packs.example.com/" + publicArchiveName
)

// publishedPackDirectories lists every pack directory that must have a
// matching tracked catalog entry. A pack's release tag is the lowercased
// directory name plus -v<version> (Qwen3.8-27B -> qwen3.8-27b-v1.0.1).
var publishedPackDirectories = []string{"Qwen3.8-27B", "Qwen3.6-35B-A3B"}

// Pack distribution archives are generated release artifacts and are not
// committed. The tests build them from the source pack with the established
// normalization: the pack's three files in sorted order, regular mode 0644,
// epoch mtime, zero ownership, GNU tar format, and a nameless gzip stream —
// byte-identical output for identical pack contents.
func writePackArchive(t *testing.T, packDirectory, destination string) {
	t.Helper()
	output, err := os.Create(destination)
	if err != nil {
		t.Fatal(err)
	}
	compressed := gzip.NewWriter(output)
	archive := tar.NewWriter(compressed)
	for _, name := range []string{"README.md", "RUNTIME_RECIPE.md", "pack.json"} {
		data, err := os.ReadFile(filepath.Join(packDirectory, name))
		if err != nil {
			t.Fatal(err)
		}
		header := &tar.Header{
			Name:     name,
			Typeflag: tar.TypeReg,
			Mode:     0o644,
			Size:     int64(len(data)),
			ModTime:  time.Unix(0, 0).UTC(),
			Format:   tar.FormatGNU,
		}
		if err := archive.WriteHeader(header); err != nil {
			t.Fatal(err)
		}
		if _, err := archive.Write(data); err != nil {
			t.Fatal(err)
		}
	}
	if err := archive.Close(); err != nil {
		t.Fatal(err)
	}
	if err := compressed.Close(); err != nil {
		t.Fatal(err)
	}
	if err := output.Close(); err != nil {
		t.Fatal(err)
	}
}

func fileDigest(t *testing.T, path string) string {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	digest := sha256.Sum256(data)
	return "sha256:" + hex.EncodeToString(digest[:])
}

func writeCatalogIndex(t *testing.T, destination string, entry Entry) {
	t.Helper()
	data, err := json.MarshalIndent(Catalog{SchemaVersion: 1, Packs: []Entry{entry}}, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(destination, data, 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestPublicCatalogArchive(t *testing.T) {
	modelPacks := filepath.Join("..", "..", "model-packs")
	packPath := filepath.Join(modelPacks, publicPackDirectory)

	manifest, err := modelpack.Load(packPath)
	if err != nil {
		t.Fatal(err)
	}

	root := t.TempDir()
	archivePath := filepath.Join(root, publicArchiveName)
	writePackArchive(t, packPath, archivePath)
	repeat := filepath.Join(root, "repeat-"+publicArchiveName)
	writePackArchive(t, packPath, repeat)
	digest := fileDigest(t, archivePath)
	if digest != fileDigest(t, repeat) {
		t.Fatal("archive generation is not deterministic")
	}

	indexPath := filepath.Join(root, "index.json")
	writeCatalogIndex(t, indexPath, Entry{
		ID:         manifest.ID,
		Name:       manifest.Name,
		Version:    manifest.Version,
		ArchiveURL: publicArchiveURL,
		SHA256:     digest,
	})

	server := publicAssetServer(t, indexPath, archivePath)
	client := &Client{URL: server.URL + "/model-packs/index.json", HTTPClient: localHTTPClient(t, server)}
	result, err := client.Refresh(context.Background(), t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	acquired, err := client.Acquire(context.Background(), t.TempDir(), result.Catalog.Packs[0])
	if err != nil {
		t.Fatal(err)
	}
	defer acquired.Close()

	files, err := os.ReadDir(acquired.Path)
	if err != nil {
		t.Fatal(err)
	}
	names := make([]string, 0, len(files))
	for _, file := range files {
		names = append(names, file.Name())
	}
	if !reflect.DeepEqual(names, []string{"README.md", "RUNTIME_RECIPE.md", "pack.json"}) {
		t.Fatalf("archive files = %v", names)
	}
	if len(acquired.Manifest.Models) != 5 || len(acquired.Manifest.Runtimes) != 1 || len(acquired.Manifest.Modes) != 3 || len(acquired.Manifest.Profiles) != 96 {
		t.Fatalf("pack counts = models=%d runtimes=%d modes=%d profiles=%d", len(acquired.Manifest.Models), len(acquired.Manifest.Runtimes), len(acquired.Manifest.Modes), len(acquired.Manifest.Profiles))
	}
	wantRegistry := "ghcr.io/wu1ff/qwen38-27b-b70@sha256:c0c9b8f382298bdd90f78ae2f4700637241c7a8e2b6c7ab591dc933c76b73cbf"
	wantDigest := "sha256:c0c9b8f382298bdd90f78ae2f4700637241c7a8e2b6c7ab591dc933c76b73cbf"
	if len(acquired.Manifest.Runtimes) != 1 || acquired.Manifest.Runtimes[0].Registry != wantRegistry || acquired.Manifest.Runtimes[0].Digest != wantDigest {
		t.Fatalf("runtime authority = %#v", acquired.Manifest.Runtimes)
	}
	environment := acquired.Manifest.Runtimes[0].Launch.Environment
	if environment["CCL_SYCL_ALLREDUCE_TMP_BUF"] != "1" || environment["CCL_SYCL_ALLGATHERV_TMP_BUF"] != "1" {
		t.Fatalf("collective environment = %#v", environment)
	}
}

// The tracked catalog must stay internally honest: generated pack archives
// are not committed, so no entry may advertise an archive inside this
// repository's model-packs/archives/ tree. Publish the archive somewhere
// real before adding an entry for it.
func TestTrackedCatalogListsNoRepositoryArchives(t *testing.T) {
	indexPath := filepath.Join("..", "..", "model-packs", "index.json")
	index, err := os.Open(indexPath)
	if err != nil {
		t.Fatal(err)
	}
	catalog, err := Parse(index)
	closeErr := index.Close()
	if err != nil {
		t.Fatal(err)
	}
	if closeErr != nil {
		t.Fatal(closeErr)
	}
	for _, entry := range catalog.Packs {
		if strings.Contains(entry.ArchiveURL, "/model-packs/archives/") {
			t.Fatalf("catalog pack %q advertises repository archive %s; archives under model-packs/archives/ are generated and untracked", entry.ID, entry.ArchiveURL)
		}
	}
}

// The tracked catalog entries for the published packs must stay in lockstep
// with the pack source directories: matching id, name, and version; the
// SHA-256 of the archive built from that source by the established
// normalization; and the archive hosted at the pack tag's GitHub release
// download URL. Validated without requiring the archive to be committed.
func TestTrackedCatalogMatchesPublishedPack(t *testing.T) {
	index, err := os.Open(filepath.Join("..", "..", "model-packs", "index.json"))
	if err != nil {
		t.Fatal(err)
	}
	catalog, err := Parse(index)
	closeErr := index.Close()
	if err != nil {
		t.Fatal(err)
	}
	if closeErr != nil {
		t.Fatal(closeErr)
	}
	if len(catalog.Packs) != len(publishedPackDirectories) {
		t.Fatalf("tracked catalog lists %d packs, want exactly the %d published packs %v", len(catalog.Packs), len(publishedPackDirectories), publishedPackDirectories)
	}

	for _, directory := range publishedPackDirectories {
		packPath := filepath.Join("..", "..", "model-packs", directory)
		manifest, err := modelpack.Load(packPath)
		if err != nil {
			t.Fatal(err)
		}
		var entry Entry
		for _, candidate := range catalog.Packs {
			if candidate.ID == manifest.ID {
				entry = candidate
			}
		}
		if entry.ID == "" {
			t.Fatalf("pack %s (%s) has no catalog entry; catalogs and pack directories must be added together", manifest.ID, directory)
		}
		if entry.Name != manifest.Name || entry.Version != manifest.Version {
			t.Fatalf("catalog entry %+v does not match pack manifest id/name/version %+v", entry, manifest)
		}

		archiveName := manifest.ID + "-" + manifest.Version + ".tar.gz"
		archivePath := filepath.Join(t.TempDir(), archiveName)
		writePackArchive(t, packPath, archivePath)
		if digest := fileDigest(t, archivePath); entry.SHA256 != digest {
			t.Fatalf("catalog sha256 %s does not match archive built from pack source %s", entry.SHA256, digest)
		}

		wantURL := "https://github.com/wu1ff/B70-LLM-Controller/releases/download/" + strings.ToLower(directory) + "-v" + manifest.Version + "/" + archiveName
		if entry.ArchiveURL != wantURL {
			t.Fatalf("catalog archive_url %s does not match pack release download URL %s", entry.ArchiveURL, wantURL)
		}
	}
}

func publicAssetServer(t *testing.T, indexPath, archivePath string) *httptest.Server {
	t.Helper()
	server := httptest.NewTLSServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		switch request.URL.Path {
		case "/model-packs/index.json":
			http.ServeFile(writer, request, indexPath)
		case "/" + publicArchiveName:
			http.ServeFile(writer, request, archivePath)
		default:
			http.NotFound(writer, request)
		}
	}))
	t.Cleanup(server.Close)
	return server
}

func localHTTPClient(t *testing.T, server *httptest.Server) *http.Client {
	t.Helper()
	transport := server.Client().Transport.(*http.Transport).Clone()
	transport.DialContext = func(ctx context.Context, network, _ string) (net.Conn, error) {
		return (&net.Dialer{}).DialContext(ctx, network, server.Listener.Addr().String())
	}
	transport.TLSClientConfig = &tls.Config{InsecureSkipVerify: true} // Test server only.
	return &http.Client{Transport: transport}
}
