package catalog

import (
	"archive/tar"
	"compress/gzip"
	"context"
	"crypto/sha256"
	"crypto/tls"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"
	"time"

	"b70ctl/internal/modelpack"
)

const (
	publicPackDirectory = "Qwen3.8-27B"
	publicArchiveName   = "qwen38-27b-b70-1.0.6.tar.gz"
	// Fixture URL for the locally served archive; generated catalogs never
	// reference the repository itself because generated archives are not
	// tracked there.
	publicArchiveURL = "https://packs.example.com/" + publicArchiveName
)

// publishedPackDirectories lists every pack directory that must have a
// matching tracked catalog entry. A pack's release tag is the lowercased
// directory name plus -v<version> (Qwen3.8-27B -> qwen3.8-27b-v1.0.1).
var publishedPackDirectories = []string{"Qwen3.8-27B", "Qwen3.6-35B-A3B", "Qwen3.8-Flash-Next"}

// The legacy release and development archive fixtures use regular mode 0644,
// epoch mtime, zero ownership, GNU tar format and a nameless gzip stream.
func writePackArchive(t *testing.T, packDirectory, destination string) {
	t.Helper()
	writeNamedPackArchive(t, packDirectory, destination, packArchiveNames(t, packDirectory))
}

func writeNamedPackArchive(t *testing.T, packDirectory, destination string, names []string) {
	t.Helper()
	output, err := os.Create(destination)
	if err != nil {
		t.Fatal(err)
	}
	compressed := gzip.NewWriter(output)
	archive := tar.NewWriter(compressed)
	for _, name := range names {
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

func packArchiveNames(t *testing.T, packDirectory string) []string {
	t.Helper()
	names := []string{"README.md", "RUNTIME_RECIPE.md", "pack.json"}
	if _, err := os.Stat(filepath.Join(packDirectory, "WORK_STORAGE.md")); err == nil {
		names = append(names, "WORK_STORAGE.md")
	} else if !os.IsNotExist(err) {
		t.Fatal(err)
	}
	patchRoot := filepath.Join(packDirectory, "patches")
	if _, err := os.Stat(patchRoot); os.IsNotExist(err) {
		return names
	} else if err != nil {
		t.Fatal(err)
	}
	for _, name := range packTreeNames(t, patchRoot) {
		names = append(names, "patches/"+name)
	}
	sort.Strings(names)
	return names
}

func TestFlashNextSourceArchiveLoad(t *testing.T) {
	packDirectory := filepath.Join("..", "..", "model-packs", "Qwen3.8-Flash-Next")
	manifest, err := modelpack.Load(packDirectory)
	if err != nil {
		t.Fatal(err)
	}
	want := packTreeNames(t, packDirectory)
	archivePath := filepath.Join(t.TempDir(), "flashnext-source.tar.gz")
	writeNamedPackArchive(t, packDirectory, archivePath, want)
	archiveBytes, err := os.ReadFile(archivePath)
	if err != nil {
		t.Fatal(err)
	}
	repeatPath := filepath.Join(t.TempDir(), "repeat-flashnext-source.tar.gz")
	writeNamedPackArchive(t, packDirectory, repeatPath, want)
	if fileDigest(t, archivePath) != fileDigest(t, repeatPath) {
		t.Fatal("Flash-Next source archive generation is not deterministic")
	}
	server := archiveServer(t, archiveBytes)
	defer server.Close()
	entry := catalogEntry(server.URL, archiveBytes, manifest.ID, manifest.Name, manifest.Version)
	client := &Client{HTTPClient: server.Client()}
	acquired, err := client.Acquire(context.Background(), t.TempDir(), entry)
	if err != nil {
		t.Fatal(err)
	}
	defer acquired.Close()

	got := packTreeNames(t, acquired.Path)
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("extracted files = %v, want %v", got, want)
	}
	for _, name := range got {
		if fileDigest(t, filepath.Join(acquired.Path, name)) != fileDigest(t, filepath.Join(packDirectory, name)) {
			t.Fatalf("extracted file differs from development source: %s", name)
		}
	}
	patchRoot := filepath.Join(acquired.Path, "patches")
	verifyTreeChecksums(t, patchRoot, packTreeNames(t, patchRoot))
}

func packTreeNames(t *testing.T, root string) []string {
	t.Helper()
	var names []string
	err := filepath.WalkDir(root, func(path string, entry os.DirEntry, err error) error {
		if err != nil || entry.IsDir() {
			return err
		}
		if !entry.Type().IsRegular() {
			return fmt.Errorf("pack entry %s is not a regular file", path)
		}
		relative, err := filepath.Rel(root, path)
		if err != nil {
			return err
		}
		names = append(names, filepath.ToSlash(relative))
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	sort.Strings(names)
	return names
}

func verifyTreeChecksums(t *testing.T, root string, names []string) {
	t.Helper()
	sums, err := os.ReadFile(filepath.Join(root, "SHA256SUMS"))
	if err != nil {
		t.Fatal(err)
	}
	remaining := make(map[string]bool, len(names))
	for _, name := range names {
		if name != "SHA256SUMS" {
			remaining[name] = true
		}
	}
	for _, line := range strings.Split(strings.TrimSpace(string(sums)), "\n") {
		parts := strings.SplitN(line, "  ", 2)
		if len(parts) != 2 || !remaining[parts[1]] {
			t.Fatalf("%s: invalid, unexpected or duplicate checksum line %q", root, line)
		}
		if actual := fileDigest(t, filepath.Join(root, parts[1])); actual != "sha256:"+parts[0] {
			t.Fatalf("%s: checksum mismatch for %s", root, parts[1])
		}
		delete(remaining, parts[1])
	}
	if len(remaining) != 0 {
		t.Fatalf("%s: missing checksums for %v", root, remaining)
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
	wantRegistry := "ghcr.io/wu1ff/qwen38-27b-b70@sha256:f3006020add52dedfed08947b05234ef4c058b47158afe61c0e79f9ef60a6f4a"
	wantDigest := "sha256:f3006020add52dedfed08947b05234ef4c058b47158afe61c0e79f9ef60a6f4a"
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

// Match catalog metadata to the source manifests and verify the published
// checksums with each pack's release inputs and archive format.
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
		tag := strings.ToLower(directory) + "-v" + manifest.Version
		if directory == "Qwen3.8-Flash-Next" {
			writeFlashNextReleaseArchive(t, tag, directory, archivePath)
		} else {
			writePackArchive(t, packPath, archivePath)
		}
		if digest := fileDigest(t, archivePath); entry.SHA256 != digest {
			t.Fatalf("catalog SHA-256 for %s = %s, rebuilt release archive = %s", tag, entry.SHA256, digest)
		}

		wantURL := "https://github.com/wu1ff/B70-LLM-Controller/releases/download/" + tag + "/" + archiveName
		if entry.ArchiveURL != wantURL {
			t.Fatalf("catalog archive_url %s does not match pack release download URL %s", entry.ArchiveURL, wantURL)
		}
	}
}

func writeFlashNextReleaseArchive(t *testing.T, tag, directory, destination string) {
	t.Helper()
	source := filepath.Join(t.TempDir(), "tagged-pack.tar")
	export := exec.Command("git", "archive", "--format=tar", "-o", source, "refs/tags/"+tag+":model-packs/"+directory)
	export.Dir = filepath.Join("..", "..")
	if output, err := export.CombinedOutput(); err != nil {
		t.Fatalf("export pack release %s (use a checkout with its release tag): %v\n%s", tag, err, output)
	}
	// The v6 asset was published with Python tarfile and level-6 zlib.
	// Stream the tagged files in name order with the original metadata rules.
	const repack = `
import gzip
from pathlib import PurePosixPath
import sys
import tarfile

with tarfile.open(sys.argv[1]) as source, open(sys.argv[2], "wb") as raw:
    with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=6) as compressed:
        with tarfile.open(fileobj=compressed, mode="w|", format=tarfile.PAX_FORMAT) as release:
            for member in sorted(source.getmembers(), key=lambda member: PurePosixPath(member.name)):
                if member.isdir():
                    continue
                if not member.isfile():
                    raise ValueError(f"unexpected release entry {member.name}")
                entry = tarfile.TarInfo(member.name)
                entry.mode = 0o755 if member.mode & 0o111 else 0o644
                entry.size = member.size
                with source.extractfile(member) as content:
                    release.addfile(entry, content)
`
	if output, err := exec.Command("python3", "-c", repack, source, destination).CombinedOutput(); err != nil {
		t.Fatalf("repack %s with the Flash-Next release format: %v\n%s", tag, err, output)
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
