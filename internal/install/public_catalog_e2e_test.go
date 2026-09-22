package install

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
	"testing"
	"time"

	"b70ctl/internal/catalog"
	"b70ctl/internal/hf"
	"b70ctl/internal/modelstore"
	"b70ctl/internal/packstore"
	runtimebackend "b70ctl/internal/runtime"
)

const (
	e2ePackDirectory = "Qwen3.8-27B"
	e2ePackID        = "qwen38-27b-b70"
	e2ePackName      = "Qwen3.8 27B B70 Pack (MTP1 / dFlash2)"
	e2ePackVersion   = "1.0.4"
	e2eArchiveName   = "qwen38-27b-b70-1.0.4.tar.gz"
	e2eArchiveURL    = "https://packs.example.com/" + e2eArchiveName
)

func TestPublicCatalogHostPreparationE2E(t *testing.T) {
	if os.Getenv("B70_PUBLIC_CATALOG_E2E") != "1" {
		t.Skip("set B70_PUBLIC_CATALOG_E2E=1 to use the host model and runtime inventory")
	}

	modelRoot := "/mnt/models/diotui-models"
	root := t.TempDir()
	archivePath := filepath.Join(root, e2eArchiveName)
	writeHostE2EArchive(t, filepath.Join("..", "..", "model-packs", e2ePackDirectory), archivePath)
	indexPath := writeHostE2EIndex(t, root, archivePath)
	server := hostE2EServer(t, indexPath, archivePath)
	client := &catalog.Client{URL: server.URL + "/model-packs/index.json", HTTPClient: hostE2EHTTPClient(t, server)}

	configRoot := t.TempDir()
	dataRoot := t.TempDir()
	t.Setenv("XDG_CONFIG_HOME", configRoot)
	t.Setenv("XDG_DATA_HOME", dataRoot)
	result, err := client.Refresh(context.Background(), filepath.Join(dataRoot, "b70ctl"))
	if err != nil {
		t.Fatal(err)
	}
	if result.Cached || result.Catalog.SchemaVersion != 1 || len(result.Catalog.Packs) != 1 {
		t.Fatalf("catalog refresh = %#v", result)
	}
	acquired, err := client.Acquire(context.Background(), filepath.Join(dataRoot, "b70ctl"), result.Catalog.Packs[0])
	if err != nil {
		t.Fatal(err)
	}
	defer acquired.Close()

	inventory, err := modelstore.Scan(modelRoot)
	if err != nil {
		t.Fatal(err)
	}
	targets := Targets(acquired.Manifest)
	selected := make([]string, len(targets))
	for index, target := range targets {
		selected[index] = target.ID
	}
	plan, err := BuildPlan(acquired.Manifest, selected, inventory)
	if err != nil {
		t.Fatal(err)
	}
	counts := plan.Counts()
	if counts.SelectedTargets != 4 || counts.Existing != 5 || counts.Incomplete != 0 || counts.Downloads != 0 || counts.Skipped != 0 {
		t.Fatalf("preparation plan counts = %#v", counts)
	}
	if len(plan.Runtimes) != 1 {
		t.Fatalf("runtime plan count = %d", len(plan.Runtimes))
	}
	if inspection := runtimebackend.InspectRuntime(plan.Runtimes[0]); inspection.Status != runtimebackend.ImagePresent {
		t.Fatalf("runtime image is not reusable: %#v", inspection)
	}

	prepared, err := Prepare(context.Background(), acquired.Path, filepath.Join(dataRoot, "b70ctl", "model-packs"), filepath.Join(dataRoot, "b70ctl"), modelRoot, packstore.SourceRemote, plan, hf.NewClient(""), nil)
	if err != nil {
		t.Fatal(err)
	}
	if prepared.Pack.Source != packstore.SourceRemote || len(prepared.Items) != 5 || len(prepared.RuntimeItems) != 1 {
		t.Fatalf("remote preparation = %#v", prepared)
	}
	for _, item := range prepared.Items {
		if item.Outcome != Reused {
			t.Fatalf("artifact %s outcome = %s", item.Artifact.ModelID, item.Outcome)
		}
	}
	if prepared.RuntimeItems[0].Outcome != runtimebackend.AcquisitionReused {
		t.Fatalf("runtime acquisition = %#v", prepared.RuntimeItems[0])
	}
	t.Logf("remote local-HTTPS E2E PASS: models=%d runtimes=%d modes=%d profiles=%d, artifacts Reused=%d, runtime=%s", len(acquired.Manifest.Models), len(acquired.Manifest.Runtimes), len(acquired.Manifest.Modes), len(acquired.Manifest.Profiles), len(prepared.Items), prepared.RuntimeItems[0].Outcome)
}

// The distribution archive is a generated release artifact, not a committed
// fixture: build it from the source pack with the established normalization
// (sorted files, mode 0644, epoch mtime, zero ownership, GNU tar, nameless
// gzip) in a temporary directory.
func writeHostE2EArchive(t *testing.T, packDirectory, destination string) {
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

func writeHostE2EIndex(t *testing.T, directory, archivePath string) string {
	t.Helper()
	data, err := os.ReadFile(archivePath)
	if err != nil {
		t.Fatal(err)
	}
	digest := sha256.Sum256(data)
	index := struct {
		SchemaVersion int             `json:"schema_version"`
		Packs         []catalog.Entry `json:"packs"`
	}{
		SchemaVersion: 1,
		Packs: []catalog.Entry{{
			ID:         e2ePackID,
			Name:       e2ePackName,
			Version:    e2ePackVersion,
			ArchiveURL: e2eArchiveURL,
			SHA256:     "sha256:" + hex.EncodeToString(digest[:]),
		}},
	}
	encoded, err := json.MarshalIndent(index, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(directory, "index.json")
	if err := os.WriteFile(path, encoded, 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func hostE2EServer(t *testing.T, indexPath, archivePath string) *httptest.Server {
	t.Helper()
	server := httptest.NewTLSServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		switch request.URL.Path {
		case "/model-packs/index.json":
			http.ServeFile(writer, request, indexPath)
		case "/" + e2eArchiveName:
			http.ServeFile(writer, request, archivePath)
		default:
			http.NotFound(writer, request)
		}
	}))
	t.Cleanup(server.Close)
	return server
}

func hostE2EHTTPClient(t *testing.T, server *httptest.Server) *http.Client {
	t.Helper()
	transport := server.Client().Transport.(*http.Transport).Clone()
	transport.DialContext = func(ctx context.Context, network, _ string) (net.Conn, error) {
		return (&net.Dialer{}).DialContext(ctx, network, server.Listener.Addr().String())
	}
	transport.TLSClientConfig = &tls.Config{InsecureSkipVerify: true} // Test server only.
	return &http.Client{Transport: transport}
}
