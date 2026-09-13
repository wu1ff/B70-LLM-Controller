package catalog

import (
	"archive/tar"
	"compress/gzip"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	"b70ctl/internal/modelpack"
)

type AcquiredPack struct {
	Path     string
	Manifest *modelpack.Manifest
	root     string
}

// DownloadTempPrefix is the throwaway temporary-directory prefix Acquire
// creates beneath the application data root. Startup cleanup removes
// crash leftovers with this prefix.
const DownloadTempPrefix = ".pack-download-"

func (pack *AcquiredPack) Close() error {
	if pack == nil || pack.root == "" {
		return nil
	}
	err := os.RemoveAll(pack.root)
	pack.root = ""
	return err
}

func (client *Client) Acquire(ctx context.Context, dataRoot string, entry Entry) (*AcquiredPack, error) {
	if err := os.MkdirAll(dataRoot, 0o755); err != nil {
		return nil, fmt.Errorf("create application data directory: %w", err)
	}
	root, err := os.MkdirTemp(dataRoot, DownloadTempPrefix)
	if err != nil {
		return nil, fmt.Errorf("create temporary pack directory: %w", err)
	}
	failed := true
	defer func() {
		if failed {
			_ = os.RemoveAll(root)
		}
	}()

	archivePath := filepath.Join(root, "pack.tar.gz")
	if err := client.download(ctx, entry, archivePath); err != nil {
		return nil, err
	}
	packPath := filepath.Join(root, "pack")
	if err := extract(archivePath, packPath); err != nil {
		return nil, err
	}
	manifest, err := modelpack.Load(packPath)
	if err != nil {
		return nil, fmt.Errorf("load downloaded model pack: %w", err)
	}
	if manifest.ID != entry.ID || manifest.Name != entry.Name || manifest.Version != entry.Version {
		return nil, errors.New("catalog entry does not match downloaded pack")
	}
	if err := os.Remove(archivePath); err != nil {
		return nil, fmt.Errorf("remove temporary pack archive: %w", err)
	}
	failed = false
	return &AcquiredPack{Path: packPath, Manifest: manifest, root: root}, nil
}

func (client *Client) download(ctx context.Context, entry Entry, destination string) error {
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, entry.ArchiveURL, nil)
	if err != nil {
		return fmt.Errorf("create pack archive request: %w", err)
	}
	response, err := client.HTTPClient.Do(request)
	if err != nil {
		return fmt.Errorf("download pack archive: %w", err)
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		return fmt.Errorf("download pack archive: HTTP %s", response.Status)
	}
	file, err := os.OpenFile(destination, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
	if err != nil {
		return fmt.Errorf("create temporary pack archive: %w", err)
	}
	hash := sha256.New()
	_, copyErr := io.Copy(io.MultiWriter(file, hash), response.Body)
	closeErr := file.Close()
	if copyErr != nil {
		return fmt.Errorf("download pack archive: %w", copyErr)
	}
	if closeErr != nil {
		return fmt.Errorf("close temporary pack archive: %w", closeErr)
	}
	actual := "sha256:" + hex.EncodeToString(hash.Sum(nil))
	if actual != entry.SHA256 {
		return errors.New("pack archive checksum mismatch")
	}
	return nil
}

func extract(archivePath, destination string) error {
	file, err := os.Open(archivePath)
	if err != nil {
		return fmt.Errorf("open pack archive: %w", err)
	}
	defer file.Close()
	compressed, err := gzip.NewReader(file)
	if err != nil {
		return fmt.Errorf("open pack archive gzip: %w", err)
	}
	defer compressed.Close()
	if err := os.Mkdir(destination, 0o755); err != nil {
		return fmt.Errorf("create extraction directory: %w", err)
	}

	reader := tar.NewReader(compressed)
	for {
		header, err := reader.Next()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return fmt.Errorf("read pack archive: %w", err)
		}
		clean := filepath.Clean(filepath.FromSlash(header.Name))
		if header.Name == "" || filepath.IsAbs(clean) || clean == ".." || strings.HasPrefix(clean, ".."+string(filepath.Separator)) {
			return errors.New("pack archive contains an unsafe path")
		}
		target := filepath.Join(destination, clean)
		relative, err := filepath.Rel(destination, target)
		if err != nil || relative == ".." || strings.HasPrefix(relative, ".."+string(filepath.Separator)) {
			return errors.New("pack archive contains an unsafe path")
		}
		switch header.Typeflag {
		case tar.TypeDir:
			if err := os.MkdirAll(target, 0o755); err != nil {
				return fmt.Errorf("create pack directory: %w", err)
			}
		case tar.TypeReg, tar.TypeRegA:
			if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
				return fmt.Errorf("create pack directory: %w", err)
			}
			output, err := os.OpenFile(target, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o644)
			if err != nil {
				return fmt.Errorf("create pack file: %w", err)
			}
			_, copyErr := io.Copy(output, reader)
			closeErr := output.Close()
			if copyErr != nil {
				return fmt.Errorf("extract pack file: %w", copyErr)
			}
			if closeErr != nil {
				return fmt.Errorf("close pack file: %w", closeErr)
			}
		default:
			return errors.New("pack archive contains an unsupported entry")
		}
	}
	return nil
}
