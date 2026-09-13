package tui

import (
	"testing"

	"b70ctl/internal/catalog"
	"b70ctl/internal/packstore"
)

func TestCatalogEntryInstalledRequiresExactIDAndVersion(t *testing.T) {
	installed := []packstore.InstalledPack{{ID: "pack", Version: "1.0.0"}}
	if !catalogEntryInstalled(installed, catalog.Entry{ID: "pack", Version: "1.0.0"}) {
		t.Fatal("exact catalog entry was not installed")
	}
	if catalogEntryInstalled(installed, catalog.Entry{ID: "pack", Version: "1.1.0"}) {
		t.Fatal("different version was marked installed")
	}
}

func TestSourceNameIncludesRemote(t *testing.T) {
	if sourceName(packstore.SourceLocal) != "Local" || sourceName(packstore.SourceRemote) != "Remote" {
		t.Fatalf("source names = %q, %q", sourceName(packstore.SourceLocal), sourceName(packstore.SourceRemote))
	}
}
