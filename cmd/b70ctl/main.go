package main

import (
	"fmt"
	"os"
	"os/signal"
	"sync"
	"syscall"

	"golang.org/x/term"

	"b70ctl/internal/catalog"
	"b70ctl/internal/config"
	"b70ctl/internal/hf"
	"b70ctl/internal/packstore"
	"b70ctl/internal/tempsweep"
	"b70ctl/internal/tui"
)

var version = "dev"

func main() {
	if len(os.Args) == 2 && os.Args[1] == "--version" {
		fmt.Printf("B70 LLM Controller %s\n", version)
		return
	}
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "b70ctl:", err)
		os.Exit(1)
	}
}

func run() error {
	if !term.IsTerminal(int(os.Stdin.Fd())) || !term.IsTerminal(int(os.Stdout.Fd())) {
		return fmt.Errorf("an interactive terminal is required")
	}
	paths, err := config.ResolvePaths()
	if err != nil {
		return err
	}
	value, err := config.Load(paths.ConfigFile)
	if err != nil {
		return err
	}
	// Startup runs before any Controller download or import operation can
	// exist, so every matched leftover is a crash orphan. Resumable model
	// staging is preserved by design: it never matches these prefixes.
	if _, err := tempsweep.Sweep(
		tempsweep.Root{Path: value.ModelDirectory, Prefix: hf.LegacyDownloadPrefix},
		tempsweep.Root{Path: paths.DataRoot, Prefix: catalog.DownloadTempPrefix},
		tempsweep.Root{Path: paths.PacksRoot, Prefix: packstore.ImportTempPrefix},
	); err != nil {
		fmt.Fprintln(os.Stderr, "b70ctl: warning: skipped temporary cleanup:", err)
	}
	state, err := term.MakeRaw(int(os.Stdin.Fd()))
	if err != nil {
		return fmt.Errorf("enable terminal raw mode: %w", err)
	}
	var once sync.Once
	restore := func() {
		once.Do(func() {
			_ = term.Restore(int(os.Stdin.Fd()), state)
			fmt.Fprint(os.Stdout, "\x1b[0m\x1b[?25h")
		})
	}
	defer restore()

	signals := make(chan os.Signal, 1)
	signal.Notify(signals, syscall.SIGHUP, syscall.SIGTERM, syscall.SIGQUIT)
	defer signal.Stop(signals)
	go func() {
		signal := <-signals
		restore()
		if number, ok := signal.(syscall.Signal); ok {
			os.Exit(128 + int(number))
		}
		os.Exit(1)
	}()

	return tui.Run(os.Stdin, os.Stdout, value, paths, version)
}
