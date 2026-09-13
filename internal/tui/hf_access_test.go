package tui

import (
	"context"
	"strings"
	"sync"
	"testing"

	"b70ctl/internal/hf"
)

type accessScriptClient struct {
	mu          sync.Mutex
	inspectErrs []error // consumed per Inspect call; nil error afterwards
	calls       int
}

func (client *accessScriptClient) Inspect(ctx context.Context, repo, revision string) (hf.Repository, error) {
	client.mu.Lock()
	defer client.mu.Unlock()
	client.calls++
	if client.calls <= len(client.inspectErrs) && client.inspectErrs[client.calls-1] != nil {
		return hf.Repository{}, client.inspectErrs[client.calls-1]
	}
	return hf.Repository{Repo: repo, Revision: revision, Files: []hf.File{{Path: "model.bin"}}}, nil
}

// Install is never reached by these tests: every scripted Inspect error
// aborts the flow first, and the successful retry stops at the download
// confirmation prompt.
func (client *accessScriptClient) Install(context.Context, string, hf.Repository, func(hf.Progress)) (hf.Result, error) {
	return hf.Result{}, nil
}

func TestDownloadModelAccessErrorsShowTokenGuidance(t *testing.T) {
	tests := []struct {
		name string
		err  error
		want []string
		ban  []string
	}{
		{
			name: "no token and access-hidden revision",
			err:  hf.ErrAccessUncertain,
			want: []string{"Could not access model revision", "No Hugging Face token is configured", "Settings → Hugging Face Token"},
			ban:  []string{"revision not found"},
		},
		{
			name: "authentication explicitly required",
			err:  hf.ErrAuthenticationRequired,
			want: []string{"Hugging Face access required", "No Hugging Face token is configured for this model", "Settings → Hugging Face Token"},
			ban:  []string{"revision not found"},
		},
		{
			name: "configured token denied",
			err:  hf.ErrAccessDenied,
			want: []string{"Hugging Face access denied", "The configured token could not access this model"},
			ban:  []string{"No Hugging Face token is configured", "revision not found"},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			client := &accessScriptClient{inspectErrs: []error{test.err}}
			stubHFClient(t, client)
			application, output := modelActionTestApp(t, []byte{27}, t.TempDir(), t.TempDir())
			ok, err := application.downloadModel(tuiTestModel())
			if ok || err != nil {
				t.Fatalf("downloadModel() = %v, %v", ok, err)
			}
			screen := strings.ToLower(stripANSI(readTestOutput(t, output)))
			for _, want := range test.want {
				if !strings.Contains(screen, strings.ToLower(want)) {
					t.Fatalf("screen missing %q:\n%s", want, screen)
				}
			}
			for _, banned := range test.ban {
				if strings.Contains(screen, strings.ToLower(banned)) {
					t.Fatalf("screen must not claim %q:\n%s", banned, screen)
				}
			}
		})
	}
}

func TestDownloadModelRetryAfterTokenSaveUsesNewTokenWithoutRestart(t *testing.T) {
	client := &accessScriptClient{inspectErrs: []error{hf.ErrAccessUncertain}}
	var tokens []string
	original := newHFClient
	newHFClient = func(token string) hfClient {
		tokens = append(tokens, token)
		return client
	}
	t.Cleanup(func() { newHFClient = original })

	// Enter dismisses the first attempt's access notice; the trailing Esc
	// (kept last so the input parser does not merge it into a sequence)
	// declines the retry's download confirmation.
	application, output := modelActionTestApp(t, []byte{'\n', 27}, t.TempDir(), t.TempDir())
	model := tuiTestModel()

	if ok, err := application.downloadModel(model); ok || err != nil {
		t.Fatalf("first downloadModel() = %v, %v", ok, err)
	}
	if screen := strings.ToLower(stripANSI(readTestOutput(t, output))); !strings.Contains(screen, "could not access model revision") {
		t.Fatalf("first attempt did not show the access guidance:\n%s", screen)
	}
	if err := hf.Set("retried-token"); err != nil {
		t.Fatal(err)
	}
	if ok, err := application.downloadModel(model); ok || err != nil {
		t.Fatalf("second downloadModel() = %v, %v", ok, err)
	}
	if got, want := strings.Join(tokens, ","), ",retried-token"; got != want {
		t.Fatalf("tokens passed to newHFClient = %q, want %q (retry must read the saved token without restart)", got, want)
	}
	if client.calls != 2 {
		t.Fatalf("Inspect calls = %d, want 2", client.calls)
	}
}
