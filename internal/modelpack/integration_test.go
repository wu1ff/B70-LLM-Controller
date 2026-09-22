package modelpack

import (
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"
)

func publicQwenPackRoot() string {
	return filepath.Join("..", "..", "model-packs", "Qwen3.8-27B")
}

func TestPublicQwenPackRenderEquivalence(t *testing.T) {
	authoritativeRoot := os.Getenv("B70_ACCEPTED_EXPORT_PATH")
	if authoritativeRoot == "" {
		t.Skip("set B70_ACCEPTED_EXPORT_PATH to run the accepted-export render proof")
	}
	root := publicQwenPackRoot()
	manifest, err := Load(root)
	if err != nil {
		t.Fatal(err)
	}
	if len(manifest.Models) != 5 || len(manifest.Runtimes) != 1 || len(manifest.Modes) != 3 || len(manifest.Profiles) != 96 {
		t.Fatalf("pack counts = models=%d runtimes=%d modes=%d profiles=%d", len(manifest.Models), len(manifest.Runtimes), len(manifest.Modes), len(manifest.Profiles))
	}

	program := `import importlib.util,json,pathlib,sys
root=pathlib.Path(sys.argv[1])
spec=importlib.util.spec_from_file_location("pack_validate", root/"validate.py")
module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
pack=module.load(root/"pack.json")
json.dump({p["id"]:module.render(pack,p["id"]) for p in pack["profiles"]},sys.stdout,sort_keys=True)`
	command := exec.Command("python3", "-c", program, authoritativeRoot)
	command.Env = append(os.Environ(), "PYTHONDONTWRITEBYTECODE=1")
	output, err := command.Output()
	if err != nil {
		t.Fatalf("canonical renderer: %v", err)
	}
	var canonical map[string]canonicalResolved
	if err := json.Unmarshal(output, &canonical); err != nil {
		t.Fatal(err)
	}
	if len(canonical) != len(manifest.Profiles) {
		t.Fatalf("canonical profiles = %d, public profiles = %d", len(canonical), len(manifest.Profiles))
	}
	for _, profile := range manifest.Profiles {
		got, err := Resolve(manifest, profile)
		if err != nil {
			t.Fatalf("%s: %v", profile.ID, err)
		}
		want := canonical[profile.ID]
		if got.ContainerPort != want.ContainerPort || got.HealthPath != want.HealthPath || got.Image != want.Image || got.Digest != want.Digest ||
			!reflect.DeepEqual(got.DockerArgs, want.DockerArgs) || !reflect.DeepEqual(got.Environment, want.Environment) ||
			!reflect.DeepEqual(got.Mounts, want.Mounts) || !reflect.DeepEqual(got.Command, want.Command) {
			t.Fatalf("%s Controller render differs\n got: %#v\nwant: %#v", profile.ID, got, want)
		}
	}
	t.Logf("Controller render equivalence: PASS %d/%d", len(manifest.Profiles), len(canonical))
}

func TestPublicQwenPackServePositionalOrder(t *testing.T) {
	manifest, err := Load(publicQwenPackRoot())
	if err != nil {
		t.Fatal(err)
	}
	models := make(map[string]Model, len(manifest.Models))
	for _, model := range manifest.Models {
		models[model.ID] = model
	}
	for _, profile := range manifest.Profiles {
		resolved, err := Resolve(manifest, profile)
		if err != nil {
			t.Fatalf("%s: %v", profile.ID, err)
		}
		mountPath := models[profile.ModelID].MountPath
		serve, err := ServeCommand(resolved, mountPath)
		if err != nil {
			t.Fatalf("%s: %v", profile.ID, err)
		}
		if serve[0] != "vllm" || serve[1] != "serve" || serve[2] != mountPath {
			t.Fatalf("%s serve prefix = %#v, want [vllm serve %s]", profile.ID, serve[:3], mountPath)
		}
		for index, token := range serve {
			if token == mountPath && index != 2 {
				t.Fatalf("%s model positional also appears at index %d", profile.ID, index)
			}
		}
	}
	t.Logf("vLLM serve positional-order: PASS %d/%d", len(manifest.Profiles), len(manifest.Profiles))
}

func TestPublicQwenPackFinalRuntimeContract(t *testing.T) {
	manifest, err := Load(publicQwenPackRoot())
	if err != nil {
		t.Fatal(err)
	}
	wantDigest := "sha256:314786fd704d5393630e4e292a60bc30e5ade1fa4aa372cf986106e16828c90f"
	wantRegistry := "ghcr.io/wu1ff/qwen38-27b-b70@" + wantDigest
	if len(manifest.Runtimes) != 1 || manifest.Runtimes[0].Digest != wantDigest || manifest.Runtimes[0].Registry != wantRegistry {
		t.Fatalf("runtime authority = %#v", manifest.Runtimes)
	}
	for _, profile := range manifest.Profiles {
		resolved, err := Resolve(manifest, profile)
		if err != nil {
			t.Fatalf("%s: %v", profile.ID, err)
		}
		if resolved.Digest != wantDigest || resolved.Environment["CCL_SYCL_ALLREDUCE_TMP_BUF"] != "1" || resolved.Environment["CCL_SYCL_ALLGATHERV_TMP_BUF"] != "1" {
			t.Fatalf("%s final runtime contract = digest %q environment %#v", profile.ID, resolved.Digest, resolved.Environment)
		}
	}
	t.Logf("final runtime contract: PASS %d/%d", len(manifest.Profiles), len(manifest.Profiles))
}

func TestPublicQwenPackGraphPolicyContract(t *testing.T) {
	manifest, err := Load(publicQwenPackRoot())
	if err != nil {
		t.Fatal(err)
	}
	graph64 := `"cudagraph_mode":"PIECEWISE","cudagraph_capture_sizes":[1,2,4,8,16,32,64],"max_cudagraph_capture_size":64`
	piecewise8 := `"cudagraph_mode":"PIECEWISE","cudagraph_capture_sizes":[1,2,4,8],"max_cudagraph_capture_size":8`
	baseGraph := `"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[1],"max_cudagraph_capture_size":1`
	mtpGraph := `"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[1,2,4],"max_cudagraph_capture_size":4`
	models := make(map[string]Model, len(manifest.Models))
	for _, model := range manifest.Models {
		models[model.ID] = model
	}
	graph64Profiles, tp1DFlashProfiles := 0, 0
	for _, profile := range manifest.Profiles {
		resolved, err := Resolve(manifest, profile)
		if err != nil {
			t.Fatalf("%s: %v", profile.ID, err)
		}
		serve, err := ServeCommand(resolved, models[profile.ModelID].MountPath)
		if err != nil {
			t.Fatalf("%s: %v", profile.ID, err)
		}
		joined := strings.Join(serve, " ")
		if strings.Count(joined, "--compilation-config") != 1 {
			t.Fatalf("%s: expected exactly one compilation-config, serve = %s", profile.ID, joined)
		}
		switch {
		case profile.Mode == "dflash2" && profile.Cards == 1:
			if !strings.Contains(joined, piecewise8) || strings.Contains(joined, graph64) {
				t.Fatalf("%s: TP1 dFlash2 must keep the unchanged PIECEWISE8 ceiling", profile.ID)
			}
			tp1DFlashProfiles++
		case profile.Mode == "dflash2":
			if !strings.Contains(joined, graph64) || strings.Contains(joined, piecewise8) {
				t.Fatalf("%s: TP2/TP4 dFlash2 must use the widened PIECEWISE64 capture", profile.ID)
			}
			graph64Profiles++
		case profile.Mode == "base":
			if !strings.Contains(joined, baseGraph) {
				t.Fatalf("%s: Base graph policy changed", profile.ID)
			}
		case profile.Mode == "mtp":
			if !strings.Contains(joined, mtpGraph) {
				t.Fatalf("%s: MTP1 graph policy changed", profile.ID)
			}
		}
	}
	if graph64Profiles != 28 || tp1DFlashProfiles != 2 {
		t.Fatalf("dFlash2 graph-policy split = graph64 %d, tp1 %d; want 28, 2", graph64Profiles, tp1DFlashProfiles)
	}
	t.Logf("graph policy: PASS graph64_tp2_tp4=%d tp1_piecewise8=%d base/mtp unchanged", graph64Profiles, tp1DFlashProfiles)
}

func TestPublicQwenPackRepresentativeFP8TP264KDFlash2Contract(t *testing.T) {
	manifest, err := Load(publicQwenPackRoot())
	if err != nil {
		t.Fatal(err)
	}
	var selected *Profile
	for index := range manifest.Profiles {
		if manifest.Profiles[index].ID == "fp8-dflash2-tp2-65536" {
			selected = &manifest.Profiles[index]
			break
		}
	}
	if selected == nil {
		t.Fatal("representative FP8 TP2 64K dFlash2 profile is missing")
	}
	if selected.ModelID != "qwen38-27b-fp8" || selected.Cards != 2 || selected.TensorParallel != 2 || selected.Context != 65536 || selected.Mode != "dflash2" {
		t.Fatalf("representative profile = %#v", *selected)
	}
	resolved, err := Resolve(manifest, *selected)
	if err != nil {
		t.Fatal(err)
	}
	if resolved.Digest != "sha256:314786fd704d5393630e4e292a60bc30e5ade1fa4aa372cf986106e16828c90f" || resolved.Environment["CCL_SYCL_ALLREDUCE_TMP_BUF"] != "1" || resolved.Environment["CCL_SYCL_ALLGATHERV_TMP_BUF"] != "1" {
		t.Fatalf("representative resolved contract = %#v", resolved)
	}
	if len(resolved.Mounts) != 2 || resolved.Mounts[0].ModelID != "qwen38-27b-fp8" || resolved.Mounts[1].ModelID != "qwen38-27b-dflash2" {
		t.Fatalf("representative mounts = %#v", resolved.Mounts)
	}
	t.Logf("representative contract: profile=%s digest=%s allreduce=%s allgatherv=%s mounts=%d", selected.ID, resolved.Digest, resolved.Environment["CCL_SYCL_ALLREDUCE_TMP_BUF"], resolved.Environment["CCL_SYCL_ALLGATHERV_TMP_BUF"], len(resolved.Mounts))
}

type canonicalResolved struct {
	ContainerPort int               `json:"container_port"`
	HealthPath    string            `json:"health_path"`
	DockerArgs    []string          `json:"docker_args"`
	Environment   map[string]string `json:"environment"`
	Mounts        []Mount           `json:"mounts"`
	Image         string            `json:"image"`
	Digest        string            `json:"digest"`
	Command       []string          `json:"command"`
}

func TestPublicQwenPackMatrix(t *testing.T) {
	manifest, err := Load(publicQwenPackRoot())
	if err != nil {
		t.Fatal(err)
	}
	if len(manifest.Models) != 5 || len(manifest.Runtimes) != 1 || len(manifest.Modes) != 3 || len(manifest.Profiles) != 96 {
		t.Fatalf("pack counts = models=%d runtimes=%d modes=%d profiles=%d", len(manifest.Models), len(manifest.Runtimes), len(manifest.Modes), len(manifest.Profiles))
	}
	targets, assistants := 0, 0
	for _, model := range manifest.Models {
		switch model.Kind {
		case "target":
			targets++
		case "assistant":
			assistants++
		}
	}
	if targets != 4 || assistants != 1 {
		t.Fatalf("model kinds = targets=%d assistants=%d", targets, assistants)
	}
	counts := map[string]int{}
	for _, profile := range manifest.Profiles {
		counts[profile.ModelID]++
	}
	want := map[string]int{"qwen38-27b-fp8": 18, "qwen38-27b-uncensored-fp8": 18, "qwen38-27b-int4": 30, "qwen38-27b-uncensored-int4": 30}
	if !reflect.DeepEqual(counts, want) {
		t.Fatalf("matrix breakdown = %v, want %v", counts, want)
	}
	labels := make([]string, 0, len(manifest.Modes))
	for _, mode := range manifest.Modes {
		labels = append(labels, mode.DisplayName)
	}
	sort.Strings(labels)
	if !reflect.DeepEqual(labels, []string{"Base", "MTP1", "dFlash2"}) {
		t.Fatalf("mode labels = %v", labels)
	}
	unsupported := []qualification{
		{modelID: "qwen38-27b-fp8", cards: 1, context: 32768, mode: "base"},
		{modelID: "qwen38-27b-fp8", cards: 2, context: 131072, mode: "base"},
		{modelID: "qwen38-27b-fp8", cards: 2, context: 131072, mode: "mtp"},
		{modelID: "qwen38-27b-fp8", cards: 2, context: 131072, mode: "dflash2"},
		{modelID: "qwen38-27b-fp8", cards: 2, context: 262144, mode: "base"},
		{modelID: "qwen38-27b-fp8", cards: 2, context: 262144, mode: "mtp"},
		{modelID: "qwen38-27b-fp8", cards: 2, context: 262144, mode: "dflash2"},
		{modelID: "qwen38-27b-uncensored-fp8", cards: 1, context: 32768, mode: "base"},
		{modelID: "qwen38-27b-uncensored-fp8", cards: 2, context: 131072, mode: "base"},
		{modelID: "qwen38-27b-uncensored-fp8", cards: 2, context: 131072, mode: "mtp"},
		{modelID: "qwen38-27b-uncensored-fp8", cards: 2, context: 131072, mode: "dflash2"},
		{modelID: "qwen38-27b-uncensored-fp8", cards: 2, context: 262144, mode: "base"},
		{modelID: "qwen38-27b-uncensored-fp8", cards: 2, context: 262144, mode: "mtp"},
		{modelID: "qwen38-27b-uncensored-fp8", cards: 2, context: 262144, mode: "dflash2"},
		{modelID: "qwen38-27b-int4", cards: 1, context: 131072, mode: "mtp"},
		{modelID: "qwen38-27b-int4", cards: 1, context: 65536, mode: "dflash2"},
		{modelID: "qwen38-27b-uncensored-int4", cards: 1, context: 131072, mode: "mtp"},
		{modelID: "qwen38-27b-uncensored-int4", cards: 1, context: 65536, mode: "dflash2"},
	}
	rows := map[qualification]struct{}{}
	for _, profile := range manifest.Profiles {
		rows[qualification{modelID: profile.ModelID, cards: profile.Cards, context: profile.Context, mode: profile.Mode}] = struct{}{}
	}
	for _, item := range unsupported {
		if _, exists := rows[item]; exists {
			t.Fatalf("unsupported qualification is present: %#v", item)
		}
	}
	if len(rows) != 96 {
		t.Fatalf("selector rows = %d", len(rows))
	}
}
