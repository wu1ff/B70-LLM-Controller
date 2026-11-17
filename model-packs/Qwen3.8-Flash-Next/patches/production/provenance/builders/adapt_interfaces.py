from pathlib import Path
import shutil
w=Path(__file__).resolve().parents[1];v=w/'source/vllm';m=Path('/home/wulf/Desktop/Qfn-Mtp-vllm33')
p=v/'vllm/models/qwen4_exp/nvidia/model.py';s=p.read_text();a=s.index('<<<<<<<');b=s.index('\nfrom .low_latency_gemm',a)
s=s[:a]+'''from ..common.hyperconnection import is_hc_sequence_parallel_enabled
if current_platform.is_xpu():
    from .b70_hyperconnection import GatedResidual, HyperConnectionConfig
else:
    from .hyperconnection import GatedResidual, HyperConnectionConfig
from .dflash2_features import AUX_LAYER_BOUNDARIES, DFlash2FeatureCapture
'''+s[b:];p.write_text(s)
# Keep the current common producer/consumer and descriptor-validated mmap.
p=v/'vllm/models/qwen4_exp/common/disk_ple.py';s=(m/'astra/candidates/ple-bounded-clear/overlay/opt/venv/lib/python3.12/site-packages/vllm/models/qwen4_exp/common/disk_ple.py').read_text()
a=s.index('    spec = config.speculative_config');b=s.index('    captures =',a)
old=(v/'vllm/models/qwen4_exp/nvidia/b70_disk.py').read_text();oa=old.index('    spec = config.speculative_config');ob=old.index('    if any(',oa)
s=s[:a]+'    mbt = scheduler.max_num_batched_tokens\n'+old[oa:ob]+s[b:]
s=s.replace('or parallel.use_ubatching','or parallel.use_ubatching\n        or parallel.use_sequence_parallel_moe\n        or parallel.enable_hc_sp')
s=s.replace('if (\n        max_total_tokens != mbt','if (\n        mbt != 1024\n        or max_total_tokens != mbt')
s=s.replace('# Disk transport by Wulf; adapted to v0.33 common PLE for Wulf with Codex.','# Disk transport by Wulf; adapted to v0.33 DFlash2 common PLE with Codex.')
p.write_text(s)
# Compatibility import for retained DFlash diagnostics.
(v/'vllm/models/qwen4_exp/nvidia/b70_disk.py').write_text('"""DFlash2 disk PLE uses the v0.33 common prefetch interface."""\nfrom ..common.disk_ple import Qwen4ExpPLEDiskBF16Embedding\n')
for rel in ['vllm/models/qwen4_exp/nvidia/ngram_embedding.py','vllm/models/qwen4_exp/nvidia/model_state.py']:
    p=v/rel;s=p.read_text()
    if p.name=='ngram_embedding.py':
        s=s.replace('from .b70_disk import','from ..common.disk_ple import')
        s=s.replace('if self.ngram_embedding.supports_prefetch:', 'if self.ngram_embedding.supports_prefetch and not isinstance(\n            self.ngram_embedding, Qwen4ExpPLEDiskBF16Embedding\n        ):')
        s=s.replace('        if os.environ.get("B70_PLE_DISK_BF16") == "1":\n            embedding_cls', '        if os.environ.get("B70_PLE_DISK_BF16") == "1":\n            if self.use_sequence_parallel:\n                raise ValueError("Disk BF16 PLE requires the original non-SP layout")\n            embedding_cls')
        s=s.replace('        persistent_buffers = {','        disk = isinstance(self.ngram_embedding, Qwen4ExpPLEDiskBF16Embedding)\n        if disk:\n            if getattr(self, "_disk_load_started", False):\n                raise RuntimeError("Disk BF16 PLE rejects weight/hash reload")\n            self._disk_load_started = True\n        persistent_buffers = {',1)
        s=s.replace('        if disk:\n            self.ngram_embedding.load_mmap', '        if disk:\n            if not persistent_buffers.keys() <= loaded:\n                raise ValueError("Disk BF16 PLE requires all checkpoint hash buffers")\n            self.ngram_embedding.load_mmap')
    else:
        s=s.replace('            Qwen4ExpPLEDiskBF16Embedding,\n','')
        s=s.replace('        self._disk_ple_modules = tuple(', '        from ..common.disk_ple import Qwen4ExpPLEDiskBF16Embedding\n\n        self._disk_ple_modules = tuple(')
        s=s.replace('            if num_tokens_padded == 0:\n                continue','            if num_tokens_padded == 0:\n                module.ngram_embedding.prepare(None, 0, 0)\n                continue')
        s=s.replace('                num_tokens_padded,\n            )','                num_tokens_padded,\n                is_padding=input_batch.is_padding,\n            )')
    p.write_text(s)
# The old shortcut lacks v0.33's hidden-state store. Released K0 keeps KV writes.
p=v/'vllm/v1/worker/gpu/spec_decode/dflash/speculator.py';s=p.read_text();a=s.index('        # An unfinished prompt chunk');b=s.index('        batch_sync,',a);s=s[:a]+s[b:];p.write_text(s)
# Avoid the old all-group metadata reuse: custom DFlash graphs own their clones.
p=v/'vllm/v1/attention/backends/gdn_attn.py';s=p.read_text().replace('and device.type in ("cuda", "xpu")','and device.type == "cuda"').replace('# Opts into MRV2\'s fused aligned-index precompute.','# Opts into MRV2\'s CUDA-only aligned-index precompute.');p.write_text(s)
# The accepted DFlash FA2 caller already supplies q_scale/k_scale/v_scale.
# The newer standalone layout validation maps the checked descriptor itself.
shutil.copyfile(m/'upstream/vllm/vllm/models/qwen4_exp/intel_checkpoint.py',v/'vllm/models/qwen4_exp/intel_checkpoint.py')
# Frontend and checkpoint preparation bytes are the accepted common assets.
shutil.copytree(m/'source/assets',w/'source/assets',dirs_exist_ok=True)
shutil.copyfile(m/'source/prepare_intel.py',w/'source/prepare_intel.py')
p=w/'source/prepare_intel.py';s=p.read_text().replace('upstream/vllm/vllm/models/qwen4_exp/intel_checkpoint.py','source/vllm/vllm/models/qwen4_exp/intel_checkpoint.py');p.write_text(s)
