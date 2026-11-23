"""Selectively adapt MTP's per-builder staging after the baseline is frozen."""
from pathlib import Path
import shutil
w=Path(__file__).resolve().parents[1]
assert (w/'evidence/baseline-freeze.json').is_file(),'Baseline must be retained first'
v=w/'source/vllm';m=Path('/home/wulf/Desktop/Qfn-Mtp-vllm33')
p=v/'vllm/v1/attention/backends/gdn_spec_metadata.py'
s=(m/'upstream/vllm/vllm/v1/attention/backends/gdn_spec_metadata.py').read_text()
s=s.replace("import numpy as np", "import numpy as np\nfrom vllm.logger import init_logger\n_reported = False")
s=s.replace('        or not b.use_full_cuda_graph\n','')
s=s.replace('Accept uniform actual K0/1/3/5 with trailing zero-length padding.', 'Accept DFlash actual widths 1–8 with trailing zero-length padding.')
s=s.replace('    active, query_len = shape\n', '''    active, query_len = shape
    global _reported
    if not _reported:
        _reported = True
        init_logger(__name__).info(
            "DFlash per-builder GDN metadata active: reserved=%d actual_width=%d requests=%d",
            b.num_spec + 1, query_len, m.num_reqs)
''')
p.write_text(s)
p=v/'vllm/v1/attention/backends/gdn_attn.py';s=p.read_text()
# Opt in only this model, XPU and the fixed DFlash contract.
s=s.replace('import torch\n','import torch\nimport os\n\n_DFLASH_FUSED_METADATA = os.getenv("B70_DFLASH_FUSED_GDN_METADATA", "0") == "1"\n',1)
s=s.replace('if self.compilation_config.max_cudagraph_capture_size is not None:', '''if (
            self.compilation_config.max_cudagraph_capture_size is not None
            and not (_DFLASH_FUSED_METADATA and device.type == "xpu")
        ):''',1)
a='''        m = common_attn_metadata

'''
b='''        m = common_attn_metadata
        spec = self.speculative_config
        if (
            _DFLASH_FUSED_METADATA
            and self.use_spec_decode
            and type(self) is GDNAttentionMetadataBuilder
            and m.query_start_loc.device.type == "xpu"
            and spec.method == "dflash"
            and spec.num_speculative_tokens == 7
            and spec.draft_model_config.architectures == ["DFlash2DraftModel"]
        ):
            from .gdn_spec_metadata import try_build_uniform_spec_metadata
            metadata = try_build_uniform_spec_metadata(
                self, m, num_accepted_tokens, num_decode_draft_tokens_cpu
            )
            if metadata is not None:
                return metadata

'''
assert a in s;s=s.replace(a,b,1);p.write_text(s)
q=w/'qualification/metadata.py';s=(m/'astra/scripts/k7_metadata_probe.py').read_text()
s=s.replace('[(1,1,8),(2,2,4),(3,3,4),(4,4,4),(4,3,4)]', '[(1,1,q) for q in range(1,9)]+[(2,2,8),(3,3,8),(4,4,8),(4,3,8)]')
s=s.replace('with torch.xpu.graph(g):stage()', 'with torch.xpu.graph(g, stream=torch.xpu.Stream(device=rank)):stage()')
q.write_text(s)
print('Prepared isolated per-builder metadata candidate; released fallback retained.')
