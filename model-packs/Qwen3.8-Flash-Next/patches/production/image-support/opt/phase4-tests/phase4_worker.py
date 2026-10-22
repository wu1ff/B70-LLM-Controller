"""Campaign worker with explicit, removable performance instrumentation."""

import importlib
import json
import os
from pathlib import Path
from types import MethodType

from phase3_worker import Phase3Worker


class Phase4Worker(Phase3Worker):
    def load_model(self, **kwargs):
        super().load_model(**kwargs)
        from vllm.models.qwen4_exp.nvidia import c1_verify

        self._phase4_audit_verify = c1_verify.verify
        self._phase4_audit_propose = self.model_runner.speculator.propose
        self._audit_steps = 10000
        self.phase4_command('{"action":"clean"}')
        if os.getenv("PHASE4_QUALIFY_ON_LOAD") == "1":
            self.phase4_command('{"action":"regression"}')

    def phase4_command(self, payload):
        from vllm.models.qwen4_exp.nvidia import c1_verify

        command = json.loads(payload)
        action = command["action"]
        runner = self.model_runner
        if action == "clean":
            if hasattr(self, "_phase4_profiler"):
                self._phase4_profiler.close()
                del self._phase4_profiler
            if hasattr(self, "_phase4_hc_graph_methods"):
                import torch

                torch.xpu.synchronize()
                for module, name, original in self._phase4_hc_graph_methods:
                    setattr(module, name, original)
                del self._phase4_hc_graph_methods
                del self._phase4_hc_graph_entries
            spec = runner.speculator
            spec.propose = MethodType(type(spec).propose, spec)
            sampler = spec.candidate_sampler
            sampler.sample = MethodType(type(sampler).sample, sampler)
            importlib.reload(c1_verify)
            for module in runner.model.modules():
                for kind in ("_forward_hooks", "_forward_pre_hooks"):
                    hooks = getattr(module, kind)
                    for key, hook in list(hooks.items()):
                        if getattr(hook, "__module__", "").startswith(
                            ("phase2_", "phase3_", "phase4_")
                        ):
                            del hooks[key]
                            getattr(module, kind + "_with_kwargs", {}).pop(key, None)
                            getattr(module, kind + "_always_called", {}).pop(key, None)
            return {"rank": self.rank, "clean": True}
        if action == "profile":
            import phase4_profile

            importlib.reload(phase4_profile)
            self._phase4_profiler = phase4_profile.Profile(self, command)
            return {"rank": self.rank, "profile": command}
        if action == "regression":
            # The observer must time/capture the candidate being qualified.
            for name, cell in zip(
                self._phase4_audit_verify.__code__.co_freevars,
                self._phase4_audit_verify.__closure__,
            ):
                if name == "vector_verify":
                    cell.cell_contents = c1_verify.verify
            c1_verify.verify = self._phase4_audit_verify
            runner.speculator.propose = self._phase4_audit_propose
            return {"rank": self.rank, "regression": True}
        if action == "experiment":
            module = importlib.reload(importlib.import_module(command["module"]))
            return module.run(self, command)
        if action == "resources":
            import torch

            return {
                "rank": self.rank,
                "allocated": torch.xpu.memory_allocated(),
                "reserved": torch.xpu.memory_reserved(),
                "graph_reserved": torch.xpu.memory_stats().get(
                    "reserved_bytes.all.current"
                ),
                "num_blocks": runner.kv_cache_config.num_blocks,
            }
        raise ValueError(action)
