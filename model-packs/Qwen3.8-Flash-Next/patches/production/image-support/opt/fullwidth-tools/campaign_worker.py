"""Performance development commands without installed audit hooks."""

import importlib
import json

from vllm.v1.worker.xpu_worker import XPUWorker


class CampaignWorker(XPUWorker):
    def campaign_command(self, payload):
        command = json.loads(payload)
        module = importlib.reload(importlib.import_module(command.pop("module")))
        return module.run(self, command)
