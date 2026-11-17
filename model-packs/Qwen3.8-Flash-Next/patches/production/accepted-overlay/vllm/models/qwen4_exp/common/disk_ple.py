# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Disk transport by Wulf; adapted to v0.33 DFlash2 common PLE with Codex.
"""Checkpoint-specific BF16 mmap transport; no resident or UVA table allocation."""

import ctypes
import os
import threading
from typing import ClassVar

import torch

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import get_current_vllm_config
from vllm.logger import init_logger

from .ngram_embedding import (
    Qwen4ExpPLEEmbedding,
    Qwen4ExpPLEEmbeddingMethod,
    Qwen4ExpPLEPinnedHostEmbedding,
    Qwen4ExpPLEUnquantizedEmbeddingMethod,
)

logger = init_logger(__name__)

# Keep DMA owners alive if a device fence fails. Process teardown is the only
# safe release in that case; never fall back to a different table mid-forward.
_disk_bf16_quarantine: list[object] = []


def _ple_disk_bf16_enabled() -> bool:
    return os.environ.get("B70_PLE_DISK_BF16") == "1"


def _validate_disk_bf16_config(
    embedding_method: Qwen4ExpPLEEmbeddingMethod,
    params_dtype: torch.dtype,
    max_total_tokens: int,
) -> int:
    config = get_current_vllm_config()
    parallel = config.parallel_config
    scheduler = config.scheduler_config
    if os.environ.get("B70_PLE_DIRECT_PINNED") == "1":
        raise ValueError("Disk BF16 PLE and #008 direct-pinned loading are exclusive")
    if any(
        os.environ.get(name) == "1"
        for name in (
            "B70_PLE_FP8",
            "B70_PLE_INT8",
            "B70_PLE_INT8_NVME",
            "B70_PLE_INT8_NVME_LOOKAHEAD",
        )
    ):
        raise ValueError("Disk BF16 PLE cannot enable another PLE storage gate")
    if (
        not _is_xpu()
        or params_dtype != torch.bfloat16
        or not isinstance(embedding_method, Qwen4ExpPLEUnquantizedEmbeddingMethod)
        or not os.environ.get("PLE_TABLE_PATH")
        or parallel.tensor_parallel_size != 4
        or not parallel.enable_expert_parallel
        or parallel.pipeline_parallel_size != 1
        or parallel.data_parallel_size != 1
        or parallel.use_ubatching
        or parallel.use_sequence_parallel_moe
        or parallel.enable_hc_sp
        or parallel.decode_context_parallel_size != 1
        or parallel.prefill_context_parallel_size != 1
        or scheduler.max_num_seqs != 4
        or getattr(config.cache_config, "kv_offloading_size", None)
    ):
        raise ValueError(
            "Disk BF16 PLE requires the frozen XPU BF16 "
            "TP4+EP/PP1/DP1, four-slot contract"
        )
    # Resolve the actual scheduler, including the default async subclass.
    # Exact class identities reject arbitrary subclasses and method overrides.
    from vllm.v1.core.sched.async_scheduler import AsyncScheduler
    from vllm.v1.core.sched.scheduler import Scheduler

    resolved = scheduler.get_scheduler_cls()
    if (
        resolved not in (Scheduler, AsyncScheduler)
        or resolved.schedule is not Scheduler.schedule
        or resolved._mamba_block_aligned_split
        is not Scheduler._mamba_block_aligned_split
    ):
        raise ValueError(
            "Disk BF16 PLE requires the retained aggregate-budget scheduler contract"
        )
    mbt = scheduler.max_num_batched_tokens
    if scheduler.max_num_scheduled_tokens not in (None, mbt):
        raise ValueError("Disk BF16 PLE requires the original aggregate budget")
    spec = config.speculative_config
    if spec is not None:
        draft = spec.draft_model_config
        if (
            spec.method != "dflash"
            or draft.architectures != ["DFlash2DraftModel"]
            or draft.dtype != torch.bfloat16
            or draft.quantization is not None
            or spec.draft_parallel_config.tensor_parallel_size != 4
            or not 1 <= spec.num_speculative_tokens
            < spec.draft_model_config.hf_config.dflash_config.get("block_size", 0)
            or spec.draft_sample_method != "probabilistic"
            or spec.rejection_sample_method != "standard"
            or spec.uses_dynamic_speculative_decoding()
            or spec.enable_adaptive_verification
        ):
            raise ValueError(
                "Disk PLE speculation requires BF16 TP4 DFlash2, trained-width "
                "proposals, conditional probabilistic drafting and standard rejection"
            )
    captures = config.compilation_config.cudagraph_capture_sizes or []
    capacity = max([max_total_tokens, *captures])
    if (
        mbt != 1024
        or max_total_tokens != mbt
        or capacity != mbt
        or any(not isinstance(n, int) or n <= 0 or n > mbt for n in captures)
    ):
        raise ValueError(
            "Disk BF16 PLE effective dispatch/capture capacity must equal MBT"
        )
    return capacity


def _protect_disk_bf16_mapping(
    table: torch.Tensor, path: str, file_stat: os.stat_result
) -> None:
    """Make the torch-owned file mapping read-only, without touching its pages."""
    page_size = os.sysconf("SC_PAGE_SIZE")
    start = table.data_ptr() // page_size * page_size
    end = (
        (table.data_ptr() + table.numel() * table.element_size() + page_size - 1)
        // page_size
        * page_size
    )
    stat = os.stat(path)
    if (stat.st_dev, stat.st_ino, stat.st_size) != (
        file_stat.st_dev,
        file_stat.st_ino,
        file_stat.st_size,
    ):
        raise ValueError("Intel PLE file changed during mmap setup")
    cursor = start
    with open("/proc/self/maps") as maps:
        for line in maps:
            fields = line.split(maxsplit=5)
            left, right = (int(value, 16) for value in fields[0].split("-"))
            if right <= cursor or left >= end:
                continue
            major, minor = (int(value, 16) for value in fields[3].split(":"))
            if (
                left > cursor
                or "p" not in fields[1]
                or int(fields[4]) != stat.st_ino
                or os.makedev(major, minor) != stat.st_dev
            ):
                raise ValueError(
                    "PLE tensor is not a private mmap of the selected BF16 file"
                )
            cursor = min(right, end)
    if cursor != end:
        raise ValueError("PLE tensor mapping does not cover the selected BF16 table")
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mprotect.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    libc.mprotect.restype = ctypes.c_int
    if libc.mprotect(start, end - start, 1) != 0:  # PROT_READ
        raise OSError(ctypes.get_errno(), "Cannot protect BF16 PLE mmap")


def _is_xpu() -> bool:
    from vllm.platforms import current_platform

    return current_platform.is_xpu()


class Qwen4ExpPLEDiskBF16Embedding(Qwen4ExpPLEPinnedHostEmbedding):
    """Immutable torch mmap plus one serialized, bounded XPU DMA stage.

    The common prefetch consumer reduces owned rows and copies them into
    graph-owned output. The producer runs before forward or FULL replay.
    """

    disk_bf16: ClassVar[bool] = True

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        params_dtype: torch.dtype,
        padding_size: int,
        prefix: str,
        embedding_method: Qwen4ExpPLEEmbeddingMethod,
        num_ngram_heads: int = 1,
        max_total_tokens: int = 0,
        data_parallel_rank: int = 0,
        reduce_results: bool = True,
    ) -> None:
        capacity = _validate_disk_bf16_config(
            embedding_method, params_dtype, max_total_tokens
        )
        if (num_embeddings, embedding_dim, num_ngram_heads) != (320001536, 160, 16):
            raise ValueError("Disk BF16 PLE requires the original B70 table layout")
        if not reduce_results:
            raise ValueError("Disk BF16 PLE requires the original non-SP reduction")
        Qwen4ExpPLEEmbedding.__init__(
            self,
            num_embeddings,
            embedding_dim,
            params_dtype=params_dtype,
            padding_size=padding_size,
            prefix=prefix,
            embedding_method=embedding_method,
            num_ngram_heads=num_ngram_heads,
            max_total_tokens=max_total_tokens,
            data_parallel_rank=data_parallel_rank,
            reduce_results=reduce_results,
        )
        if self.etp_data_parallel_size != 1 or self.tp_size != 4:
            raise ValueError("Disk BF16 PLE requires ETP=TP=4")
        self._stream_mod = torch.xpu
        self._capacity = capacity
        self._heads = num_ngram_heads
        self._output_dim = num_ngram_heads * embedding_dim
        self._prefetch_buffer = torch.empty(
            capacity,
            num_ngram_heads,
            embedding_dim,
            dtype=params_dtype,
            device="xpu",
        )
        self._stage = torch.empty(
            capacity,
            num_ngram_heads,
            embedding_dim,
            dtype=params_dtype,
            device="cpu",
            pin_memory=True,
        )
        if not self._stage.is_pinned():
            raise RuntimeError("Disk BF16 PLE stage failed to pin on XPU")
        self._mmap_owner = None
        self._table = None
        self._pending = False
        self._copy_event = torch.xpu.Event()
        self._owner_stream = None
        self._owner_thread = None
        self._lock = threading.Lock()
        self._failed = False
        self._closed = False
        self._prepared_tokens = 0

    def allocate_embedding_weight(
        self, num_embeddings: int, embedding_dim: int, dtype: torch.dtype
    ) -> torch.Tensor:
        # Metadata/loaded-name compatibility only: never reserve a TP shard.
        return torch.empty(0, embedding_dim, dtype=dtype, device="cpu")

    def weight_loader(self, *args, **kwargs) -> None:
        raise RuntimeError("Disk BF16 PLE rejects checkpoint row-shards and reloads")

    def load_mmap(self, path: str) -> None:
        if self._mmap_owner is not None or self._closed or self._failed:
            raise RuntimeError("Disk BF16 PLE rejects mmap reload")
        from ..intel_checkpoint import map_ple_table

        owner = map_ple_table(path)
        table = owner["table"]
        if (
            not isinstance(table, torch.Tensor)
            or table.device.type != "cpu"
            or table.dtype != torch.bfloat16
            or tuple(table.shape) != (self.org_vocab_size, self.embedding_dim)
            or not table.is_contiguous()
            or table.requires_grad
            or table.is_pinned()
        ):
            raise ValueError("Disk BF16 PLE requires a contiguous CPU BF16 table")
        _protect_disk_bf16_mapping(table, path, owner["file_stat"])
        self._mmap_owner = owner
        self._table = table
        logger.info(
            "Disk BF16 PLE mmap ready: shape=%s, capacity=%d, pinned_stage_bytes=%d, "
            "weight_elements=%d, tp_range=[%d,%d)",
            tuple(table.shape),
            self._capacity,
            self._stage.numel() * self._stage.element_size(),
            self.weight.numel(),
            self.shard_indices.org_vocab_start_index,
            self.shard_indices.org_vocab_end_index,
        )

    def _check_capacity(self, num_tokens: int) -> None:
        if not 0 <= num_tokens <= self._capacity:
            raise ValueError(
                f"Disk BF16 PLE stage overflow: {num_tokens} > {self._capacity}"
            )
        if self._closed or self._failed:
            raise RuntimeError("Disk BF16 PLE stage is closed or quarantined")
        if self._stream_mod.is_current_stream_capturing():
            raise RuntimeError("Disk BF16 PLE cannot prepare host rows inside capture")

    def _wait_for_stage(self) -> None:
        if self._pending:
            try:
                self._copy_event.synchronize()
            except BaseException:
                self._failed = True
                _disk_bf16_quarantine.append(self)
                raise
            self._pending = False

    def _fill_stage(
        self,
        ids_cpu: torch.Tensor,
        num_tokens: int,
        is_padding_cpu: torch.Tensor | None,
        num_tokens_padded: int,
    ) -> None:
        """CPU-only selected-row gather; pads/non-owned contributions are zero."""
        # Only this prefix is copied to the device producer for this dispatch.
        # Clear padded rows as well as actual rows, preserving graph padding.
        self._stage[:num_tokens_padded].zero_()
        ids = ids_cpu[:num_tokens].reshape(-1)
        active = torch.ones_like(ids, dtype=torch.bool)
        if is_padding_cpu is not None:
            active = (~is_padding_cpu[:num_tokens]).repeat_interleave(self._heads)
        active_ids = ids[active]
        if active_ids.numel() and (
            active_ids.min() < 0 or active_ids.max() >= self.org_vocab_size
        ):
            raise ValueError("Disk BF16 PLE received an out-of-table row ID")
        owned = (
            active
            & (ids >= self.shard_indices.org_vocab_start_index)
            & (ids < self.shard_indices.org_vocab_end_index)
        )
        # Advanced indexing allocates at most dispatch capacity, never a shard.
        stage = self._stage[:num_tokens].view(-1, self.embedding_dim)
        stage[owned] = self._table[ids[owned]]

    def prepare(
        self,
        ngram_ids: torch.Tensor | None,
        num_tokens: int,
        num_tokens_padded: int,
        *,
        is_padding: torch.Tensor | None = None,
    ) -> None:
        """Refresh the fixed GPU producer outside forward, including FULL replay."""
        self._check_capacity(num_tokens_padded)
        if not 0 <= num_tokens <= num_tokens_padded:
            raise ValueError("Disk BF16 PLE invalid actual/padded token counts")
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Concurrent Disk BF16 PLE stage preparation")
        try:
            stream = self._stream_mod.current_stream()
            thread = threading.get_ident()
            # The worker RPC loop serializes prepare -> forward/replay. Host
            # ownership may change only between these submitted dispatches.
            # Recheck after acquiring the lock: close/failure may have raced us.
            self._check_capacity(num_tokens_padded)
            self._wait_for_stage()  # CPU DMA source must be safe before refill.
            if self._owner_stream is not None and (
                stream != self._owner_stream or thread != self._owner_thread
            ):
                # The copy event precedes the model's GPU reads. Drain the OLD
                # stream as well, including FULL replay which bypasses Python.
                # This occurs only on ownership changes, outside capture; the
                # ordinary same-owner path retains current-stream ordering.
                try:
                    self._owner_stream.synchronize()
                except BaseException:
                    self._failed = True
                    _disk_bf16_quarantine.append(self)
                    raise
            # Publish the new owner only after both fences have succeeded.
            self._owner_stream, self._owner_thread = stream, thread
            if ngram_ids is None:
                # Deterministic dummy contents before warmup/capture; no table IO.
                self._stage[:num_tokens_padded].zero_()
            else:
                if (
                    self._table is None
                    or ngram_ids.dtype != torch.int64
                    or tuple(ngram_ids.shape) != (num_tokens_padded, self._heads)
                    or ngram_ids.device != self._prefetch_buffer.device
                ):
                    raise ValueError("Disk BF16 PLE missing mmap or invalid IDs")
                if is_padding is not None and (
                    is_padding.dtype != torch.bool
                    or tuple(is_padding.shape) != (num_tokens_padded,)
                    or is_padding.device != self._prefetch_buffer.device
                ):
                    raise ValueError("Disk BF16 PLE invalid padding mask")
                ids_cpu = ngram_ids.to(device="cpu", non_blocking=False)
                is_padding_cpu = (
                    is_padding.to(device="cpu", non_blocking=False)
                    if is_padding is not None
                    else None
                )
                self._fill_stage(ids_cpu, num_tokens, is_padding_cpu, num_tokens_padded)
            if num_tokens_padded:
                # Same-stream ordering prevents overwrite while a previous graph
                # reads the GPU buffer. The event protects the CPU source.
                self._pending = True
                try:
                    self._prefetch_buffer[:num_tokens_padded].copy_(
                        self._stage[:num_tokens_padded], non_blocking=True
                    )
                    self._copy_event.record(stream)
                except BaseException:
                    self._failed = True
                    _disk_bf16_quarantine.append(self)
                    raise
            self._prepared_tokens = num_tokens_padded
        finally:
            self._lock.release()

    def start_prefetch(self, ngram_ids: torch.Tensor) -> None:
        # Disk IDs and DMA are prepared by ModelState, outside captured forward.
        return None

    @eager_break_during_capture
    def _finalize_prefetch(
        self, prefetch_output: torch.Tensor, output: torch.Tensor
    ) -> None:
        if self._closed or self._failed or self._prepared_tokens != output.shape[0]:
            raise RuntimeError("Disk BF16 PLE consumer has no prepared dispatch")
        embeddings = self._reduce_etp_embeddings(prefetch_output[: output.shape[0]])
        output.copy_(embeddings.flatten(-2))

    def close(self) -> None:
        # Worker lifetime owns all buffers. Explicit teardown fences before free.
        if getattr(self, "_closed", True):
            return
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Cannot close a busy Disk BF16 PLE stage")
        try:
            if self._failed:
                raise RuntimeError("Cannot release a quarantined Disk BF16 PLE stage")
            self._wait_for_stage()
            # Also drain graph readers of the GPU destination.
            if self._owner_stream is not None:
                try:
                    self._owner_stream.synchronize()
                except BaseException:
                    self._failed = True
                    _disk_bf16_quarantine.append(self)
                    raise
            self._closed = True
        finally:
            self._lock.release()

    def __del__(self):
        try:
            self.close()
        except BaseException:
            _disk_bf16_quarantine.append(self)


# --- B70 expanded campaign 2026-11-16: parallel row gather (env-gated) -------
# B70_PLE_PARALLEL_GATHER=<threads>: _fill_stage's CPU work (zero the padded
# stage prefix, validate active IDs, copy owned rows) runs in one native call
# that copies rows from the SAME read-only file mmap with a small persistent
# thread pool, so page faults on the disk-backed table overlap. The staged
# bytes are identical (bitwise verified live on all ranks); nothing becomes
# resident beyond the OS page cache that the mmap already uses.
_b70_gather_threads = int(os.environ.get("B70_PLE_PARALLEL_GATHER", "0"))
if _b70_gather_threads:
    _b70_gather_state = {}
    _b70_original_fill_stage = Qwen4ExpPLEDiskBF16Embedding._fill_stage

    def _b70_gather_lib():
        lib = _b70_gather_state.get("lib")
        if lib is None:
            import vllm_xpu_kernels as _b70_vxk
            from pathlib import Path as _b70_Path

            lib = ctypes.CDLL(str(_b70_Path(_b70_vxk.__file__).with_name("libple_gather.so")))
            lib.ple_gather_init.argtypes = [ctypes.c_int]
            lib.ple_gather_init.restype = ctypes.c_int
            lib.ple_gather.argtypes = [
                ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
                ctypes.c_int64, ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p,
            ]
            lib.ple_gather.restype = ctypes.c_int
            if lib.ple_gather_init(_b70_gather_threads) != min(_b70_gather_threads, 32):
                raise RuntimeError("Disk BF16 PLE parallel gather pool failed to start")
            _b70_gather_state["lib"] = lib
            logger.info("Disk BF16 PLE parallel gather active: threads=%d", _b70_gather_threads)
        return lib

    def _b70_fill_stage(self, ids_cpu, num_tokens, is_padding_cpu, num_tokens_padded):
        ids = ids_cpu[:num_tokens].reshape(-1)
        pad = is_padding_cpu[:num_tokens] if is_padding_cpu is not None else None
        if (
            ids.dtype != torch.int64
            or not ids.is_contiguous()
            or not self._stage.is_contiguous()
            or (pad is not None and (pad.dtype != torch.bool or not pad.is_contiguous()))
        ):
            return _b70_original_fill_stage(self, ids_cpu, num_tokens, is_padding_cpu, num_tokens_padded)
        order = self.__dict__.get("_b70_gather_order")
        if order is None:
            order = self._b70_gather_order = torch.empty(
                self._capacity * self._heads, dtype=torch.int64
            )
        row_bytes = self.embedding_dim * self._stage.element_size()
        rc = _b70_gather_lib().ple_gather(
            self._table.data_ptr(), row_bytes, self.org_vocab_size, ids.data_ptr(),
            pad.data_ptr() if pad is not None else None, self._heads, ids.numel(),
            self.shard_indices.org_vocab_start_index,
            self.shard_indices.org_vocab_end_index,
            self._stage.data_ptr(), num_tokens_padded * self._heads * row_bytes,
            order.data_ptr(),
        )
        if rc == 1:
            raise ValueError("Disk BF16 PLE received an out-of-table row ID")
        if rc != 0:
            raise RuntimeError("Disk BF16 PLE parallel gather failed")

    Qwen4ExpPLEDiskBF16Embedding._fill_stage = _b70_fill_stage
