

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
