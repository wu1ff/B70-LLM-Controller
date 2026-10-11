#define moe flashnext_moe_guard
#define remap_hidden_states flashnext_remap_hidden_states
#include "csrc/moe/remap_hidden_states.cpp"
#include <torch/library.h>
TORCH_LIBRARY(_moe_remap_guard, m) {
 m.def("remap_hidden_states(Tensor hidden_states, Tensor? hidden_states_scales, Tensor(a!) remapped_hidden_states, Tensor(b!)? remapped_hidden_states_scales, Tensor? expert_map, Tensor(c!) rows_per_expert, Tensor(d!) unpermuted_row_to_permuted_row, Tensor topk_ids, int total_experts_num, int local_experts_num) -> ()");
 m.impl("remap_hidden_states", torch::kXPU, &remap_hidden_states);
}
