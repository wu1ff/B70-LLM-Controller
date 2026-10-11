// Source-built correction for vllm-xpu-kernels 0.1.14.1.
#define gdn flashnext_index64_gdn
#define causal_conv1d_spec flashnext_index64_causal_conv1d_spec
#define causal_conv1d_non_spec flashnext_index64_causal_conv1d_non_spec
#define gated_delta_rule_spec flashnext_index64_gated_delta_rule_spec
#define gated_delta_rule_non_spec flashnext_index64_gated_delta_rule_non_spec
#define gdn_attention flashnext_index64_gdn_attention
#include "csrc/xpu/gdn_attn/gdn_attn_interface.cpp"
#include <torch/library.h>
TORCH_LIBRARY(_gdn_index64, xpu_ops) {
  xpu_ops.def(
      "causal_conv1d_spec(Tensor! z, Tensor "
      "projected_states_qkvz, Tensor projected_states_ba,"
      "int num_k_heads, int num_v_heads, int head_k_dim, int head_v_dim,"
      "Tensor! conv_state, Tensor conv_weights, Tensor? "
      "conv_bias, str activation,"
      "int num_prefills, int num_decodes, int num_spec_decodes,"
      "Tensor spec_query_start_loc, Tensor spec_token_indx, "
      "Tensor spec_state_indices_tensor, Tensor num_accepted_tokens,"
      "int num_actual_tokens, int tp_size, bool reorder_input) -> Tensor[]");
  xpu_ops.impl("causal_conv1d_spec", torch::kXPU, &causal_conv1d_spec);

  xpu_ops.def(
      "causal_conv1d_non_spec(Tensor! z, Tensor "
      "projected_states_qkvz, Tensor projected_states_ba,"
      "int num_k_heads, int num_v_heads, int head_k_dim, int head_v_dim,"
      "Tensor! conv_state, Tensor conv_weights, Tensor? "
      "conv_bias, str activation,"
      "int num_prefills, int num_decodes, int num_spec_decodes, Tensor? "
      "has_initial_state, Tensor non_spec_query_start_loc, Tensor? "
      "non_spec_token_indx, Tensor non_spec_state_indices_tensor,"
      "int num_actual_tokens, int tp_size, bool reorder_input) -> Tensor[]");
  xpu_ops.impl("causal_conv1d_non_spec", torch::kXPU, &causal_conv1d_non_spec);

  xpu_ops.def(
      "gated_delta_rule_spec(Tensor! core_attn_out,"
      "Tensor q, Tensor k, Tensor v, Tensor b, Tensor a,"
      "int num_v_heads, int head_v_dim,"
      "Tensor A_log, Tensor dt_bias, Tensor! ssm_state,"
      "int num_prefills, int num_decodes, int num_spec_decodes,"
      "Tensor spec_query_start_loc, Tensor spec_token_indx, "
      "Tensor spec_state_indices_tensor, Tensor num_accepted_tokens,"
      "int num_actual_tokens, int tp_size) -> ()");
  xpu_ops.impl("gated_delta_rule_spec", torch::kXPU, &gated_delta_rule_spec);

  xpu_ops.def(
      "gated_delta_rule_non_spec(Tensor! core_attn_out,"
      "Tensor q, Tensor k, Tensor v, Tensor b, Tensor a,"
      "int num_v_heads, int head_v_dim,"
      "Tensor A_log, Tensor dt_bias, Tensor! ssm_state,"
      "int num_prefills, int num_decodes, int num_spec_decodes, Tensor? "
      "has_initial_state, Tensor non_spec_query_start_loc, Tensor? "
      "non_spec_token_indx, Tensor non_spec_state_indices_tensor,"
      "int num_actual_tokens, int tp_size) -> ()");
  xpu_ops.impl(
      "gated_delta_rule_non_spec", torch::kXPU, &gated_delta_rule_non_spec);

  xpu_ops.def(
      "gdn_attention(Tensor! core_attn_out, Tensor! z, Tensor "
      "projected_states_qkvz, Tensor projected_states_ba,"
      "int num_k_heads, int num_v_heads, int head_k_dim, int head_v_dim,"
      "Tensor! conv_state, Tensor! ssm_state, Tensor conv_weights, Tensor? "
      "conv_bias, str activation, Tensor A_log, Tensor dt_bias,"
      "int num_prefills, int num_decodes, int num_spec_decodes, Tensor? "
      "has_initial_state, Tensor? "
      "non_spec_query_start_loc,  Tensor? non_spec_token_indx,"
      "Tensor? non_spec_state_indices_tensor, Tensor? spec_query_start_loc, "
      "Tensor? spec_token_indx, Tensor? spec_state_indices_tensor, Tensor? "
      "num_accepted_tokens, int num_actual_tokens, int "
      "tp_size, bool reorder_input) -> ()");
  xpu_ops.impl("gdn_attention", torch::kXPU, &gdn_attention);
}
