template <typename T, int Width, bool ReorderInput>
struct PreparedConv {
 public:
  static constexpr int sub_group_size = 32;
  static constexpr int group_size = 64;
  static constexpr int elems_per_item = 4;
  static constexpr int elems_per_group = group_size * elems_per_item;

  PreparedConv(
      float* q_out,
      float* k_out,
      T* v_out,
      T* z_out,
      float* b_out,
      float* a_out,
      const T* mixed_qkvz,
      const T* mixed_ba, const float* A_log, const T* dt_bias,
      const T* conv_weights,
      const T* conv_bias,
      T* conv_states,
      const int conv_states_stride_0,
      const int* query_start_loc,
      const int* token_indx,
      const int* cache_indices,
      const int cache_indices_stride_0,
      const int* num_accepted_tokens,
      const ActMode& act_mode,
      const int& pad_slot_id,
      const int& num_spec_decodes,
      const int& num_spec_tokens,
      const int& num_k_heads,
      const int& head_k_dim,
      const int& num_v_heads,
      const int& head_v_dim,
      const int& qkvz_elems,
      const int& conv_elems)
      : q_out(q_out),
        k_out(k_out),
        v_out(v_out),
        z_out(z_out),
        b_out(b_out),
        a_out(a_out),
        mixed_qkvz(mixed_qkvz),
        mixed_ba(mixed_ba), A_log(A_log), dt_bias(dt_bias),
        conv_weights(conv_weights),
        conv_bias(conv_bias),
        conv_states(conv_states),
        conv_states_stride_0(conv_states_stride_0),
        query_start_loc(query_start_loc),
        token_indx(token_indx),
        cache_indices(cache_indices),
        cache_indices_stride_0(cache_indices_stride_0),
        num_accepted_tokens(num_accepted_tokens),
        act_mode(act_mode),
        pad_slot_id(pad_slot_id),
        num_spec_decodes(num_spec_decodes),
        num_spec_tokens(num_spec_tokens),
        num_k_heads(num_k_heads),
        head_k_dim(head_k_dim),
        num_v_heads(num_v_heads),
        head_v_dim(head_v_dim),
        qkvz_elems(qkvz_elems),
        conv_elems(conv_elems) {}

  static inline sycl::nd_range<2>
  get_nd_range(const int num_spec_decodes, const int qkvz_elems) {
    const int groups_per_token =
        (qkvz_elems + elems_per_group - 1) / elems_per_group;
    sycl::range<2> local(1, group_size);
    sycl::range<2> global(num_spec_decodes, groups_per_token);
    return sycl::nd_range<2>(global * local, local);
  }

  static inline void act_swish(float& x, float beta = 1.0f) {
    x = x / (1.0f + sycl::exp(-x * beta));
  }
  static inline void act_silu(float& x) { act_swish(x, 1.0f); }

  [[sycl::reqd_sub_group_size(sub_group_size)]] void
  operator()(sycl::nd_item<2> item) const {
    const int batch_id = item.get_group(0);
    const int local_group_id = item.get_group(1);
    const int local_id = item.get_local_linear_id();
    const int qkvz_elems_id =
        local_group_id * elems_per_group + local_id * elems_per_item;

    if (qkvz_elems_id >= qkvz_elems) {
      return;
    }

    const int q_dim = head_k_dim;
    const int k_dim = head_k_dim;
    const int v_dim = head_v_dim * num_v_heads / num_k_heads;
    const int z_dim = head_v_dim * num_v_heads / num_k_heads;
    const int qkvz_dim = q_dim + k_dim + v_dim + z_dim;

    int k_heads_id = qkvz_elems_id / qkvz_dim;
    int qkvz_dim_id = qkvz_elems_id % qkvz_dim;

    // Determine role (q/k/v/z) and the mixed_qkvz offset for this feature.
    bool is_q = false, is_k = false, is_v = false, is_z = false;
    int mixed_qkvz_id = qkvz_elems_id;
    int reordered_elems_id = 0;
    if (qkvz_dim_id < q_dim) {
      is_q = true;
      if constexpr (ReorderInput) {
        mixed_qkvz_id = k_heads_id * k_dim + qkvz_dim_id;
      }
      reordered_elems_id = k_heads_id * q_dim + qkvz_dim_id;
    } else if (qkvz_dim_id < q_dim + k_dim) {
      is_k = true;
      if constexpr (ReorderInput) {
        mixed_qkvz_id =
            num_k_heads * head_k_dim + k_heads_id * k_dim + qkvz_dim_id - q_dim;
      }
      reordered_elems_id =
          num_k_heads * q_dim + k_heads_id * k_dim + qkvz_dim_id - q_dim;
    } else if (qkvz_dim_id < q_dim + k_dim + v_dim) {
      is_v = true;
      if constexpr (ReorderInput) {
        mixed_qkvz_id = 2 * num_k_heads * head_k_dim + k_heads_id * v_dim +
                        qkvz_dim_id - (q_dim + k_dim);
      }
      reordered_elems_id = num_k_heads * (q_dim + k_dim) + k_heads_id * v_dim +
                           qkvz_dim_id - (q_dim + k_dim);
    } else {
      is_z = true;
      if constexpr (ReorderInput) {
        mixed_qkvz_id = 2 * num_k_heads * head_k_dim +
                        num_v_heads * head_v_dim + k_heads_id * z_dim +
                        qkvz_dim_id - (q_dim + k_dim + v_dim);
      }
    }

    const int token_start = query_start_loc[batch_id];
    const int token_end = query_start_loc[batch_id + 1];
    const int actual_num_tokens = token_end - token_start;

    // -- b / a reorder (mixed_ba GLOBAL → b_out/a_out LOCAL) ----------------
    for (int token_id_local = token_start; token_id_local < token_end;
         ++token_id_local) {
      const int global_t = token_indx[token_id_local];
      if constexpr (ReorderInput) {
        if (qkvz_elems_id < num_v_heads) {
          int step_local = token_id_local * num_v_heads;
          int step_global = global_t * num_v_heads;
#pragma unroll
          for (int e = 0; e < elems_per_item; ++e) {
            b_out[step_local + qkvz_elems_id + e] =
              1.f / (1.f + sycl::exp(-float(mixed_ba[step_global * 2 + qkvz_elems_id + e])));
            const int h = qkvz_elems_id + e;
          float av = float(mixed_ba[step_global * 2 + num_v_heads + qkvz_dim_id + e]) + float(dt_bias[h]);
          float sp = av < 20.f ? sycl::log(1.f + sycl::exp(av)) : av;
          a_out[step_local + h] = sycl::exp(-sycl::exp(A_log[h]) * sp);
          }
        }
      } else {
        if (qkvz_dim_id < (num_v_heads / num_k_heads)) {
          int step_local = token_id_local * num_v_heads +
                           k_heads_id * num_v_heads / num_k_heads;
          int step_global =
              global_t * num_v_heads + k_heads_id * num_v_heads / num_k_heads;
          const int ba_elems_per_item =
              sycl::min(elems_per_item, num_v_heads / num_k_heads);
#pragma unroll
          for (int e = 0; e < ba_elems_per_item; ++e) {
            b_out[step_local + qkvz_dim_id + e] =
                mixed_ba[step_global * 2 + qkvz_dim_id + e];
            a_out[step_local + qkvz_dim_id + e] = mixed_ba
                [step_global * 2 + num_v_heads / num_k_heads + qkvz_dim_id + e];
          }
        }
      }
    }

    // -- z reorder: simple copy through token_indx, no convolution. ---------
    if (is_z) {
      if (z_out == nullptr) return;
      int z_elems_id =
          k_heads_id * z_dim + qkvz_dim_id - (q_dim + k_dim + v_dim);
      for (int token_id_local = token_start; token_id_local < token_end;
           ++token_id_local) {
        const int global_t = token_indx[token_id_local];
#pragma unroll
        for (int e = 0; e < elems_per_item; ++e) {
          z_out[global_t * num_k_heads * z_dim + z_elems_id + e] =
              mixed_qkvz[global_t * qkvz_elems + mixed_qkvz_id + e];
        }
      }
      return;
    }

    if (!(is_q || is_k || is_v)) {
      return;
    }

    // -- conv1d on q/k/v lanes ----------------------------------------------
    // Single cache line (column 0); accepted window starts at row init_row.
    int init_row = num_accepted_tokens[batch_id] - 1;
    if (init_row < 0) init_row = 0;
    // Match the effective state length used by the CUDA/Triton path.  The
    // leading stride may span interleaved states from multiple layers, so it
    // cannot be used to recover this dimension.
    const int state_len = Width - 1 + (num_spec_tokens - 1);
    const int state_id = cache_indices[batch_id * cache_indices_stride_0 + 0];
    const bool has_conv_state = (state_id != pad_slot_id);
    T* state_line_ptr = has_conv_state
                            ? conv_states + static_cast<int64_t>(state_id) * conv_states_stride_0
                            : nullptr;

    // Load weights
    T local_weights[Width * elems_per_item];
#pragma unroll
    for (int i = 0; i < Width; ++i) {
#pragma unroll
      for (int e = 0; e < elems_per_item; ++e) {
        local_weights[Width * e + i] =
            conv_weights[(reordered_elems_id + e) * Width + i];
      }
    }

    // Prime the window from rows [init_row, init_row + Width - 1); the trailing
    // slot is filled per-iter. Pad slot => zeroed prior.
    T local_input[Width * elems_per_item];
#pragma unroll
    for (int i = 0; i < Width * elems_per_item; ++i) {
      local_input[i] = static_cast<T>(0);
    }
    if (Width > 1 && has_conv_state) {
#pragma unroll
      for (int i = 0; i < Width - 1; ++i) {
#pragma unroll
        for (int e = 0; e < elems_per_item; ++e) {
          local_input[Width * e + i] = state_line_ptr
              [(init_row + i) * conv_elems + reordered_elems_id + e];
        }
      }
    }

    for (int token_id_local = token_start, t_local = 0;
         token_id_local < token_end;
         ++token_id_local, ++t_local) {
      const int global_t = token_indx[token_id_local];

      // Shift window left by 1 (for t_local == 0 the trailing slot is fresh).
      if (t_local > 0) {
#pragma unroll
        for (int i = 0; i < Width - 1; ++i) {
#pragma unroll
          for (int e = 0; e < elems_per_item; ++e) {
            local_input[Width * e + i] = local_input[Width * e + i + 1];
          }
        }
      }
      // Load new input at the trailing slot.
#pragma unroll
      for (int e = 0; e < elems_per_item; ++e) {
        local_input[Width * e + Width - 1] =
            mixed_qkvz[global_t * qkvz_elems + mixed_qkvz_id + e];
      }

      float res[elems_per_item];
#pragma unroll
      for (int e = 0; e < elems_per_item; ++e) {
        res[e] = 0.0f;
      }
#pragma unroll
      for (int i = 0; i < Width; ++i) {
#pragma unroll
        for (int e = 0; e < elems_per_item; ++e) {
          res[e] += static_cast<float>(local_input[Width * e + i]) *
                    static_cast<float>(local_weights[Width * e + i]);
        }
      }
      if (conv_bias != nullptr) {
#pragma unroll
        for (int e = 0; e < elems_per_item; ++e) {
          res[e] += conv_bias[reordered_elems_id + e];
        }
      }
      if (act_mode == ActMode::silu) {
#pragma unroll
        for (int e = 0; e < elems_per_item; ++e) {
          act_silu(res[e]);
        }
      } else if (act_mode == ActMode::swish) {
#pragma unroll
        for (int e = 0; e < elems_per_item; ++e) {
          act_swish(res[e]);
        }
      }

      // Q/K BF16 convolution rounding is retained before normalization.
      // Each subgroup covers exactly one 128-wide head, with four lanes/thread.
      if (is_q || is_k) {
        float ss = 0.f;
#pragma unroll
        for (int e=0;e<4;++e) { res[e] = float(T(res[e])); ss += res[e]*res[e]; }
        ss = sycl::reduce_over_group(item.get_sub_group(), ss, sycl::plus<>()) + 1e-6f;
#pragma unroll
        for (int e=0;e<4;++e) { res[e] /= sycl::sqrt(ss); if (is_q) res[e] *= 1.f/sycl::sqrt(128.f); }
      }
      // Write to LOCAL q/k/v at token_id_local.
      if (is_q) {
#pragma unroll
        for (int e = 0; e < elems_per_item; ++e) {
          q_out
              [token_id_local * num_k_heads * q_dim + k_heads_id * q_dim +
               qkvz_dim_id + e] = res[e];
        }
      } else if (is_k) {
#pragma unroll
        for (int e = 0; e < elems_per_item; ++e) {
          k_out
              [token_id_local * num_k_heads * k_dim + k_heads_id * k_dim +
               qkvz_dim_id - q_dim + e] = res[e];
        }
      } else {  // is_v
#pragma unroll
        for (int e = 0; e < elems_per_item; ++e) {
          v_out
              [token_id_local * num_k_heads * v_dim + k_heads_id * v_dim +
               qkvz_dim_id - (q_dim + k_dim) + e] = res[e];
        }
      }
    }

    // Roll the logical line back to column 0: rows [0, hist_rows) = shifted
    // history from row init_row+1, followed by the actual compact draft
    // inputs. Rows beyond logical_state_len are unused capacity for this
    // ragged forward and must remain unchanged.
    // In-place left shift is safe since src row (init_row+1+j) >= dst row j.
    if (Width > 1 && has_conv_state) {
      const int logical_state_len =
          state_len - (num_spec_tokens - actual_num_tokens);
      const int hist_rows = logical_state_len - actual_num_tokens;
      for (int j = 0; j < logical_state_len; ++j) {
        if (j < hist_rows) {
          const int src_row = init_row + 1 + j;
#pragma unroll
          for (int e = 0; e < elems_per_item; ++e) {
            state_line_ptr[j * conv_elems + reordered_elems_id + e] =
                state_line_ptr[src_row * conv_elems + reordered_elems_id + e];
          }
        } else {
          const int draft_t = j - hist_rows;
          const int token_id_local = token_start + draft_t;
          const int global_t = token_indx[token_id_local];
#pragma unroll
          for (int e = 0; e < elems_per_item; ++e) {
            state_line_ptr[j * conv_elems + reordered_elems_id + e] =
                mixed_qkvz[global_t * qkvz_elems + mixed_qkvz_id + e];
          }
        }
      }
    }
  }

 private:
  float* q_out;
  float* k_out;
  T* v_out;
  T* z_out;
  float* b_out;
  float* a_out;
  const T* mixed_qkvz;
  const T* mixed_ba; const float* A_log; const T* dt_bias;
  const T* conv_weights;
  const T* conv_bias;
  T* conv_states;
  const int conv_states_stride_0;
  const int* query_start_loc;
  const int* token_indx;
  const int* cache_indices;
  const int cache_indices_stride_0;
  const int* num_accepted_tokens;
  const ActMode act_mode;
  const int pad_slot_id;
  const int num_spec_decodes;
  const int num_spec_tokens;
  const int num_k_heads;
  const int head_k_dim;
  const int num_v_heads;
  const int head_v_dim;
  const int qkvz_elems;
  const int conv_elems;
};
