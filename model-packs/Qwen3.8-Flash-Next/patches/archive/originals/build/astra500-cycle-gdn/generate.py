from pathlib import Path
r=Path(__file__).resolve().parents[2]
out=r/'build/astra500-cycle-gdn';out.mkdir(exist_ok=True)
s=(r/'build/native/source/csrc/xpu/gdn_attn/causal_conv1d.hpp').read_text()
s=s[s.index('template <typename T, int Width, bool ReorderInput>\nstruct causal_conv1d_spec_kernel'):s.index('\ntemplate <typename T, int Width, bool ReorderInput>\nvoid kernel_launcher')]
# Only the single class, independent of the generic dispatch functions.
s=s[:s.index('\n};')+3]
s=s.replace('causal_conv1d_spec_kernel','PreparedConv')
for p in ['q_out','k_out','b_out','a_out']:s=s.replace('T* '+p,'float* '+p)
s=s.replace('const T* mixed_ba,','const T* mixed_ba, const float* A_log, const T* dt_bias,')
s=s.replace('mixed_ba(mixed_ba),','mixed_ba(mixed_ba), A_log(A_log), dt_bias(dt_bias),')
s=s.replace('const T* mixed_ba;','const T* mixed_ba; const float* A_log; const T* dt_bias;')
# ReorderInput is required by the wrapper; transform the two assignments.
a='''b_out[step_local + qkvz_elems_id + e] =
                mixed_ba[step_global * 2 + qkvz_elems_id + e];'''
b='''b_out[step_local + qkvz_elems_id + e] =
              1.f / (1.f + sycl::exp(-float(mixed_ba[step_global * 2 + qkvz_elems_id + e])));'''
assert a in s;s=s.replace(a,b)
a='''a_out[step_local + qkvz_elems_id + e] =
                mixed_ba[step_global * 2 + num_v_heads + qkvz_dim_id + e];'''
b='''const int h = qkvz_elems_id + e;
          float av = float(mixed_ba[step_global * 2 + num_v_heads + qkvz_dim_id + e]) + float(dt_bias[h]);
          float sp = av < 20.f ? sycl::log(1.f + sycl::exp(av)) : av;
          a_out[step_local + h] = sycl::exp(-sycl::exp(A_log[h]) * sp);'''
assert a in s;s=s.replace(a,b)
a='''      // Write to LOCAL q/k/v at token_id_local.'''
b='''      // Q/K BF16 convolution rounding is retained before normalization.
      // Each subgroup covers exactly one 128-wide head, with four lanes/thread.
      if (is_q || is_k) {
        float ss = 0.f;
#pragma unroll
        for (int e=0;e<4;++e) { res[e] = float(T(res[e])); ss += res[e]*res[e]; }
        ss = sycl::reduce_over_group(item.get_sub_group(), ss, sycl::plus<>()) + 1e-6f;
#pragma unroll
        for (int e=0;e<4;++e) { res[e] /= sycl::sqrt(ss); if (is_q) res[e] *= 1.f/sycl::sqrt(128.f); }
      }
      // Write to LOCAL q/k/v at token_id_local.'''
assert a in s;s=s.replace(a,b)
s=s.replace('group_size = 256','group_size = 64').replace('if (is_z) {','if (is_z) {\n      if (z_out == nullptr) return;')
(out/'prepared_conv.hpp').write_text(s+'\n')
s=(r/'build/native/source/csrc/xpu/gdn_attn/gated_delta_rule.hpp').read_text()
s=s[s.index('template <typename T, typename StateT, int k_bucket_size>\nstruct gated_delta_rule_spec_kernel'):]
s=s[:s.index('\n};')+3]
s=s.replace('int k_bucket_size>','int k_bucket_size, bool FusedNorm>')
s=s.replace('gated_delta_rule_spec_kernel','PreparedRecurrent')
s=s.replace('static constexpr int group_size = 256;','static constexpr int group_size = FusedNorm ? 1024 : 256;')
for p in ['q','k','b','a']:s=s.replace('const T* '+p+',','const float* '+p+',').replace('const T* '+p+';','const float* '+p+';')
s=s.replace('T* core_attn_out,','T* core_attn_out, const T* mixed, const T* norm, float norm_eps, float* scratch,')
s=s.replace(': core_attn_out(core_attn_out),',': core_attn_out(core_attn_out), mixed(mixed), norm(norm), norm_eps(norm_eps), scratch(scratch),')
s=s.replace('T* core_attn_out;','T* core_attn_out; const T* mixed; const T* norm; float norm_eps; float* scratch;')
s=s.replace('float beta = act_sigmoid(b_local);','float beta = b_local;')
s=s.replace('float a_local = a[t * num_v_heads + num_v_heads_id] + dt_bias_local;\n      float g = sycl::exp(A_log_local * act_softplus(a_local));','float g = a[t * num_v_heads + num_v_heads_id];')
a=s.index('      q_sum = sycl::reduce_over_group');b=s.index('      float kv_mem',a)
s=s[:a]+s[b:]
a='''          core_attn_out
              [global_t * num_v_heads * head_v_dim +
               num_v_heads_id * head_v_dim + head_v_dim_id + i] = res[i];'''
b='''          if constexpr (FusedNorm) scratch[t_local * 128 + head_v_dim_id + i] = float(T(res[i]));
          else core_attn_out[global_t * num_v_heads * head_v_dim + num_v_heads_id * head_v_dim + head_v_dim_id + i] = res[i];'''
assert a in s;s=s.replace(a,b)
a='''    }
  }

 private:'''
b='''    }
    if constexpr (FusedNorm) {
      item.barrier(sycl::access::fence_space::local_space);
      if (sg_id < token_end-token_start) {
        float vals[4]; float sum=0.f;
#pragma unroll
        for (int e=0;e<4;++e) { vals[e]=scratch[sg_id*128+sg_local_id*4+e]; sum+=vals[e]*vals[e]; }
        sum=sycl::reduce_over_group(sg,sum,sycl::plus<>());
        float inv=sycl::rsqrt(sum/128.f+norm_eps);
        int glob=token_indx[token_start+sg_id];
#pragma unroll
        for (int e=0;e<4;++e) {
          int d=sg_local_id*4+e;
          float z=float(mixed[glob*4096+2560+num_v_heads_id*128+d]);
          float gate=z/(1.f+sycl::exp(-z));
          core_attn_out[glob*1536+num_v_heads_id*128+d]=vals[e]*inv*float(norm[d])*gate;
        }
      }
    }
  }

 private:'''
assert a in s;s=s.replace(a,b)
s=s.replace('float* scratch','sycl::local_accessor<float,1> scratch')
(out/'prepared_recurrent.hpp').write_text(s+'\n')
