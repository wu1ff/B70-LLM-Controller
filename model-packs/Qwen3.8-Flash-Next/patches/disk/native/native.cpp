// SPDX-License-Identifier: Apache-2.0
// Narrow BF16 qpacked8/head256/page-tile64 dispatcher. Native producer and
// reduction arithmetic are inherited; only reduction output-coordinate tiling
// differs. Unsupported serving shapes continue through the original library.
#include <sycl/sycl.hpp>
#include <torch/all.h>
#include <torch/library.h>
#include <c10/core/DeviceGuard.h>
#include <c10/xpu/XPUStream.h>
#include "csrc/xpu/attn/xe_2/paged_decode.hpp"

template<int Tiles>
void dispatch_b70(sycl::queue& queue,const paged_decode_args_t& args) {
 using P=decode_policy_q8_h256_p64;
 using S=Stride<int,_1,int,int>;
 PagedDecodeConfig<typename P::ShapeQK,typename P::ShapePV,
  typename P::ShapeOut,typename P::SubgroupLayoutQK,void,1,true,false,false,
  bfloat16_t,bfloat16_t,bfloat16_t,bfloat16_t,void,S,S,
  Stride<_1,int,int,int>,S,S,void,void,void,void,Tiles>::kernel_dispatch(queue,args);
}
void b70_forward(const at::Tensor& q,const at::Tensor& k,const at::Tensor& v,
                 at::Tensor& out,const at::Tensor& cu,const at::Tensor& used,
                 const at::Tensor& bt,const at::Tensor& plan,const at::Tensor& work,
                 double scale,int64_t max_len,int64_t tiles) {
 TORCH_CHECK(q.is_xpu() && q.scalar_type()==at::kBFloat16 && k.scalar_type()==at::kBFloat16 && v.scalar_type()==at::kBFloat16);
 TORCH_CHECK(q.dim()==3 && (q.size(0)==2 || q.size(0)==4 || q.size(0)==6 || q.size(0)==8) && q.size(1)==6 && q.size(2)==256 && q.stride(2)==1);
 TORCH_CHECK(k.dim()==4 && k.size(1)==832 && k.size(2)==1 && k.size(3)==256 && k.sizes()==v.sizes());
 TORCH_CHECK(out.sizes()==q.sizes() && out.is_contiguous() && out.scalar_type()==q.scalar_type());
 TORCH_CHECK(cu.scalar_type()==at::kInt && cu.numel()==q.size(0)+1 && cu.is_contiguous());
 TORCH_CHECK(used.scalar_type()==at::kInt && used.numel()==q.size(0) && used.is_contiguous());
 TORCH_CHECK(bt.scalar_type()==at::kInt && bt.dim()==2 && bt.size(0)==q.size(0) && bt.is_contiguous());
 TORCH_CHECK(plan.scalar_type()==at::kInt && plan.numel()==q.size(0) && plan.is_contiguous());
 TORCH_CHECK(work.scalar_type()==at::kInt && work.sizes()==at::IntArrayRef({q.size(0)*64,4}) && work.is_contiguous());
 TORCH_CHECK(q.device()==k.device() && q.device()==v.device() && q.device()==out.device() && q.device()==cu.device() && q.device()==used.device() && q.device()==bt.device() && q.device()==plan.device() && q.device()==work.device());
 check_paged_kv_cache_strides(k,v);
 c10::DeviceGuard guard(q.device());
 auto temp=at::empty({q.size(0),64,6,256},q.options());
 auto sums=at::empty({q.size(0),6,64},q.options().dtype(at::kFloat));
 auto maxima=at::empty_like(sums);
 paged_decode_args_t a{};
 a.query=q.data_ptr();a.key=k.data_ptr();a.value=v.data_ptr();a.out=out.data_ptr();
 a.tem_out=temp.data_ptr();a.exp_sums=sums.data_ptr();a.max_logits=maxima.data_ptr();
 a.block_table=bt.data_ptr();a.cu_seqlens_q=cu.data_ptr();a.cu_seqlens_k=used.data_ptr();
 a.max_queries=1;a.max_keys=max_len;a.total_seqlen_q=q.size(0);
 a.total_seqlen_k=get_paged_kv_cache_effective_total_seqlen(k);
 a.sm_scale=scale;a.batch_size=q.size(0);a.num_heads_q=6;a.num_heads_k=1;
 a.head_size=256;a.v_head_size=256;a.max_blocks_per_seq=bt.size(1);a.block_size=832;
 a.is_varlen=true;a.is_paged=true;a.is_causal=true;a.is_local=false;a.is_sink=false;
 a.window_size_left=-1;a.window_size_right=-1;a.num_kv_splits=64;
 a.splits_per_seq=plan.data_ptr<int>();a.work_list=work.data_ptr<int>();a.total_wgs=q.size(0)*64;
 a.k_stride_page=k.stride(0);a.k_stride_seq=k.stride(1);a.k_stride_heads=k.stride(2);
 a.v_stride_page=v.stride(0);a.v_stride_seq=v.stride(1);a.v_stride_heads=v.stride(2);
 a.q_stride_seq=q.stride(0);a.q_stride_heads=q.stride(1);a.q_stride_batch=0;
 a.page_stride_elements=get_paged_kv_cache_page_stride_elements(k);
 auto& queue=c10::xpu::getCurrentXPUStream().queue();
 if(tiles==1)dispatch_b70<1>(queue,a);
 else if(tiles==4)dispatch_b70<4>(queue,a);
 else TORCH_CHECK(false,"Only diagnostic1 or selected4 output tiles are supported");
}
TORCH_LIBRARY(_b70_attn_reduce,m) {
 m.def("forward(Tensor q, Tensor k, Tensor v, Tensor(a!) out, Tensor cu, Tensor used, Tensor bt, Tensor plan, Tensor work, float scale, int max_len, int tiles=4) -> ()");
 m.impl("forward",torch::kXPU,&b70_forward);
}
