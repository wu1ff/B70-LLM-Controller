// SPDX-License-Identifier: Apache-2.0
// Derived from the preserved vLLM XPU speculative GDN kernels.
#include <sycl/sycl.hpp>
#include <torch/all.h>
#include <torch/library.h>
#include "csrc/utils.h"
namespace astra_gdn {
static constexpr int sub_group_size=32;
enum class ActMode { silu=0, swish=1 };
#include "prepared_conv.hpp"
#include "prepared_recurrent.hpp"
using T=sycl::ext::oneapi::bfloat16;
T* bf(at::Tensor t) {return reinterpret_cast<T*>(t.data_ptr());}
const int* ip(at::Tensor t) {return t.data_ptr<int>();}
std::vector<at::Tensor> prepare(at::Tensor x,at::Tensor ba,at::Tensor conv,at::Tensor w,
    at::Tensor A,at::Tensor dt,at::Tensor starts,at::Tensor map,at::Tensor ids,at::Tensor accepted,bool need_z) {
  TORCH_CHECK(x.scalar_type()==at::kBFloat16 && x.is_contiguous() && x.size(1)==4096,"packed BF16 projection required");
  TORCH_CHECK(ba.sizes()==at::IntArrayRef({x.size(0),24}) && ba.is_contiguous() && ba.scalar_type()==at::kBFloat16,"BA layout");
  TORCH_CHECK(w.sizes()==at::IntArrayRef({2560,4}) && w.is_contiguous(),"conv shape");
  TORCH_CHECK(conv.scalar_type()==at::kBFloat16 && conv.size(2)==2560 && conv.stride(1)==2560 && conv.stride(2)==1,"conv layout");
  TORCH_CHECK(A.scalar_type()==at::kFloat && dt.scalar_type()==at::kBFloat16,"gate dtype");
  TORCH_CHECK(ids.dim()==2 && ids.size(1)<=8 && starts.numel()==ids.size(0)+1 && accepted.numel()==ids.size(0),"metadata shape");
  int tokens=map.numel(), batch=ids.size(0), capacity=ids.size(1);
  TORCH_CHECK(conv.size(1)>=capacity+2,"conv capacity");
  auto opt=x.options();
  auto q=at::empty({tokens,4,128},opt.dtype(at::kFloat));auto k=at::empty_like(q);
  auto v=at::empty({tokens,12,128},opt);auto z=need_z?at::empty({x.size(0),12,128},opt):at::empty({0},opt);
  auto b=at::empty({tokens,12},opt.dtype(at::kFloat));auto a=at::empty_like(b);
  using Kernel=PreparedConv<T,4,true>;
  Kernel task(q.data_ptr<float>(),k.data_ptr<float>(),bf(v),need_z?bf(z):nullptr,b.data_ptr<float>(),a.data_ptr<float>(),
      bf(x),bf(ba),A.data_ptr<float>(),bf(dt),bf(w),nullptr,bf(conv),conv.stride(0),ip(starts),ip(map),ip(ids),ids.stride(0),ip(accepted),
      ActMode::silu,-1,batch,capacity,4,128,12,128,4096,2560);
  auto& queue=at::xpu::getCurrentXPUStream(x.device().index()).queue();
  queue.submit([&](sycl::handler& cgh) {cgh.parallel_for(Kernel::get_nd_range(batch,4096),task);});
  return {q,k,v,b,a,z};
}
template<bool Fused>
void launch(at::Tensor out,at::Tensor x,at::Tensor norm,double eps,at::Tensor q,at::Tensor k,at::Tensor v,
    at::Tensor b,at::Tensor a,at::Tensor A,at::Tensor dt,at::Tensor state,at::Tensor starts,at::Tensor map,at::Tensor ids,at::Tensor accepted) {
  using Kernel=PreparedRecurrent<T,float,4,Fused>;
  auto& queue=at::xpu::getCurrentXPUStream(x.device().index()).queue();
  queue.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<float,1> slm(sycl::range<1>(Fused?1024:1),cgh);
    Kernel task(bf(out),bf(x),bf(norm),float(eps),slm,
      q.data_ptr<float>(),k.data_ptr<float>(),bf(v),b.data_ptr<float>(),a.data_ptr<float>(),A.data_ptr<float>(),bf(dt),state.data_ptr<float>(),state.stride(0),
      ip(starts),ip(map),ip(ids),ids.stride(0),ip(accepted),ids.size(0),ids.size(1),4,128,12,128);
    cgh.parallel_for(Kernel::get_nd_range(ids.size(0),12,128),task);
  });
}
void recurrent(at::Tensor out,at::Tensor x,at::Tensor norm,double eps,at::Tensor q,at::Tensor k,at::Tensor v,
    at::Tensor b,at::Tensor a,at::Tensor A,at::Tensor dt,at::Tensor state,at::Tensor starts,at::Tensor map,at::Tensor ids,at::Tensor accepted,bool fused) {
  TORCH_CHECK(state.scalar_type()==at::kFloat && state.size(1)==12 && state.size(2)==128 && state.size(3)==128,"SSM layout");
  TORCH_CHECK(state.stride(1)==16384 && state.stride(2)==128 && state.stride(3)==1,"SSM strides");
  TORCH_CHECK(norm.numel()==128 && norm.is_contiguous() && norm.scalar_type()==at::kBFloat16,"norm layout");
  if(fused)launch<true>(out,x,norm,eps,q,k,v,b,a,A,dt,state,starts,map,ids,accepted);
  else launch<false>(out,x,norm,eps,q,k,v,b,a,A,dt,state,starts,map,ids,accepted);
}
}
TORCH_LIBRARY(astra_gdn,m){
 m.def("prepare(Tensor x, Tensor ba, Tensor(a!) conv, Tensor w, Tensor A, Tensor dt, Tensor starts, Tensor map, Tensor ids, Tensor accepted, bool need_z) -> Tensor[]");
 m.def("recurrent(Tensor(a!) out, Tensor x, Tensor norm, float eps, Tensor q, Tensor k, Tensor v, Tensor b, Tensor a, Tensor A, Tensor dt, Tensor(b!) state, Tensor starts, Tensor map, Tensor ids, Tensor accepted, bool fused) -> ()");
}
TORCH_LIBRARY_IMPL(astra_gdn,XPU,m){m.impl("prepare",&astra_gdn::prepare);m.impl("recurrent",&astra_gdn::recurrent);}
