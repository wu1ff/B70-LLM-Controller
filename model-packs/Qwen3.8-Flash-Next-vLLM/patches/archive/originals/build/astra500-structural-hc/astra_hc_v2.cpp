// SPDX-License-Identifier: Apache-2.0
#include <torch/all.h>
#include "csrc/utils.h"
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include "gemm_xe2_policy.hpp"
#include "fused_hc_v2.hpp"
namespace AstraHC2 {
template<int TM> class FusedHC;
template<int TM> void launch(at::Tensor a, at::Tensor w, at::Tensor x, at::Tensor out) {
  TORCH_CHECK(a.scalar_type()==at::kBFloat16 && w.scalar_type()==at::kBFloat16 &&
      x.scalar_type()==at::kBFloat16 && out.scalar_type()==at::kBFloat16,"BF16 required");
  TORCH_CHECK(a.is_contiguous() && w.is_contiguous() && x.is_contiguous() && out.is_contiguous(),"contiguous required");
  const int m=a.size(0);
  TORCH_CHECK(a.dim()==2 && a.size(1)==320 && w.sizes()==at::IntArrayRef({10240,320}) &&
      x.sizes()==at::IntArrayRef({m,10240}) && out.sizes()==at::IntArrayRef({m,2560}),"shape mismatch");
  using T=cutlass::bfloat16_t;
  struct P: MoE::xe_gemm_policy_base {
    using WGTile=Shape<Int<TM>,_64,_32>;
    using SGLayout=Layout<Shape<Int<TM/8>,_4,_1>,Stride<_4,_1,_0>>;
  };
  using MMA=typename MoE::TiledMMAHelper<MMA_Atom<XE_DPAS_TT<8,float,T>>,Layout<typename P::WGTile>,typename P::SGLayout>::TiledMMA;
  auto mma=MMA{};
  auto A=MoE::make_moe_tensor<T,'R'>(reinterpret_cast<T*>(a.data_ptr()),m,320);
  auto B=MoE::make_moe_tensor<T,'R'>(reinterpret_cast<T*>(w.data_ptr()),10240,320);
  // C supplies only coordinates; the epilogue writes the reduced [M,2560] output.
  auto C=MoE::make_moe_tensor<T,'R'>(reinterpret_cast<T*>(out.data_ptr()),m,10240);
  auto* xn=reinterpret_cast<T*>(x.data_ptr());auto* y=reinterpret_cast<T*>(out.data_ptr());
  auto& q=at::xpu::getCurrentXPUStream(a.device().index()).queue();
  namespace sx=sycl::ext::oneapi::experimental; namespace ix=sycl::ext::intel::experimental;
  sx::properties props{sx::sub_group_size<16>,ix::grf_size<256>};
  q.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<float,1> slm(sycl::range<1>(TM*64),cgh);
    cgh.parallel_for<FusedHC<TM>>(sycl::nd_range<3>(sycl::range<3>((m+TM-1)/TM,160,TM*8),sycl::range<3>(1,1,TM*8)),props,
      [=](sycl::nd_item<3> it) {
        auto coord=make_coord(int(it.get_group(0)),int(it.get_group(1)),_,0);
        auto localC=C;
        fused_hc_up<void,void,void>(A,B,static_cast<T*>(nullptr),static_cast<T*>(nullptr),localC,coord,mma,xn,y,
          slm.template get_multi_ptr<sycl::access::decorated::no>().get(),m);
      });
  });
}
void up(at::Tensor a, at::Tensor w, at::Tensor x, at::Tensor out) {
  if(a.size(0)<=8)launch<8>(a,w,x,out);else launch<32>(a,w,x,out);
}
}
TORCH_LIBRARY(astra_hc_v2,m){m.def("up(Tensor a, Tensor w, Tensor x, Tensor(a!) out) -> ()");}
TORCH_LIBRARY_IMPL(astra_hc_v2,XPU,m){m.impl("up",&AstraHC2::up);}
