// SPDX-License-Identifier: Apache-2.0
#include <torch/all.h>
#include "csrc/utils.h"
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include "gemm_xe2_policy.hpp"
#include "down_tile.hpp"
namespace AstraDownV2 {
using T=cutlass::bfloat16_t;
struct Policy:MoE::xe_gemm_policy_base {using WGTile=Shape<_8,_16,_64>;using SGLayout=Layout<Shape<_1,_1,_1>,Stride<_1,_1,_0>>;};
class Partials;class Finish;
std::vector<at::Tensor> down(at::Tensor x,at::Tensor w,int64_t valid,int64_t split,bool activated) {
 TORCH_CHECK(x.scalar_type()==at::kBFloat16 && w.scalar_type()==at::kBFloat16 && x.is_contiguous() && w.is_contiguous(),"contiguous BF16");
 int m=x.size(0),n=w.size(0);TORCH_CHECK(m>=1 && m<=8 && x.size(1)==10240 && w.size(1)==10240 && (n==336 || n==320),"HC shape");
 TORCH_CHECK(valid==(n==336?324:320) && split>=128 && split%64==0 && 10240%split==0,"HC partition");
 int tiles=10240/split;
 auto partial=at::empty({tiles,m,n},x.options().dtype(at::kFloat));
 auto raw=at::empty({m,n},x.options());auto lora=at::empty({m,320},x.options());
 auto* xp=reinterpret_cast<T*>(x.data_ptr());auto* wp=reinterpret_cast<T*>(w.data_ptr());
 auto* pp=partial.data_ptr<float>();auto* yp=reinterpret_cast<T*>(raw.data_ptr());auto* lp=reinterpret_cast<T*>(lora.data_ptr());
 using MMA=typename MoE::TiledMMAHelper<MMA_Atom<XE_DPAS_TT<8,float,T>>,Layout<typename Policy::WGTile>,typename Policy::SGLayout>::TiledMMA;
 auto mma=MMA{};auto& queue=at::xpu::getCurrentXPUStream(x.device().index()).queue();
 namespace sx=sycl::ext::oneapi::experimental;namespace ix=sycl::ext::intel::experimental;
 sx::properties props{sx::sub_group_size<16>,ix::grf_size<256>};
 queue.parallel_for<Partials>(sycl::nd_range<3>(sycl::range<3>(tiles,n/16,16),sycl::range<3>(1,1,16)),props,
  [=](sycl::nd_item<3> it){
    int tile=it.get_group(0),coltile=it.get_group(1),lane=it.get_local_id(2);
    auto A=make_tensor(make_gmem_ptr(xp+tile*split),make_shape(m,int(split)),make_stride(10240,Int<1>{}));
    auto B=make_tensor(make_gmem_ptr(wp+tile*split),make_shape(n,int(split)),make_stride(10240,Int<1>{}));
    auto C=MoE::make_moe_tensor<T,'R'>(yp,m,n);
    auto c=down_tile<void,void,void>(A,B,static_cast<T*>(nullptr),static_cast<T*>(nullptr),C,make_coord(0,coltile,_,0),mma);
    for(int row=0;row<m;++row)pp[(tile*m+row)*n+coltile*16+lane]=(coltile*16+lane<valid)?c(row):0.f;
  });
 queue.parallel_for<Finish>(sycl::nd_range<1>(((m*n+255)/256)*256,256),[=](sycl::nd_item<1> it){
   int i=it.get_global_id(0);if(i>=m*n)return;float sum=0.f;
   for(int tile=0;tile<tiles;++tile)sum+=pp[tile*m*n+i];
   T b=T(sum);yp[i]=b;
   int col=i%n,row=i/n;
   if(activated && col<320){float a=float(b)*.25f;lp[row*320+col]=T(a*sycl::native::recip(1.f+sycl::native::exp2(-a*1.4426950408889634f)));}
 });
 return {raw,lora};
}
}
TORCH_LIBRARY(astra_down_v2,m){m.def("down(Tensor x, Tensor w, int valid, int split, bool activated) -> Tensor[]");}
TORCH_LIBRARY_IMPL(astra_down_v2,XPU,m){m.impl("down",&AstraDownV2::down);}
