// SPDX-License-Identifier: Apache-2.0
// Direct eight-token MoE layout: no activation remap or atomic tile queue.
#include <torch/all.h>
#include "csrc/utils.h"
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include "gemm_xe2_policy.hpp"
#include "direct_gemm.hpp"
namespace AstraDirect2 {
using T=cutlass::bfloat16_t;
struct UpPolicy:MoE::xe_gemm_policy_base {
 using WGTile=Shape<_8,_16,_64>;
 using SGLayout=Layout<Shape<_1,_1,_1>,Stride<_1,_1,_0>>;
};
struct DownPolicy:MoE::xe_gemm_policy_base {
 using WGTile=Shape<_8,_64,_64>;
 using SGLayout=Layout<Shape<_1,_4,_1>,Stride<_4,_1,_0>>;
};
template<class Id>class Route;
class GateUp;
class Down;
template<class Id>class Gather;

template<class Id>
void route(sycl::queue& q,const Id* ids,const int* map,int* meta) {
 q.parallel_for<Route<Id>>(sycl::nd_range<1>(128,128),[=](sycl::nd_item<1> it){
  int e=it.get_local_id(0),used=0;
  for(int i=0;i<80;++i) {
   int id=int(ids[i]);
   if(id>=0 && id<512 && map[id]==e)used=1;
  }
  int offset=sycl::exclusive_scan_over_group(it.get_group(),used,sycl::plus<int>());
  int count=sycl::reduce_over_group(it.get_group(),used,sycl::plus<int>());
  if(used)meta[1+offset]=e;
  if(e==0)meta[0]=count;
 });
}
void up(sycl::queue& q,const T* x,const int8_t* w,const T* s,T* act,const int* meta) {
 using P=UpPolicy;
 using MMA=typename MoE::TiledMMAHelper<MMA_Atom<XE_DPAS_TT<8,float,T>>,Layout<typename P::WGTile>,typename P::SGLayout>::TiledMMA;
 auto mma=MMA{};
 namespace sx=sycl::ext::oneapi::experimental; namespace ix=sycl::ext::intel::experimental;
 sx::properties props{sx::sub_group_size<16>,ix::grf_size<256>};
 q.parallel_for<GateUp>(sycl::nd_range<3>(sycl::range<3>(1,1024,16),sycl::range<3>(1,1,16)),props,
  [=](sycl::nd_item<3> it){
   int total=meta[0]*40;
   for(int task=it.get_group(1);task<total;task+=1024) {
    int e=meta[1+task/40],tile=task%40;
    auto A=MoE::make_moe_tensor<const T,'R'>(x,8,2560);
    auto B0=MoE::make_moe_tensor<int4_t,'R'>(reinterpret_cast<int4_t*>(const_cast<int8_t*>(w)+int64_t(e)*1280*1280),640,2560);
    auto B1=MoE::make_moe_tensor<int4_t,'R'>(reinterpret_cast<int4_t*>(const_cast<int8_t*>(w)+int64_t(e)*1280*1280+640*1280),640,2560);
    auto D=MoE::make_moe_tensor<T,'R'>(act+e*8*640,8,640);
    auto coord=make_coord(0,tile,_,0);
    auto gate=direct_gemm<void,void,void,MoE::B_DTYPE::INT4,128>(A,B0,s+int64_t(e)*1280*20,static_cast<T*>(nullptr),D,coord,mma);
    auto value=direct_gemm<void,void,void,MoE::B_DTYPE::INT4,128>(A,B1,s+int64_t(e)*1280*20+640*20,static_cast<T*>(nullptr),D,coord,mma);
    int col=tile*16+it.get_local_id(2);
    for(int m=0;m<8;++m) {
      float g=float(T(gate(m))),v=float(T(value(m)));
      float silu=float(T(g/(1.0f+sycl::exp(-g))));
      act[(e*8+m)*640+col]=T(silu*v);
    }
   }
  });
}
void down(sycl::queue& q,const T* act,const int8_t* w,const T* s,T* stage,const int* meta) {
 using P=DownPolicy;
 using MMA=typename MoE::TiledMMAHelper<MMA_Atom<XE_DPAS_TT<8,float,T>>,Layout<typename P::WGTile>,typename P::SGLayout>::TiledMMA;
 auto mma=MMA{};
 namespace sx=sycl::ext::oneapi::experimental; namespace ix=sycl::ext::intel::experimental;
 sx::properties props{sx::sub_group_size<16>,ix::grf_size<256>};
 q.parallel_for<Down>(sycl::nd_range<3>(sycl::range<3>(1,256,64),sycl::range<3>(1,1,64)),props,
  [=](sycl::nd_item<3> it){
   int total=meta[0]*40;
   for(int task=it.get_group(1);task<total;task+=256) {
    int e=meta[1+task/40],tile=task%40;
    auto A=MoE::make_moe_tensor<const T,'R'>(act+e*8*640,8,640);
    auto B=MoE::make_moe_tensor<int4_t,'R'>(reinterpret_cast<int4_t*>(const_cast<int8_t*>(w)+int64_t(e)*2560*320),2560,640);
    auto D=MoE::make_moe_tensor<T,'R'>(stage+e*8*2560,8,2560);
    auto coord=make_coord(0,tile,_,0);
    MoE::xe_gemm_4bits<void,void,void,MoE::B_DTYPE::INT4,128>(A,B,s+int64_t(e)*2560*5,static_cast<T*>(nullptr),D,coord,mma);
   }
  });
}
template<class Id>
void gather(sycl::queue& q,const T* stage,const Id* ids,const int* map,const float* weights,T* out) {
 q.parallel_for<Gather<Id>>(sycl::nd_range<1>(8*2560,256),[=](sycl::nd_item<1> it){
  int i=it.get_global_id(0),row=i/2560,col=i%2560;
  float sum=0;
  for(int k=0;k<10;++k) {
   int id=int(ids[row*10+k]);int e=id>=0 && id<512?map[id]:-1;
   if(e>=0)sum+=float(stage[(e*8+row)*2560+col])*weights[row*10+k];
  }
  out[i]=T(sum);
 });
}
void forward(at::Tensor x,at::Tensor w1,at::Tensor s1,at::Tensor w2,at::Tensor s2,
             at::Tensor ids,at::Tensor weights,at::Tensor map,at::Tensor output) {
 TORCH_CHECK(x.sizes()==at::IntArrayRef({8,2560}) && output.sizes()==x.sizes(),"eight target rows required");
 TORCH_CHECK(x.scalar_type()==at::kBFloat16 && output.scalar_type()==at::kBFloat16 &&
  s1.scalar_type()==at::kBFloat16 && s2.scalar_type()==at::kBFloat16 &&
  w1.scalar_type()==at::kChar && w2.scalar_type()==at::kChar &&
  map.scalar_type()==at::kInt && weights.scalar_type()==at::kFloat,"wrong dtype");
 TORCH_CHECK(w1.sizes()==at::IntArrayRef({128,1280,1280}) && s1.sizes()==at::IntArrayRef({128,1280,20}) &&
  w2.sizes()==at::IntArrayRef({128,2560,320}) && s2.sizes()==at::IntArrayRef({128,2560,5}) &&
  ids.sizes()==at::IntArrayRef({8,10}) && weights.sizes()==ids.sizes() && map.numel()==512,"wrong geometry");
 for(const auto& t:{x,w1,s1,w2,s2,ids,weights,map,output})
  TORCH_CHECK(t.is_contiguous() && t.device()==x.device(),"contiguous same-device tensors required");
 auto meta=at::empty({129},x.options().dtype(at::kInt));
 auto act=at::empty({128,8,640},x.options());
 auto stage=at::empty({128,8,2560},x.options());
 auto& q=at::xpu::getCurrentXPUStream(x.device().index()).queue();
 if(ids.scalar_type()==at::kInt) route(q,ids.data_ptr<int32_t>(),map.data_ptr<int>(),meta.data_ptr<int>());
 else { TORCH_CHECK(ids.scalar_type()==at::kLong,"IDs must be int32/int64");route(q,ids.data_ptr<int64_t>(),map.data_ptr<int>(),meta.data_ptr<int>()); }
 up(q,reinterpret_cast<T*>(x.data_ptr()),w1.data_ptr<int8_t>(),reinterpret_cast<T*>(s1.data_ptr()),reinterpret_cast<T*>(act.data_ptr()),meta.data_ptr<int>());
 down(q,reinterpret_cast<T*>(act.data_ptr()),w2.data_ptr<int8_t>(),reinterpret_cast<T*>(s2.data_ptr()),reinterpret_cast<T*>(stage.data_ptr()),meta.data_ptr<int>());
 if(ids.scalar_type()==at::kInt) gather(q,reinterpret_cast<T*>(stage.data_ptr()),ids.data_ptr<int32_t>(),map.data_ptr<int>(),weights.data_ptr<float>(),reinterpret_cast<T*>(output.data_ptr()));
 else gather(q,reinterpret_cast<T*>(stage.data_ptr()),ids.data_ptr<int64_t>(),map.data_ptr<int>(),weights.data_ptr<float>(),reinterpret_cast<T*>(output.data_ptr()));
}
}
TORCH_LIBRARY(astra_direct_v2,m) {
 m.def("forward(Tensor x, Tensor w1, Tensor s1, Tensor w2, Tensor s2, Tensor ids, Tensor weights, Tensor map, Tensor(a!) output) -> ()");
}
TORCH_LIBRARY_IMPL(astra_direct_v2,XPU,m){m.impl("forward",&AstraDirect2::forward);}
