// SPDX-License-Identifier: Apache-2.0
// Direct eight-token MoE layout: no activation remap or atomic tile queue.
#include <torch/all.h>
#include "csrc/utils.h"
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include "gemm_xe2_policy.hpp"
#include "consumer_gemm.hpp"
namespace AstraConsumers {
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
void route(sycl::queue& q,const Id* ids,int rank,int* meta) {
 q.parallel_for<Route<Id>>(sycl::nd_range<1>(512,512),[=](sycl::nd_item<1> it){
  int e=it.get_local_id(0),used=0;
  for(int i=0;i<80;++i)if(int(ids[i])==e)used=1;
  int first=(rank-e%4+4)%4;
  int amount=used*(1+(first==0));
  int offset=sycl::exclusive_scan_over_group(it.get_group(),amount,sycl::plus<int>());
  int count=sycl::reduce_over_group(it.get_group(),amount,sycl::plus<int>());
  if(used)for(int j=0;j<amount;++j) {
   int slot=(5*e+first+4*j)/4;
   meta[1+offset+j]=slot;meta[161+slot]=offset+j;
  }
  if(e==0)meta[0]=count;
 });
}

template<int Groups> class Chain;
template<int Groups>
void chain(sycl::queue& q,const T* x,const int8_t* w1,const T* s1,
           const int8_t* w2,const T* s2,T* stage,const int* meta) {
 using P=UpPolicy;
 using MMA=typename MoE::TiledMMAHelper<MMA_Atom<XE_DPAS_TT<8,float,T>>,Layout<typename P::WGTile>,typename P::SGLayout>::TiledMMA;
 auto mma=MMA{};
 namespace sx=sycl::ext::oneapi::experimental; namespace ix=sycl::ext::intel::experimental;
 sx::properties props{sx::sub_group_size<16>,ix::grf_size<256>};
 q.submit([&](sycl::handler& cgh) {
  sycl::local_accessor<T,1> act(sycl::range<1>(8*128),cgh);
  cgh.parallel_for<Chain<Groups>>(sycl::nd_range<3>(sycl::range<3>(1,160,Groups*16),sycl::range<3>(1,1,Groups*16)),props,
   [=](sycl::nd_item<3> it) {
    int ordinal=it.get_group(1);
    if(ordinal>=meta[0])return;
    int e=meta[1+ordinal],sg=it.get_local_id(2)/16,lane=it.get_local_id(2)%16;
    auto A=MoE::make_moe_tensor<const T,'R'>(x,8,2560);
    auto B0=MoE::make_moe_tensor<int4_t,'R'>(reinterpret_cast<int4_t*>(const_cast<int8_t*>(w1)+int64_t(e)*256*1280),128,2560);
    auto B1=MoE::make_moe_tensor<int4_t,'R'>(reinterpret_cast<int4_t*>(const_cast<int8_t*>(w1)+int64_t(e)*256*1280+128*1280),128,2560);
    auto D=MoE::make_moe_tensor<T,'R'>(stage+ordinal*8*2560,8,128);
    // For 16 subgroups, pairs separately own gate and up; 8 subgroups
    // execute both serially. The paired version uses 4KiB scratch below.
    for(int tile=sg;tile<8;tile+=Groups) {
     auto coord=make_coord(0,tile,_,0);
     auto gate=direct_gemm<void,void,void,MoE::B_DTYPE::INT4,128>(A,B0,s1+int64_t(e)*256*20,static_cast<T*>(nullptr),D,coord,mma);
     auto value=direct_gemm<void,void,void,MoE::B_DTYPE::INT4,128>(A,B1,s1+int64_t(e)*256*20+128*20,static_cast<T*>(nullptr),D,coord,mma);
     int col=tile*16+lane;
     for(int row=0;row<8;++row) {
      float g=float(T(gate(row))),v=float(T(value(row)));
      float silu=float(T(g/(1.0f+sycl::exp(-g))));
      act[row*128+col]=T(silu*v);
     }
    }
    sycl::group_barrier(it.get_group());
    auto LA=MoE::make_moe_tensor<const T,'R'>(x,8,128); // coordinates only
    auto B=MoE::make_moe_tensor<int4_t,'R'>(reinterpret_cast<int4_t*>(const_cast<int8_t*>(w2)+int64_t(e)*2560*64),2560,128);
    auto Out=MoE::make_moe_tensor<T,'R'>(stage+ordinal*8*2560,8,2560);
    for(int tile=sg;tile<160;tile+=Groups) {
     auto accum=direct_gemm<void,void,void,MoE::B_DTYPE::INT4,128,true>(LA,B,s2+int64_t(e)*2560,static_cast<T*>(nullptr),Out,make_coord(0,tile,_,0),mma,act.template get_multi_ptr<sycl::access::decorated::no>().get());
     for(int row=0;row<8;++row)stage[(ordinal*8+row)*2560+tile*16+lane]=T(accum(row));
    }
   });
 });
}
template<class Id>
void gather(sycl::queue& q,const T* stage,const Id* ids,int rank,const int* meta,const float* weights,T* out) {
 q.parallel_for<Gather<Id>>(sycl::nd_range<1>(8*2560,256),[=](sycl::nd_item<1> it){
  int i=it.get_global_id(0),row=i/2560,col=i%2560;
  float sum=0;
  for(int k=0;k<10;++k) {
   int e=int(ids[row*10+k]);
   if(e>=0 && e<512)for(int g=0;g<5;++g)if((5*e+g)%4==rank) {
    int ordinal=meta[161+(5*e+g)/4];
    sum+=float(stage[(ordinal*8+row)*2560+col])*weights[row*10+k];
   }
  }
  out[i]=T(sum);
 });
}

void forward(at::Tensor x,at::Tensor w1,at::Tensor s1,at::Tensor w2,at::Tensor s2,
             at::Tensor ids,at::Tensor weights,int64_t rank,at::Tensor output) {
 TORCH_CHECK(x.sizes()==at::IntArrayRef({8,2560}) && output.sizes()==x.sizes(),"eight target rows required");
 TORCH_CHECK(x.scalar_type()==at::kBFloat16 && output.scalar_type()==at::kBFloat16 &&
  s1.scalar_type()==at::kBFloat16 && s2.scalar_type()==at::kBFloat16 &&
  w1.scalar_type()==at::kChar && w2.scalar_type()==at::kChar &&
  weights.scalar_type()==at::kFloat,"wrong dtype");
 TORCH_CHECK(w1.sizes()==at::IntArrayRef({640,256,1280}) && s1.sizes()==at::IntArrayRef({640,256,20}) &&
  w2.sizes()==at::IntArrayRef({640,2560,64}) && s2.sizes()==at::IntArrayRef({640,2560,1}) &&
  ids.sizes()==at::IntArrayRef({8,10}) && weights.sizes()==ids.sizes() && rank>=0 && rank<4,"wrong geometry");
 for(const auto& t:{x,w1,s1,w2,s2,ids,weights,output})
  TORCH_CHECK(t.is_contiguous() && t.device()==x.device(),"contiguous same-device tensors required");
 auto meta=at::empty({801},x.options().dtype(at::kInt));
 
 auto stage=at::empty({160,8,2560},x.options());
 auto& q=at::xpu::getCurrentXPUStream(x.device().index()).queue();
 if(ids.scalar_type()==at::kInt) route(q,ids.data_ptr<int32_t>(),rank,meta.data_ptr<int>());
 else { TORCH_CHECK(ids.scalar_type()==at::kLong,"IDs must be int32/int64");route(q,ids.data_ptr<int64_t>(),rank,meta.data_ptr<int>()); }
 chain<8>(q,reinterpret_cast<T*>(x.data_ptr()),w1.data_ptr<int8_t>(),reinterpret_cast<T*>(s1.data_ptr()),w2.data_ptr<int8_t>(),reinterpret_cast<T*>(s2.data_ptr()),reinterpret_cast<T*>(stage.data_ptr()),meta.data_ptr<int>());
 if(ids.scalar_type()==at::kInt) gather(q,reinterpret_cast<T*>(stage.data_ptr()),ids.data_ptr<int32_t>(),rank,meta.data_ptr<int>(),weights.data_ptr<float>(),reinterpret_cast<T*>(output.data_ptr()));
 else gather(q,reinterpret_cast<T*>(stage.data_ptr()),ids.data_ptr<int64_t>(),rank,meta.data_ptr<int>(),weights.data_ptr<float>(),reinterpret_cast<T*>(output.data_ptr()));
}
}
TORCH_LIBRARY(astra_consumers,m) {
 m.def("forward(Tensor x, Tensor w1, Tensor s1, Tensor w2, Tensor s2, Tensor ids, Tensor weights, int rank, Tensor(a!) output) -> ()");
}
TORCH_LIBRARY_IMPL(astra_consumers,XPU,m){m.impl("forward",&AstraConsumers::forward);}
