// SPDX-License-Identifier: Apache-2.0
// Two W4 projections share the loaded A fragment and loop schedule. Each
// subgroup owns all accumulators; there is no cross-subgroup state or barrier.
#pragma once
#include "grouped_gemm_xe2.hpp"
namespace DominantW4Fast {
using namespace cute;
using namespace MoE;
using T=cutlass::bfloat16_t;
template<class ATensor,class BTensor,class DTensor,class MMA>
CUTE_DEVICE auto dual_gemm(ATensor const& A, BTensor const& B0, BTensor const& B1,
 const T* S0,const T* S1,DTensor& C,int tile,MMA const& mma) {
 auto item=sycl::ext::oneapi::this_work_item::get_nd_item<3>();
 int lane=item.get_local_linear_id()%16;
 auto wg=mma.tile_mnk();
 auto ca=make_identity_tensor(A.shape());
 auto cb=make_identity_tensor(B0.shape());
 auto cc=make_identity_tensor(C.shape());
 auto ga=local_tile(ca,select<0,2>(wg),make_coord(0,_));
 auto gb=local_tile(cb,select<1,2>(wg),make_coord(tile,_));
 auto gc=local_tile(cc,wg,make_coord(0,tile,0),Step<_1,_1,X>{});
 auto ac=get_block_2d_copy_A<void>(mma,A);
 auto bc0=get_block_2d_copy_B<void>(mma,B0);
 auto bc1=get_block_2d_copy_B<void>(mma,B1);
 auto ma=mma.get_slice(lane);
 auto ta=ac.get_slice(lane);
 auto tb0=bc0.get_slice(lane);
 auto tb1=bc1.get_slice(lane);
 auto ar=ta.partition_sg_fragment_D(ga(_,_,0));
 auto br0=tb0.partition_sg_fragment_D(gb(_,_,0));
 auto br1=tb1.partition_sg_fragment_D(gb(_,_,0));
 auto am=ma.partition_sg_fragment_A(ga(_,_,0));
 auto bm0=ma.partition_sg_fragment_B(gb(_,_,0));
 auto bm1=ma.partition_sg_fragment_B(gb(_,_,0));
 auto accum0=ma.partition_sg_fragment_C(gc);
 auto accum1=ma.partition_sg_fragment_C(gc);
 auto as=ta.partition_S(ga);
 auto bs0=tb0.partition_S(gb);
 auto bs1=tb1.partition_S(gb);
 auto ap=make_block_2d_prefetch(ac);
 auto bp0=make_block_2d_prefetch(bc0);
 auto bp1=make_block_2d_prefetch(bc1);
 auto aps=ap.get_slice(lane).partition_S(ga);
 auto bps0=bp0.get_slice(lane).partition_S(gb);
 auto bps1=bp1.get_slice(lane).partition_S(gb);
 constexpr int tilek=get<2>(wg);
 constexpr int tilen=get<1>(wg);
 constexpr int channels=get<0>(get<0>(bm0.shape()));
 constexpr int nt=get<1>(bm0.shape());
 const int groups=get<1>(A.shape())/128;
 const int count=ceil_div(get<1>(A.shape()),tilek);
 constexpr int distance=6;
 float scales0[nt*channels],scales1[nt*channels];
 clear(accum0);clear(accum1);
 CUTE_UNROLL
 for(int t=0;t<distance;++t) {
  if(t<count){prefetch(ap,aps(_,_,_,t));prefetch(bp0,bps0(_,_,_,t));prefetch(bp1,bps1(_,_,_,t));}
 }
 for(int t=0;t<count;++t) {
  copy(ac,as(_,_,_,t),ar);
  copy(bc0,bs0(_,_,_,t),br0);
  copy(bc1,bs1(_,_,_,t),br1);
  if((t*tilek)%128==0) {
   int group=t*tilek/128;
   CUTE_UNROLL
   for(int n=0;n<nt;++n) {
    CUTE_UNROLL
    for(int c=0;c<channels;++c) {
     int col=tile*tilen+n*16+lane/channels+c*(16/channels);
     scales0[n*channels+c]=float(S0[col*groups+group]);
     scales1[n*channels+c]=float(S1[col*groups+group]);
    }
   }
  }
  if(t+distance<count) {
   prefetch(ap,aps(_,_,_,t+distance));
   prefetch(bp0,bps0(_,_,_,t+distance));prefetch(bp1,bps1(_,_,_,t+distance));
  }
  reorder(ar,am);reorder(br0,bm0);reorder(br1,bm1);
  CUTE_UNROLL
  for(int n=0;n<nt;++n) {
   CUTE_UNROLL
   for(int c=0;c<channels;++c) {
    CUTE_UNROLL
    for(int j=0;j<bm0.size()/nt/channels;++j) {
     bm0(cute::tuple(c,_),n,_)[j]=apply_scale(bm0(cute::tuple(c,_),n,_)[j],scales0[n*channels+c]);
     bm1(cute::tuple(c,_),n,_)[j]=apply_scale(bm1(cute::tuple(c,_),n,_)[j],scales1[n*channels+c]);
    }
   }
  }
  cute::gemm(mma,am,bm0,accum0);cute::gemm(mma,am,bm1,accum1);
 }
 return cute::make_tuple(accum0,accum1);
}
}
