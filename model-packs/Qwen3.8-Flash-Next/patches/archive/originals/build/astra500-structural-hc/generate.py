from pathlib import Path
p=Path(__file__).parent
s=(p/'source/csrc/xpu/grouped_gemm/xe_2/gemm_xe2.hpp').read_text()
a=s.index('template <\n    class GmemTiledCopyA,');b=s.index('template <\n    class GmemTiledCopyA,',a+10)
fn=s[a:b].replace('void xe_gemm(','void fused_hc_up(')
fn=fn.replace('    TiledMMA const& mma) {','    TiledMMA const& mma, const cutlass::bfloat16_t* normalized,\n    cutlass::bfloat16_t* output, float* gates, int rows) {')
for line in ['  auto copy_c = get_block_2d_copy_D<GmemTiledCopyC>(mma, C);\n','  auto thr_copy_c = copy_c.get_slice(local_id);\n','  auto tCrC_out = thr_copy_c.partition_sg_fragment_S(gC);\n','  auto tCgC = thr_copy_c.partition_D(gC);\n']:
 fn=fn.replace(line,'')
fn=fn.replace('  reorder(tCrC, tCrC_out);\n  copy(copy_c, tCrC_out, tCgC);','''  const int sg=local_id/16, lane=local_id%16;
  for(int m=0;m<8;++m)
    gates[sg*128+m*16+lane]=float(cutlass::bfloat16_t(tCrC(m)));
  sycl::group_barrier(item.get_group());
  if(sg==0) {
    for(int m=0;m<8;++m) {
      int row=wg_m*8+m, col=wg_n*16+lane;
      if(row<rows) {
        float acc=0;
        for(int stream=0;stream<4;++stream) {
          float g=gates[stream*128+m*16+lane];
          float sig=sycl::native::recip(1.0f+sycl::native::exp2(-g*1.4426950408889634f));
          acc=sycl::fma(sig,float(normalized[row*10240+stream*2560+col]),acc);
        }
        output[row*2560+col]=cutlass::bfloat16_t(acc*.25f);
      }
    }
  }''')
(p/'fused_hc.hpp').write_text(s[:s.index('#include')]+'''#pragma once
#include "grouped_gemm_xe2.hpp"
namespace AstraHC {
using namespace cute;
using namespace MoE;
'''+fn+'\n}\n')
