// Modified for the B70 Flash-Next DFlash2 campaign; notice export 2026-10-22.
// DFlash2 campaign native kernel/residency integration; original upstream licensing and attribution remain below. Exact accepted inputs and build commands are in production/manifests/native-builds.json.
// Implementation attribution: Wulf integration and Astra campaign records; see licenses/NOTICE.md.
// This notice adds no license grant and does not replace upstream notices.
/***************************************************************************************************
 * Copyright (c) 2025 Intel Corporation. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 *LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
    \brief CUTLASS Intel BMG MoE API example based on sycl-tla Group GEMM

*/
#include <torch/all.h>
#include "csrc/utils.h"

#include <cute/tensor.hpp>
#include <random>

#include <cute/util/compat.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <sycl/sycl.hpp>

#include <cute/tensor.hpp>

#include "cutlass/kernel_hardware_info.h"
#include "cutlass/platform/platform.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_fill.h"

#include "gemm_xe2_policy.hpp"
#include "grouped_gemm_xe2.hpp"

#pragma clang diagnostic ignored "-Wpass-failed"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

namespace AstraMoeK {
using namespace MoE;
using namespace cute;

// type tag to define a unique sycl kernel name
template <
    typename,
    typename,
    typename,
    typename,
    char,
    char,
    class,
    A_DTYPE,
    B_DTYPE>
class AstraGemmName;

template <
    char layoutA,
    char layoutB,
    class policy,
    A_DTYPE TENSOR_A_DTYPE,
    B_DTYPE TENSOR_B_DTYPE,
    typename ElementA,
    typename ElementB,
    typename ElementS,
    typename ElementBI,
    typename ElementD>
void MoEGEMMLauncher(
    sycl::queue& stream,
    const ElementA* activations,
    const ElementB* weights,
    const ElementS* scales,
    const ElementBI* bias,
    ElementD* outputs,
    const int gemm_n,
    const int gemm_k,
    const int* rows_per_expert,
    const int num_experts,
    const int group_size,
    int32_t* atomic_buffer) {
  using ElementA_non_CV = cutlass::platform::remove_cv_t<ElementA>;
  auto op = XE_DPAS_TT<8, float, ElementA_non_CV>{};

  using WGTile = typename policy::WGTile;
  using SGLayout = typename policy::SGLayout;
  using MMA = typename TiledMMAHelper<
      MMA_Atom<decltype(op)>,
      Layout<WGTile>,
      SGLayout>::TiledMMA;
  auto mma = MMA{};

  int sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
  auto MaxThreadsPerWorkgroup = size(mma);

  static constexpr int MaxThreadsPerSM = policy::threads;

  TORCH_CHECK(
      MaxThreadsPerSM % MaxThreadsPerWorkgroup == 0,
      "MaxThreadsPerSM must be divisible by MaxThreadsPerWorkgroup")

  sycl::range<3> local(1, 1, MaxThreadsPerWorkgroup);
  sycl::range<3> global(
      1, sm_count * MaxThreadsPerSM / MaxThreadsPerWorkgroup, 1);

  namespace syclex = sycl::ext::oneapi::experimental;
  namespace intelex = sycl::ext::intel::experimental;

  syclex::properties kernel_props{
      syclex::sub_group_size<16>, intelex::grf_size<256>};

  using GmemTiledCopyA = typename policy::GmemTiledCopyA;
  using GmemTiledCopyB = typename policy::GmemTiledCopyB;
  using GmemTiledCopyD = typename policy::GmemTiledCopyD;

  stream.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<int32_t, 1> local_mem(sycl::range<1>(1), cgh);
    cgh.parallel_for<AstraGemmName<
        ElementA,
        ElementB,
        ElementS,
        ElementD,
        layoutA,
        layoutB,
        policy,
        TENSOR_A_DTYPE,
        TENSOR_B_DTYPE>>(
        sycl::nd_range<3>{global * local, local}, kernel_props, [=](auto) {
          MoE::MoEGEMM<
              TENSOR_A_DTYPE,
              TENSOR_B_DTYPE,
              GmemTiledCopyA,
              GmemTiledCopyB,
              GmemTiledCopyD,
              layoutA,
              layoutB,
              'R'>(
              activations,
              weights,
              scales,
              bias,
              outputs,
              mma,
              rows_per_expert,
              num_experts,
              group_size,
              gemm_n,
              gemm_k,
              atomic_buffer,
              local_mem);
        });
  });
}


template<int N, int Threads = 512, int KT = 32>
struct Policy : MoE::xe_gemm_policy_base {
  using WGTile = Shape<_8, Int<N>, Int<KT>>;
  using SGLayout = Layout<Shape<_1, Int<N/16>, _1>, Stride<Int<N/16>, _1, _0>>;
  static constexpr int threads = Threads;
};

template<class P>
void launch(at::Tensor a, at::Tensor b, at::Tensor s, at::Tensor d,
            at::Tensor rows, int64_t n, int64_t k, int64_t experts) {
  using T = cutlass::bfloat16_t;
  auto& q = at::xpu::getCurrentXPUStream(a.device().index()).queue();
  auto atomic = at::zeros({1}, a.options().dtype(at::kInt));
  MoEGEMMLauncher<'R','C',P,A_DTYPE::BITS16,B_DTYPE::INT4>(q,
    reinterpret_cast<T*>(a.data_ptr()), reinterpret_cast<uint8_t*>(b.data_ptr()),
    reinterpret_cast<T*>(s.data_ptr()), static_cast<T*>(nullptr),
    reinterpret_cast<T*>(d.data_ptr()), n, k,
    rows.data_ptr<int>(), experts, 128, atomic.data_ptr<int>());
}
void grouped(at::Tensor a, at::Tensor b, at::Tensor s, at::Tensor d,
             at::Tensor rows, int64_t n, int64_t k, int64_t experts, int64_t variant) {
  TORCH_CHECK(a.scalar_type()==at::kBFloat16 && s.scalar_type()==at::kBFloat16 &&
              d.scalar_type()==at::kBFloat16 && b.scalar_type()==at::kChar,
              "Astra MoE requires BF16 activations/scales/output and packed int4");
  TORCH_CHECK(a.is_contiguous() && b.is_contiguous() && s.is_contiguous() &&
              d.is_contiguous() && rows.is_contiguous(), "contiguous tensors required");
  TORCH_CHECK(a.dim()==2 && b.dim()==3 && s.dim()==3 && d.dim()==2 && rows.dim()==1,
              "invalid ranks");
  TORCH_CHECK(a.size(1)==k && b.size(0)==experts && b.size(1)==n && b.size(2)*2==k &&
              d.size(0)==a.size(0) && d.size(1)==n && s.size(0)==experts &&
              s.size(1)==n && s.size(2)*128==k && rows.size(0)==experts &&
              rows.scalar_type()==at::kInt && n%32==0 && k%128==0, "invalid dimensions");
  switch(variant) {
    case 0: launch<Policy<64>>(a,b,s,d,rows,n,k,experts); break;
    case 1: launch<Policy<32>>(a,b,s,d,rows,n,k,experts); break;
    case 2: launch<Policy<128>>(a,b,s,d,rows,n,k,experts); break;
    case 3: launch<Policy<256>>(a,b,s,d,rows,n,k,experts); break;
    case 4: launch<Policy<64,256>>(a,b,s,d,rows,n,k,experts); break;
    case 5: launch<Policy<128,256>>(a,b,s,d,rows,n,k,experts); break;
    case 6: launch<Policy<64,512,64>>(a,b,s,d,rows,n,k,experts); break;
    case 7: launch<Policy<64,512,128>>(a,b,s,d,rows,n,k,experts); break;
    case 8: launch<Policy<128,512,64>>(a,b,s,d,rows,n,k,experts); break;
    case 9: launch<Policy<128,512,128>>(a,b,s,d,rows,n,k,experts); break;
    case 10: launch<Policy<32,512,128>>(a,b,s,d,rows,n,k,experts); break;
    default: TORCH_CHECK(false,"unknown Astra variant");
  }
}
} // namespace AstraMoeK
TORCH_LIBRARY(astra_moe_k, m) {
 m.def("grouped(Tensor a, Tensor b, Tensor s, Tensor(a!) d, Tensor rows, int n, int k, int experts, int variant) -> ()");
}
TORCH_LIBRARY_IMPL(astra_moe_k, XPU, m) {
 m.impl("grouped", &AstraMoeK::grouped);
}
