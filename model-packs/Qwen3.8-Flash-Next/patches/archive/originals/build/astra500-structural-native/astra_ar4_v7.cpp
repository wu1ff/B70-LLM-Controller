// SPDX-License-Identifier: Apache-2.0
// TP4 extension of the retained TP2 Level Zero IPC collective.
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/experimental/esimd/memory.hpp>
#include <level_zero/ze_api.h>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>

#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <sys/syscall.h>
#include <unistd.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <vector>

namespace vllm {
namespace astra_ar4_v7 {

constexpr int kWgSize = 256;          // work-items per work-group
constexpr int kVecBytes = 16;         // bytes per work-item
constexpr int64_t kChunk = 8192;  // 4 KB per work-group
constexpr int kMaxSlots = 256;        // -> max 1 MB per call
// control region, in uint32 words: flag of slot w at w * kFlagStride (one
// flag per 64-byte line, written by the peer), epoch[w] at kEpochWord + w
// (owner only), error word at kErrWord.
constexpr int kFlagStride = 16;
constexpr int kEpochWord = 4 * kMaxSlots * kFlagStride;
constexpr int kErrWord = kEpochWord + kMaxSlots;
constexpr int64_t kCtrlBytes = 131072;
constexpr int64_t kCookieOff = kCtrlBytes - 8;
constexpr uint32_t kSpinLimit = 1u << 27;  // ~ seconds; avoids a hard GPU hang

#define ZE_CHECK(call)                                                   \
  do {                                                                   \
    ze_result_t _r = (call);                                             \
    TORCH_CHECK(_r == ZE_RESULT_SUCCESS, #call " failed: ze_result 0x", \
                std::hex, static_cast<uint32_t>(_r));                    \
  } while (0)

struct State {
  int device = -1;
  int rank = -1;
  int64_t max_bytes = 0;
  ze_context_handle_t ctx = nullptr;
  ze_device_handle_t dev = nullptr;
  char* own = nullptr;   // own buffer (device memory)
  char* peer[4] = {nullptr, nullptr, nullptr, nullptr};  // peer buffer opened via IPC
  ze_ipc_mem_handle_t ipc{};
  uint64_t cookie = 0;
  bool exported = false;
  int imported_fd[4] = {-1,-1,-1,-1};
};

static std::mutex g_mu;
static std::vector<std::unique_ptr<State>> g_states;

static State& get_state(int64_t h) {
  std::lock_guard<std::mutex> lk(g_mu);
  TORCH_CHECK(h >= 0 && h < (int64_t)g_states.size() && g_states[h],
              "custom_ar: invalid handle ", h);
  return *g_states[h];
}

// ---------------------------------------------------------------- kernel
template <typename T>
struct alignas(16) Vec {
  T v[kVecBytes / sizeof(T)];
};

// One slot per 4 KiB. Every source publishes into its own stripe of every
// peer's buffer. A rank may reuse parity e only after all peers published e+1,
// which proves that they completed their reads of e. Epochs are device owned;
// graph replays and changing uniform batch widths require no host resets.
class Ar5Kernel;
struct PeerPointers { char* p[4]; };
static void launch(sycl::queue& q, const uint16_t* x, uint16_t* y, int64_t n,
                   char* own, PeerPointers peers, int64_t max_bytes, int rank,
                   uint32_t order) {
  constexpr int V=128, WG=32;
  const int nwg=(n*2+kChunk-1)/kChunk;
  namespace es=sycl::ext::intel::esimd;
  namespace ex=sycl::ext::intel::experimental::esimd;
  q.parallel_for<Ar5Kernel>(sycl::nd_range<1>(nwg*WG,WG),
    [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
    int w=it.get_group(0), lid=it.get_local_id(0);
    auto* epoch=reinterpret_cast<uint32_t*>(own)+kEpochWord+w;
    uint32_t e=es::block_load<uint32_t,1>(epoch)[0]+1;
    int64_t half=(e&1)*max_bytes;
    int64_t off=w*kChunk+lid*V*2, i0=off/2;
    es::simd<uint16_t,V> mine=0;
    if(i0<n) {
      mine=es::block_load<uint16_t,V>(x+i0);
      for(int p=0;p<4;++p) if(p!=rank)
        ex::lsc_block_store<uint16_t,V,ex::lsc_data_size::default_size,
          es::cache_hint::uncached,es::cache_hint::uncached>(
          reinterpret_cast<uint16_t*>(peers.p[p]+kCtrlBytes+rank*2*max_bytes+half+off),mine);
    }
    es::fence<es::memory_kind::global,es::fence_flush_op::clean,es::fence_scope::system>();
    es::barrier();
    if(lid==0) {
      for(int p=0;p<4;++p) if(p!=rank)
        ex::lsc_block_store<uint32_t,1,ex::lsc_data_size::default_size,
          es::cache_hint::uncached,es::cache_hint::uncached>(
          reinterpret_cast<uint32_t*>(peers.p[p])+(rank*kMaxSlots+w)*kFlagStride,
          es::simd<uint32_t,1>(e));
      es::fence<es::memory_kind::global,es::fence_flush_op::clean,es::fence_scope::system>();
      for(int p=0;p<4;++p) if(p!=rank) {
        auto* flag=reinterpret_cast<uint32_t*>(own)+(p*kMaxSlots+w)*kFlagStride;
        uint32_t spins=0;
        while(int32_t(ex::lsc_atomic_update<es::atomic_op::load,uint32_t,1>(flag,0u)[0]-e)<0) {
          if(++spins>kSpinLimit) {
            es::block_store<uint32_t,1>(reinterpret_cast<uint32_t*>(own)+kErrWord,
                                       es::simd<uint32_t,1>(1));break;
          }
        }
      }
    }
    es::barrier();
    es::fence<es::memory_kind::global,es::fence_flush_op::invalidate,es::fence_scope::system_acquire>();
    if(i0<n) {
      es::simd<float,V> v[4];
      for(int p=0;p<4;++p) {
        es::simd<uint16_t,V> data=mine;
        if(p!=rank) data=ex::lsc_block_load<uint16_t,V,ex::lsc_data_size::default_size,
          es::cache_hint::uncached,es::cache_hint::uncached>(
          reinterpret_cast<uint16_t*>(own+kCtrlBytes+p*2*max_bytes+half+off));
        es::simd<uint32_t,V> bits=es::convert<uint32_t>(data);bits<<=16;
        v[p]=bits.bit_cast_view<float>();
      }
      es::simd<uint32_t,V> ix(i0,1), orders=order>>((ix%2560)/640*8);
      es::simd<float,V> acc=0;
      for(int j=0;j<4;++j) {
        es::simd<uint32_t,V> r=(orders>>(j*2))&3;
        es::simd<float,V> add=v[0];
        for(int p=1;p<4;++p)add.merge(v[p],r==p);
        acc=j==0?add:acc+add;
        es::simd<uint32_t,V> bits=acc.bit_cast_view<uint32_t>();
        es::simd<uint32_t,V> rounded=(bits+0x7fff+((bits>>16)&1))&0xffff0000;
        rounded.merge(bits|0x00400000,(bits&0x7fffffff)>0x7f800000);
        acc=rounded.bit_cast_view<float>();
      }
      es::simd<uint32_t,V> bits=acc.bit_cast_view<uint32_t>();
      es::block_store<uint16_t,V>(y+i0,es::convert<uint16_t>(bits>>16));
    }
    es::barrier();
    if(lid==0)es::block_store<uint32_t,1>(epoch,es::simd<uint32_t,1>(e));
  });
}

// ---------------------------------------------------------------- ops
int64_t custom_ar_create(int64_t max_bytes) {
  TORCH_CHECK(max_bytes > 0 && max_bytes % kChunk == 0,
              "custom_ar: max_bytes must be a positive multiple of ", kChunk);
  TORCH_CHECK(max_bytes <= kMaxSlots * kChunk, "custom_ar: max_bytes > ",
              kMaxSlots * kChunk);
  auto st = std::make_unique<State>();
  st->device = c10::xpu::current_device();
  st->max_bytes = max_bytes;
  sycl::queue& q = c10::xpu::getCurrentXPUStream().queue();
  st->ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
      q.get_context());
  st->dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
      q.get_device());
  const size_t total = kCtrlBytes + 8 * max_bytes;
  ze_device_mem_alloc_desc_t desc{ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC,
                                  nullptr, 0, 0};
  void* p = nullptr;
  ZE_CHECK(zeMemAllocDevice(st->ctx, &desc, total, 4096, st->dev, &p));
  st->own = static_cast<char*>(p);
  q.memset(st->own, 0, total).wait();
  // cookie identifying this buffer, checked by the peer after the IPC open
  st->cookie = ((uint64_t)getpid() << 32) ^ (uint64_t)(uintptr_t)st->own ^
               0x9e3779b97f4a7c15ull;
  q.memcpy(st->own + kCookieOff, &st->cookie, sizeof(uint64_t)).wait();
  std::lock_guard<std::mutex> lk(g_mu);
  g_states.push_back(std::move(st));
  return (int64_t)g_states.size() - 1;
}

// Returns [pid, fd, w0..w15]: the exporting pid, the dma-buf fd (valid only in
// THIS process) and the raw ze_ipc_mem_handle_t as 16 uint32 words.
std::vector<int64_t> custom_ar_export(int64_t h) {
  State& st = get_state(h);
  if (!st.exported) {
    ZE_CHECK(zeMemGetIpcHandle(st.ctx, st.own, &st.ipc));
    st.exported = true;
  }
  uint64_t fd = 0;
  if (zeMemGetFileDescriptorFromIpcHandleExp(st.ctx, st.ipc, &fd) !=
      ZE_RESULT_SUCCESS) {
    // Extension not supported by this driver: on Linux the Intel handle
    // starts with the dma-buf fd (int).
    int fd32 = 0;
    std::memcpy(&fd32, st.ipc.data, sizeof(fd32));
    fd = (uint64_t)fd32;
  }
  std::vector<int64_t> out{(int64_t)getpid(), (int64_t)fd,
                           (int64_t)(st.cookie & 0xffffffffu),
                           (int64_t)(st.cookie >> 32)};
  uint32_t w[ZE_MAX_IPC_HANDLE_SIZE / 4];
  std::memcpy(w, st.ipc.data, sizeof(w));
  for (uint32_t v : w) out.push_back(v);
  return out;
}

// Opens the peer buffer from the peer's custom_ar_export() result. Tries, in
// order: (0) the raw peer handle (drivers with opaque / pidfd IPC handles
// import the fd themselves), (1) pidfd_getfd + zeMemGetIpcHandleFromFileDescriptorExp,
// (2) pidfd_getfd + the peer handle with its fd field replaced by the local fd.
// Returns the mode that succeeded. mode_mask selects allowed modes (bit i).
int64_t custom_ar_open(int64_t h, int64_t rank, std::vector<int64_t> peer,
                       int64_t mode_mask, int64_t peer_rank) {
  State& st = get_state(h);
  TORCH_CHECK(rank >= 0 && rank < 4 && peer_rank >= 0 && peer_rank < 4 && peer_rank != rank, "custom_ar: world size 2 only, rank ",
              rank);
  TORCH_CHECK(st.peer[peer_rank] == nullptr, "custom_ar: peer already opened");
  constexpr size_t kHdr = 4;  // pid, fd, cookie lo, cookie hi
  TORCH_CHECK(peer.size() == kHdr + ZE_MAX_IPC_HANDLE_SIZE / 4,
              "custom_ar: bad peer handle info");
  const uint64_t peer_cookie =
      (uint64_t)(uint32_t)peer[2] | ((uint64_t)(uint32_t)peer[3] << 32);
  sycl::queue& q = c10::xpu::getCurrentXPUStream().queue();
  // true if p maps the peer's buffer (its cookie is readable through p)
  auto verify = [&](void* p) {
    uint64_t c = 0;
    q.memcpy(&c, static_cast<char*>(p) + kCookieOff, sizeof(c)).wait();
    if (c == peer_cookie) return true;
    zeMemCloseIpcHandle(st.ctx, p);
    return false;
  };
  ze_ipc_mem_handle_t raw{};
  uint32_t w[ZE_MAX_IPC_HANDLE_SIZE / 4];
  for (size_t i = 0; i < ZE_MAX_IPC_HANDLE_SIZE / 4; ++i)
    w[i] = (uint32_t)peer[kHdr + i];
  std::memcpy(raw.data, w, sizeof(w));
  void* p = nullptr;
  int64_t mode = -1;
  ze_result_t r0 = ZE_RESULT_ERROR_UNKNOWN, r1 = r0, r2 = r0;
  if (mode_mask & 1) {
    r0 = zeMemOpenIpcHandle(st.ctx, st.dev, raw, 0, &p);
    if (r0 == ZE_RESULT_SUCCESS) {
      if (verify(p)) mode = 0;
      else r0 = ZE_RESULT_ERROR_INVALID_ARGUMENT;
    }
  }
  if (mode < 0 && (mode_mask & 6)) {
    int pidfd = (int)syscall(SYS_pidfd_open, (pid_t)peer[0], 0);
    TORCH_CHECK(pidfd >= 0, "custom_ar: pidfd_open(", peer[0],
                ") failed, errno ", errno);
    int fd = (int)syscall(SYS_pidfd_getfd, pidfd, (int)peer[1], 0);
    int err = errno;
    close(pidfd);
    TORCH_CHECK(fd >= 0, "custom_ar: pidfd_getfd(", peer[1],
                ") failed, errno ", err,
                " (needs same user / ptrace permission)");
    if (mode_mask & 2) {
      ze_ipc_mem_handle_t ph{};
      r1 = zeMemGetIpcHandleFromFileDescriptorExp(st.ctx, (uint64_t)fd, &ph);
      if (r1 == ZE_RESULT_SUCCESS) {
        r1 = zeMemOpenIpcHandle(st.ctx, st.dev, ph, 0, &p);
        if (r1 == ZE_RESULT_SUCCESS) {
          if (verify(p)) mode = 1;
          else r1 = ZE_RESULT_ERROR_INVALID_ARGUMENT;
        }
      }
    }
    if (mode < 0 && (mode_mask & 4)) {
      ze_ipc_mem_handle_t ph = raw;
      int fd32 = fd;
      std::memcpy(ph.data, &fd32, sizeof(fd32));
      // the fd is now local: also make an embedded exporter pid (opaque
      // handle layout) point at this process
      uint32_t pw[ZE_MAX_IPC_HANDLE_SIZE / 4];
      std::memcpy(pw, ph.data, sizeof(pw));
      for (int i = 1; i < ZE_MAX_IPC_HANDLE_SIZE / 4; ++i)
        if (pw[i] == (uint32_t)peer[0]) pw[i] = (uint32_t)getpid();
      std::memcpy(ph.data + 4, pw + 1, sizeof(pw) - 4);
      r2 = zeMemOpenIpcHandle(st.ctx, st.dev, ph, 0, &p);
      if (r2 == ZE_RESULT_SUCCESS) {
        if (verify(p)) mode = 2;
        else r2 = ZE_RESULT_ERROR_INVALID_ARGUMENT;
      }
    }
    if (mode < 0) close(fd);
    else st.imported_fd[peer_rank] = fd;
  }
  TORCH_CHECK(mode >= 0, "custom_ar: could not open peer IPC handle: raw 0x",
              std::hex, (uint32_t)r0, " fdExp 0x", (uint32_t)r1,
              " patched 0x", (uint32_t)r2);
  st.peer[peer_rank] = static_cast<char*>(p);
  st.rank = (int)rank;
  return mode;
}

void custom_ar_reduce(at::Tensor x, at::Tensor y, int64_t h, int64_t order) {
  State& st=get_state(h);
  for(int p=0;p<4;++p)TORCH_CHECK(p==st.rank || st.peer[p],"peer missing");
  TORCH_CHECK(x.is_xpu() && x.get_device()==st.device && y.device()==x.device(),"wrong device");
  TORCH_CHECK(x.is_contiguous() && y.is_contiguous() && x.sizes()==y.sizes(),"invalid layout");
  TORCH_CHECK(x.scalar_type()==at::kBFloat16 && y.scalar_type()==at::kBFloat16,"BF16 required");
  TORCH_CHECK(x.numel()*2<=st.max_bytes && x.numel()%2560==0,"invalid shape");
  PeerPointers peers;for(int p=0;p<4;++p)peers.p[p]=st.peer[p];
  auto& q=c10::xpu::getCurrentXPUStream().queue();
  using T=uint16_t;
  launch(q,reinterpret_cast<const T*>(x.data_ptr()),reinterpret_cast<T*>(y.data_ptr()),
         x.numel(),st.own,peers,st.max_bytes,st.rank,uint32_t(order));
}

// Debug: nonzero if a spin wait timed out (peer never arrived).
int64_t custom_ar_error(int64_t h) {
  State& st = get_state(h);
  uint32_t v = 0;
  sycl::queue& q = c10::xpu::getCurrentXPUStream().queue();
  q.memcpy(&v, reinterpret_cast<uint32_t*>(st.own) + kErrWord,
           sizeof(v))
      .wait();
  return v;
}

void custom_ar_destroy(int64_t h) {
  std::unique_ptr<State> st;
  {
    std::lock_guard<std::mutex> lk(g_mu);
    TORCH_CHECK(h >= 0 && h < (int64_t)g_states.size() && g_states[h],
                "custom_ar: invalid handle ", h);
    st = std::move(g_states[h]);
  }
  c10::xpu::getCurrentXPUStream().queue().wait();
  for(int p=0;p<4;++p) if (st->peer[p]) zeMemCloseIpcHandle(st->ctx, st->peer[p]);
  for(int p=0;p<4;++p) if (st->imported_fd[p] >= 0) close(st->imported_fd[p]);
  if (st->exported) zeMemPutIpcHandle(st->ctx, st->ipc);
  if (st->own) zeMemFree(st->ctx, st->own);
}

int64_t custom_ar_max_bytes_limit() { return kMaxSlots * kChunk; }

}  // namespace astra_ar4_v7
}  // namespace vllm

TORCH_LIBRARY(astra_ar4_v7,m) {
 m.def("create(int max_bytes) -> int",&vllm::astra_ar4_v7::custom_ar_create);
 m.def("export(int h) -> int[]",&vllm::astra_ar4_v7::custom_ar_export);
 m.def("open(int h, int rank, int[] peer, int mode_mask, int peer_rank) -> int",&vllm::astra_ar4_v7::custom_ar_open);
 m.def("error(int h) -> int",&vllm::astra_ar4_v7::custom_ar_error);
 m.def("destroy(int h) -> ()",&vllm::astra_ar4_v7::custom_ar_destroy);
 m.def("reduce(Tensor x, Tensor(a!) y, int h, int order) -> ()");
}
TORCH_LIBRARY_IMPL(astra_ar4_v7,XPU,m) {m.impl("reduce",&vllm::astra_ar4_v7::custom_ar_reduce);}
