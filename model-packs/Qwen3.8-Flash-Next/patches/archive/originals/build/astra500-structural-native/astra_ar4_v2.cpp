// SPDX-License-Identifier: Apache-2.0
// TP4 extension of the retained TP2 Level Zero IPC collective.
#include <sycl/sycl.hpp>
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
namespace astra_ar4 {

constexpr int kWgSize = 256;          // work-items per work-group
constexpr int kVecBytes = 16;         // bytes per work-item
constexpr int64_t kChunk = kWgSize * kVecBytes;  // 4 KB per work-group
constexpr int kMaxSlots = 256;        // -> max 1 MB per call
// control region, in uint32 words: flag of slot w at w * kFlagStride (one
// flag per 64-byte line, written by the peer), epoch[w] at kEpochWord + w
// (owner only), error word at kErrWord.
constexpr int kFlagStride = 16;
constexpr int kEpochWord = 4 * kMaxSlots * kFlagStride;
constexpr int kErrWord = kEpochWord + kMaxSlots;
constexpr int64_t kCtrlBytes = 131072;
constexpr int64_t kCookieOff = kCtrlBytes - 8;
constexpr uint32_t kSpinLimit = 1u << 24;  // ~ seconds; avoids a hard GPU hang

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
template <typename T> class Ar4Kernel;
struct PeerPointers { char* p[4]; };
template <typename T>
static void launch(sycl::queue& q, const T* x, T* y, int64_t n, char* own,
                   PeerPointers peers, int64_t max_bytes, int rank,
                   uint32_t order) {
  constexpr int kPer = kVecBytes / sizeof(T);
  const int nwg = (n * sizeof(T) + kChunk - 1) / kChunk;
  uint32_t* own_flag = reinterpret_cast<uint32_t*>(own);
  uint32_t* own_epoch = own_flag + kEpochWord;
  uint32_t* own_err = own_flag + kErrWord;
  using sycl::memory_order; using sycl::memory_scope;
  using sys_ref = sycl::atomic_ref<uint32_t, memory_order::relaxed,
      memory_scope::system, sycl::access::address_space::global_space>;
  q.parallel_for<Ar4Kernel<T>>(sycl::nd_range<1>(nwg*kWgSize,kWgSize),
    [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
    const int w=it.get_group(0), lid=it.get_local_id(0);
    const uint32_t e=own_epoch[w]+1;
    const int64_t half=(e&1)*max_bytes;
    const int64_t off=w*kChunk+lid*kVecBytes;
    const int64_t i0=off/sizeof(T);
    Vec<T> mine;
    if(i0+kPer<=n) mine=*reinterpret_cast<const Vec<T>*>(x+i0);
    else for(int k=0;k<kPer;++k) mine.v[k]=i0+k<n?x[i0+k]:T(0);
    for(int p=0;p<4;++p) {
      if(p!=rank && i0<n)
        *reinterpret_cast<Vec<T>*>(peers.p[p]+kCtrlBytes+rank*2*max_bytes+half+off)=mine;
    }
    sycl::atomic_fence(memory_order::release,memory_scope::system);
    sycl::group_barrier(it.get_group());
    if(lid==0) {
      sycl::atomic_fence(memory_order::release,memory_scope::system);
      for(int p=0;p<4;++p) if(p!=rank)
        *reinterpret_cast<volatile uint32_t*>(peers.p[p]+(rank*kMaxSlots+w)*kFlagStride*4)=e;
      for(int p=0;p<4;++p) if(p!=rank) {
        sys_ref f(own_flag[(p*kMaxSlots+w)*kFlagStride]);
        uint32_t spins=0;
        while((int32_t)(f.load(memory_order::acquire)-e)<0)
          if(++spins>kSpinLimit) { own_err[0]=1; break; }
      }
    }
    sycl::group_barrier(it.get_group());
    sycl::atomic_fence(memory_order::acquire,memory_scope::system);
    if(i0<n) {
      Vec<T> v[4],out;
      for(int p=0;p<4;++p) v[p]=p==rank?mine:
        *reinterpret_cast<const Vec<T>*>(own+kCtrlBytes+p*2*max_bytes+half+off);
      for(int k=0;k<kPer;++k) {
        // Match the retained BF16 ring's row-quarter reduction order.
        const uint32_t o=order>>(((i0+k)%2560)/640*8);
        T acc=v[o&3].v[k];
        for(int j=1;j<4;++j) acc=T(float(acc)+float(v[(o>>(j*2))&3].v[k]));
        out.v[k]=acc;
      }
      if(i0+kPer<=n) *reinterpret_cast<Vec<T>*>(y+i0)=out;
      else for(int k=0;k<kPer;++k) if(i0+k<n)y[i0+k]=out.v[k];
    }
    if(lid==0)own_epoch[w]=e;
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
  using T=sycl::ext::oneapi::bfloat16;
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

}  // namespace astra_ar4
}  // namespace vllm

TORCH_LIBRARY(astra_ar4,m) {
 m.def("create(int max_bytes) -> int",&vllm::astra_ar4::custom_ar_create);
 m.def("export(int h) -> int[]",&vllm::astra_ar4::custom_ar_export);
 m.def("open(int h, int rank, int[] peer, int mode_mask, int peer_rank) -> int",&vllm::astra_ar4::custom_ar_open);
 m.def("error(int h) -> int",&vllm::astra_ar4::custom_ar_error);
 m.def("destroy(int h) -> ()",&vllm::astra_ar4::custom_ar_destroy);
 m.def("reduce(Tensor x, Tensor(a!) y, int h, int order) -> ()");
}
TORCH_LIBRARY_IMPL(astra_ar4,XPU,m) {m.impl("reduce",&vllm::astra_ar4::custom_ar_reduce);}
