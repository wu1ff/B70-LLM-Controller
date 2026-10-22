#define _GNU_SOURCE

#include <level_zero/layers/zel_tracing_api.h>

#include <dlfcn.h>
#include <execinfo.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

/*
 * Narrow workaround for intel/compute-runtime#968.
 *
 * The Level Zero loader's official tracing callbacks leave the operation
 * pointers seen by UMF/XCCL unchanged. Successful device allocations are
 * tracked by context, range, and owner. A positively classified peer residency
 * is redirected to the owner device, avoiding the peer GTT residency while
 * retaining a normal Level Zero call and result. Pool-sized and oneCCL-owned
 * allocations preserve their original residency because they are runtime
 * infrastructure rather than the large provider allocations targeted here.
 */

struct allocation {
    uintptr_t base;
    size_t size;
    ze_context_handle_t context;
    ze_device_handle_t owner;
    bool preserve_peer_residency;
    struct allocation *next;
};

static const size_t max_poolable_device_usm_size = 4 * 1024 * 1024;

static pthread_mutex_t allocation_lock = PTHREAD_MUTEX_INITIALIZER;
static struct allocation *allocations;
static zel_tracer_handle_t tracer;

static _Atomic uint64_t alloc_device_calls;
static _Atomic uint64_t allocations_tracked;
static _Atomic uint64_t frees_tracked;
static _Atomic uint64_t allocations_live;
static _Atomic uint64_t owner_residency_allowed;
static _Atomic uint64_t peer_residency_preserved;
static _Atomic uint64_t peer_residency_suppressed;
static _Atomic uint64_t peer_bytes_suppressed;
static _Atomic uint64_t other_residency_passthrough;

static pthread_once_t reporter_once = PTHREAD_ONCE_INIT;

static void shim_log_summary(const char *reason) {
    dprintf(STDERR_FILENO,
            "[l0-peer-shim pid=%ld] summary reason=%s path=loader-tracer "
            "alloc_calls=%llu tracked=%llu freed=%llu live=%llu "
            "owner_allowed=%llu peer_preserved=%llu peer_suppressed=%llu "
            "peer_bytes=%llu other_passthrough=%llu\n",
            (long)getpid(), reason,
            (unsigned long long)atomic_load_explicit(&alloc_device_calls,
                                                    memory_order_relaxed),
            (unsigned long long)atomic_load_explicit(&allocations_tracked,
                                                    memory_order_relaxed),
            (unsigned long long)atomic_load_explicit(&frees_tracked,
                                                    memory_order_relaxed),
            (unsigned long long)atomic_load_explicit(&allocations_live,
                                                    memory_order_relaxed),
            (unsigned long long)atomic_load_explicit(&owner_residency_allowed,
                                                    memory_order_relaxed),
            (unsigned long long)atomic_load_explicit(&peer_residency_preserved,
                                                    memory_order_relaxed),
            (unsigned long long)atomic_load_explicit(&peer_residency_suppressed,
                                                    memory_order_relaxed),
            (unsigned long long)atomic_load_explicit(&peer_bytes_suppressed,
                                                    memory_order_relaxed),
            (unsigned long long)atomic_load_explicit(
                &other_residency_passthrough, memory_order_relaxed));
}

struct counter_snapshot {
    uint64_t alloc_calls;
    uint64_t tracked;
    uint64_t freed;
    uint64_t live;
    uint64_t owner;
    uint64_t peer_preserved;
    uint64_t peer_suppressed;
    uint64_t peer_bytes;
    uint64_t other;
};

static struct counter_snapshot take_snapshot(void) {
    return (struct counter_snapshot){
        atomic_load_explicit(&alloc_device_calls, memory_order_relaxed),
        atomic_load_explicit(&allocations_tracked, memory_order_relaxed),
        atomic_load_explicit(&frees_tracked, memory_order_relaxed),
        atomic_load_explicit(&allocations_live, memory_order_relaxed),
        atomic_load_explicit(&owner_residency_allowed, memory_order_relaxed),
        atomic_load_explicit(&peer_residency_preserved, memory_order_relaxed),
        atomic_load_explicit(&peer_residency_suppressed, memory_order_relaxed),
        atomic_load_explicit(&peer_bytes_suppressed, memory_order_relaxed),
        atomic_load_explicit(&other_residency_passthrough, memory_order_relaxed),
    };
}

static void *reporter_main(void *unused) {
    (void)unused;
    struct counter_snapshot previous = take_snapshot();
    for (;;) {
        struct timespec delay = {.tv_sec = 15, .tv_nsec = 0};
        nanosleep(&delay, NULL);
        struct counter_snapshot current = take_snapshot();
        if (memcmp(&current, &previous, sizeof(current)) != 0) {
            shim_log_summary("periodic");
            previous = current;
        }
    }
    return NULL;
}

static void start_reporter(void) {
    pthread_t thread;
    if (pthread_create(&thread, NULL, reporter_main, NULL) == 0)
        pthread_detach(thread);
}

static void ensure_reporter(void) {
    pthread_once(&reporter_once, start_reporter);
}

static bool allocation_requires_peer_residency(void) {
    void *frames[64];
    int count = backtrace(frames, 64);
    for (int index = 0; index < count; ++index) {
        Dl_info info = {0};
        if (dladdr(frames[index], &info) == 0 || !info.dli_fname)
            continue;
        const char *base = strrchr(info.dli_fname, '/');
        base = base ? base + 1 : info.dli_fname;
        static const char ccl_prefix[] = "libccl.so.";
        if (strncmp(base, ccl_prefix, sizeof(ccl_prefix) - 1) == 0)
            return true;
    }
    return false;
}

static void track_allocation(ze_context_handle_t context,
                             ze_device_handle_t owner, void *pointer,
                             size_t size, bool preserve_peer_residency) {
    if (!pointer || size == 0)
        return;

    struct allocation *node = malloc(sizeof(*node));
    if (!node)
        return;
    *node = (struct allocation){
        .base = (uintptr_t)pointer,
        .size = size,
        .context = context,
        .owner = owner,
        .preserve_peer_residency = preserve_peer_residency,
        .next = NULL,
    };

    pthread_mutex_lock(&allocation_lock);
    struct allocation **cursor = &allocations;
    while (*cursor) {
        if ((*cursor)->base == node->base &&
            (*cursor)->context == node->context) {
            (*cursor)->size = node->size;
            (*cursor)->owner = node->owner;
            (*cursor)->preserve_peer_residency =
                node->preserve_peer_residency;
            pthread_mutex_unlock(&allocation_lock);
            free(node);
            atomic_fetch_add_explicit(&allocations_tracked, 1,
                                      memory_order_relaxed);
            ensure_reporter();
            return;
        }
        cursor = &(*cursor)->next;
    }
    node->next = allocations;
    allocations = node;
    pthread_mutex_unlock(&allocation_lock);

    atomic_fetch_add_explicit(&allocations_tracked, 1, memory_order_relaxed);
    atomic_fetch_add_explicit(&allocations_live, 1, memory_order_relaxed);
    ensure_reporter();
}

static void untrack_allocation(ze_context_handle_t context, void *pointer) {
    struct allocation *removed = NULL;
    pthread_mutex_lock(&allocation_lock);
    struct allocation **cursor = &allocations;
    while (*cursor) {
        if ((*cursor)->base == (uintptr_t)pointer &&
            (*cursor)->context == context) {
            removed = *cursor;
            *cursor = removed->next;
            break;
        }
        cursor = &(*cursor)->next;
    }
    pthread_mutex_unlock(&allocation_lock);

    if (removed) {
        free(removed);
        atomic_fetch_add_explicit(&frees_tracked, 1, memory_order_relaxed);
        atomic_fetch_sub_explicit(&allocations_live, 1, memory_order_relaxed);
    }
}

static bool classify_tracked_range(ze_context_handle_t context,
                                   const void *pointer, size_t size,
                                   ze_device_handle_t *owner,
                                   bool *preserve_peer_residency) {
    if (!pointer || size == 0)
        return false;
    uintptr_t start = (uintptr_t)pointer;
    if (size > UINTPTR_MAX - start)
        return false;
    uintptr_t end = start + size;
    bool found = false;

    pthread_mutex_lock(&allocation_lock);
    for (struct allocation *node = allocations; node; node = node->next) {
        if (node->context != context || node->size > UINTPTR_MAX - node->base)
            continue;
        uintptr_t allocation_end = node->base + node->size;
        if (start >= node->base && end <= allocation_end) {
            *owner = node->owner;
            *preserve_peer_residency = node->preserve_peer_residency;
            found = true;
            break;
        }
    }
    pthread_mutex_unlock(&allocation_lock);
    return found;
}

static void ZE_APICALL alloc_device_epilogue(
    ze_mem_alloc_device_params_t *params, ze_result_t result, void *user_data,
    void **instance_data) {
    (void)user_data;
    (void)instance_data;
    atomic_fetch_add_explicit(&alloc_device_calls, 1, memory_order_relaxed);
    if (result != ZE_RESULT_SUCCESS || !params || !params->phContext ||
        !params->phDevice || !params->psize || !params->ppptr ||
        !*params->ppptr)
        return;

    size_t size = *params->psize;
    bool preserve = size <= max_poolable_device_usm_size ||
                    allocation_requires_peer_residency();
    track_allocation(*params->phContext, *params->phDevice, **params->ppptr,
                     size, preserve);
}

static void ZE_APICALL mem_free_epilogue(ze_mem_free_params_t *params,
                                         ze_result_t result, void *user_data,
                                         void **instance_data) {
    (void)user_data;
    (void)instance_data;
    if (result == ZE_RESULT_SUCCESS && params && params->phContext &&
        params->pptr)
        untrack_allocation(*params->phContext, *params->pptr);
}

static void ZE_APICALL make_resident_prologue(
    ze_context_make_memory_resident_params_t *params, ze_result_t result,
    void *user_data, void **instance_data) {
    (void)result;
    (void)user_data;
    (void)instance_data;
    if (!params || !params->phContext || !params->phDevice || !params->pptr ||
        !params->psize)
        return;

    ze_device_handle_t owner = NULL;
    bool preserve = false;
    if (!classify_tracked_range(*params->phContext, *params->pptr,
                                *params->psize, &owner, &preserve)) {
        atomic_fetch_add_explicit(&other_residency_passthrough, 1,
                                  memory_order_relaxed);
        return;
    }

    if (owner == *params->phDevice) {
        atomic_fetch_add_explicit(&owner_residency_allowed, 1,
                                  memory_order_relaxed);
        return;
    }
    if (preserve) {
        atomic_fetch_add_explicit(&peer_residency_preserved, 1,
                                  memory_order_relaxed);
        return;
    }

    *params->phDevice = owner;
    atomic_fetch_add_explicit(&peer_residency_suppressed, 1,
                              memory_order_relaxed);
    atomic_fetch_add_explicit(&peer_bytes_suppressed, *params->psize,
                              memory_order_relaxed);
}

static void fatal_setup(ze_result_t result) {
    dprintf(STDERR_FILENO,
            "[l0-peer-shim pid=%ld] fatal tracer setup result=0x%x\n",
            (long)getpid(), (unsigned)result);
    _exit(127);
}

__attribute__((constructor)) static void shim_start(void) {
    if (setenv("ZE_ENABLE_TRACING_LAYER", "1", 1) != 0)
        fatal_setup(ZE_RESULT_ERROR_UNKNOWN);

    ze_result_t result = zeInit(0);
    zel_tracer_desc_t desc = {
        .stype = ZEL_STRUCTURE_TYPE_TRACER_DESC,
        .pNext = NULL,
        .pUserData = &tracer,
    };
    ze_callbacks_t prologues = {0};
    ze_callbacks_t epilogues = {0};
    prologues.Context.pfnMakeMemoryResidentCb = make_resident_prologue;
    epilogues.Mem.pfnAllocDeviceCb = alloc_device_epilogue;
    epilogues.Mem.pfnFreeCb = mem_free_epilogue;

    if (result == ZE_RESULT_SUCCESS)
        result = zelTracerCreate(&desc, &tracer);
    if (result == ZE_RESULT_SUCCESS)
        result = zelTracerSetPrologues(tracer, &prologues);
    if (result == ZE_RESULT_SUCCESS)
        result = zelTracerSetEpilogues(tracer, &epilogues);
    if (result == ZE_RESULT_SUCCESS)
        result = zelTracerSetEnabled(tracer, true);
    if (result != ZE_RESULT_SUCCESS)
        fatal_setup(result);

    dprintf(STDERR_FILENO,
            "[l0-peer-shim pid=%ld] active scope=large-nonccl-tracked-device-"
            "allocation-peer-residency path=loader-tracer "
            "pool_ceiling=%zu report_interval=15s\n",
            (long)getpid(), max_poolable_device_usm_size);
}

__attribute__((destructor)) static void shim_stop(void) {
    shim_log_summary("exit");
}
