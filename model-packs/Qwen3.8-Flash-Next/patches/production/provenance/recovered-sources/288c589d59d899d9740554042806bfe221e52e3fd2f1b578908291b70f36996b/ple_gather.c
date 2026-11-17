// SPDX-License-Identifier: Apache-2.0
// Parallel row gather for the Disk BF16 PLE stage (B70 DFlash2 v0.33 experiment).
//
// Byte-for-byte replacement of Qwen4ExpPLEDiskBF16Embedding._fill_stage's
// CPU work: zero the padded stage prefix, validate active IDs, then copy each
// owned row from the SAME read-only file mmap into the pinned stage. Rows are
// distributed over a small persistent thread pool so that page faults on the
// disk-backed mapping (minor and major) overlap instead of running serially.
// No table data is retained or made resident by this code.
#define _GNU_SOURCE
#include <pthread.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <sys/mman.h>

#define MAX_THREADS 32

typedef struct {
  const char* table;
  int64_t row_bytes;
  const int64_t* ids;       // flattened [n_ids]
  const int64_t* order;     // owned flat indices
  int64_t n_owned;
  char* stage;              // flattened rows [n_ids, row_bytes]
} job_t;

static pthread_t threads[MAX_THREADS];
static int nthreads = 0;
static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t cv_start = PTHREAD_COND_INITIALIZER;
static pthread_cond_t cv_done = PTHREAD_COND_INITIALIZER;
static uint64_t generation = 0;
static int remaining = 0;
static job_t job;

static void copy_part(int part, int parts) {
  for (int64_t j = part; j < job.n_owned; j += parts) {
    int64_t i = job.order[j];
    memcpy(job.stage + i * job.row_bytes, job.table + job.ids[i] * job.row_bytes,
           (size_t)job.row_bytes);
  }
}

static void* worker(void* arg) {
  int part = (int)(intptr_t)arg;  // workers take parts 1..nthreads; caller takes 0
  uint64_t seen = 0;
  for (;;) {
    pthread_mutex_lock(&mu);
    while (generation == seen) pthread_cond_wait(&cv_start, &mu);
    seen = generation;
    pthread_mutex_unlock(&mu);
    copy_part(part, nthreads + 1);
    pthread_mutex_lock(&mu);
    if (--remaining == 0) pthread_cond_signal(&cv_done);
    pthread_mutex_unlock(&mu);
  }
  return NULL;
}

int ple_gather_init(int n) {
  if (nthreads) return nthreads;
  if (n < 0) n = 0;
  if (n > MAX_THREADS) n = MAX_THREADS;
  for (int t = 0; t < n; ++t) {
    if (pthread_create(&threads[t], NULL, worker, (void*)(intptr_t)(t + 1)) != 0) return -1;
    pthread_detach(threads[t]);
    nthreads = t + 1;
  }
  return nthreads;
}

int ple_advise(void* start, size_t len, int advice) { return madvise(start, len, advice); }

// Returns 0 on success, 1 if an active ID is outside [0, vocab).
// padding: nullable per-token bool; flat index i belongs to token i / heads.
int ple_gather(const char* table, int64_t row_bytes, int64_t vocab,
               const int64_t* ids, const uint8_t* padding, int64_t heads, int64_t n_ids,
               int64_t vstart, int64_t vend, char* stage, int64_t zero_bytes,
               int64_t* order /* scratch [n_ids] */) {
  memset(stage, 0, (size_t)zero_bytes);
  int64_t n = 0;
  for (int64_t i = 0; i < n_ids; ++i) {
    if (padding && padding[i / heads]) continue;
    int64_t id = ids[i];
    if (id < 0 || id >= vocab) return 1;
  }
  for (int64_t i = 0; i < n_ids; ++i) {
    if (padding && padding[i / heads]) continue;
    int64_t id = ids[i];
    if (id >= vstart && id < vend) order[n++] = i;
  }
  if (n == 0) return 0;
  job.table = table; job.row_bytes = row_bytes; job.ids = ids;
  job.order = order; job.n_owned = n; job.stage = stage;
  if (nthreads == 0 || n < 2) {
    for (int64_t j = 0; j < n; ++j) {
      int64_t i = order[j];
      memcpy(stage + i * row_bytes, table + ids[i] * row_bytes, (size_t)row_bytes);
    }
    return 0;
  }
  pthread_mutex_lock(&mu);
  remaining = nthreads;
  ++generation;
  pthread_cond_broadcast(&cv_start);
  pthread_mutex_unlock(&mu);
  copy_part(0, nthreads + 1);
  pthread_mutex_lock(&mu);
  while (remaining) pthread_cond_wait(&cv_done, &mu);
  pthread_mutex_unlock(&mu);
  return 0;
}
