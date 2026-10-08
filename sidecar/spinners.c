/* Copyright 2026 Stanford University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* Microbenchmark for schedule perturbation: N threads spin reading
 * CLOCK_MONOTONIC and record every gap in their own execution longer than a
 * threshold, i.e. every interval during which the thread was not running.
 *
 *   spinners NTHREADS SECONDS GAP_US OUTFILE [NCPUS]
 *
 * OUTFILE gets one line per gap: "tid start_ns end_ns". With NCPUS, the
 * spinning threads (only) are confined to CPUs 0..NCPUS-1, so that helper
 * threads injected into the process (e.g. by an LD_PRELOAD shim) do not
 * compete with them.
 */

#define _GNU_SOURCE
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#define MAX_GAPS (1 << 20)

static int64_t deadline, gap_ns;
static int ncpus;

struct Thread {
  pthread_t handle;
  pid_t tid;
  int64_t (*gaps)[2];
  long ngaps;
};

static int64_t now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

static void *spin(void *arg) {
  struct Thread *t = arg;
  t->tid = syscall(SYS_gettid);
  if (ncpus > 0) {
    cpu_set_t set;
    CPU_ZERO(&set);
    for (int i = 0; i < ncpus; i++) CPU_SET(i, &set);
    sched_setaffinity(0, sizeof(set), &set);
  }
  int64_t last = now_ns();
  while (last < deadline) {
    int64_t now = now_ns();
    if (now - last > gap_ns && t->ngaps < MAX_GAPS) {
      t->gaps[t->ngaps][0] = last;
      t->gaps[t->ngaps][1] = now;
      t->ngaps++;
    }
    last = now;
  }
  return NULL;
}

int main(int argc, char **argv) {
  if (argc != 5 && argc != 6) {
    fprintf(stderr, "usage: spinners NTHREADS SECONDS GAP_US OUTFILE [NCPUS]\n");
    return 2;
  }
  ncpus = (argc == 6) ? atoi(argv[5]) : 0;
  int n = atoi(argv[1]);
  gap_ns = atoll(argv[3]) * 1000;
  struct Thread *threads = calloc(n, sizeof(*threads));
  /* start after a short delay so the tracer's warmup sees all threads */
  deadline = now_ns() + (int64_t)(atof(argv[2]) * 1e9);
  for (int i = 0; i < n; i++) {
    threads[i].gaps = malloc(sizeof(*threads[i].gaps) * MAX_GAPS);
    pthread_create(&threads[i].handle, NULL, spin, &threads[i]);
  }
  for (int i = 0; i < n; i++) pthread_join(threads[i].handle, NULL);
  FILE *out = fopen(argv[4], "w");
  for (int i = 0; i < n; i++)
    for (long j = 0; j < threads[i].ngaps; j++)
      fprintf(out, "%d %lld %lld\n", threads[i].tid, (long long)threads[i].gaps[j][0],
              (long long)threads[i].gaps[j][1]);
  fclose(out);
  return 0;
}
