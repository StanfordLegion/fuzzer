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

/* Target program for lifecycle tests of signal_shim.so and ptrace_shim (see
 * the tests in CMakeLists.txt). Each mode runs NTHREADS workers for
 * SECONDS and checks its own invariants, exiting non-zero on any violation:
 *
 *   lifecycle_target MODE NTHREADS SECONDS
 *
 * Modes:
 *   spin        workers spin and call syscall(SYS_futex, FUTEX_WAKE) and
 *               syscall(SYS_gettid), checking results and errno
 *   abi         single-threaded syscall() return value / errno checks
 *   block       workers block SIGRTMIN+7 for the first half, then unblock
 *   leader-exit main thread calls pthread_exit after 1 s
 *   exec        a worker execs /bin/true after 1 s
 *   fork        forks a child (which runs the same checks) after 1 s
 *   handler     installs its own SIGRTMIN+7 handler after 1 s
 *   fdexhaust   after 1 s, uses up all file descriptors but one for 3 s
 *               while the workers keep running, then exits with status 3 (a
 *               wrapper that wrongly concludes all threads are gone would end
 *               the process early with status 0)
 */

#define _GNU_SOURCE
#include <errno.h>
#include <linux/futex.h>
#include <pthread.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

static const char *mode;
static double seconds;
static int nthreads;
static volatile int failures;
static uint32_t futex_word;

static double now_s(void)
{
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

static void fail(const char *what)
{
  fprintf(stderr, "lifecycle_target: FAIL %s (errno %d)\n", what, errno);
  __atomic_add_fetch(&failures, 1, __ATOMIC_RELAXED);
}

/* syscall() through the (possibly interposed) wrapper must behave like a
 * direct system call, including errno. */
static void check_syscalls(void)
{
  errno = 1234;
  long tid = syscall(SYS_gettid);
  if(tid != gettid() || errno != 1234)
    fail("syscall(SYS_gettid)");

  errno = 0;
  long r = syscall(-1L);
  if(r != -1 || errno != ENOSYS)
    fail("syscall(-1)");

  errno = 4321;
  r = syscall(SYS_futex, &futex_word, FUTEX_WAKE_PRIVATE, 1, NULL, NULL, 0);
  if(r < 0 || errno != 4321)
    fail("syscall(SYS_futex, FUTEX_WAKE)");

  errno = 0;
  r = syscall(SYS_futex, NULL, FUTEX_WAKE, 1, NULL, NULL, 0);
  if(r != -1 || errno != EFAULT)
    fail("syscall(SYS_futex, FUTEX_WAKE, NULL)");
}

static void *worker(void *arg)
{
  long idx = (long)arg;
  if(!strcmp(mode, "block")) {
    sigset_t set;
    sigemptyset(&set);
    sigaddset(&set, SIGRTMIN + 7);
    pthread_sigmask(SIG_BLOCK, &set, NULL);
  }
  double start = now_s();
  int unblocked = 0;
  volatile unsigned long n = 0;
  while(now_s() - start < seconds) {
    for(int i = 0; i < 100000; i++)
      n++;
    check_syscalls();
    if(!unblocked && !strcmp(mode, "block") && now_s() - start > seconds / 2) {
      sigset_t set;
      sigemptyset(&set);
      sigaddset(&set, SIGRTMIN + 7);
      pthread_sigmask(SIG_UNBLOCK, &set, NULL);
      unblocked = 1;
    }
    if(idx == 0 && !strcmp(mode, "exec") && now_s() - start > 1) {
      execl("/bin/true", "true", (char *)NULL);
      fail("execl");
    }
  }
  return NULL;
}

static void own_handler(int sig) { (void)sig; }

int main(int argc, char **argv)
{
  if(argc != 4) {
    fprintf(stderr, "usage: lifecycle_target MODE NTHREADS SECONDS\n");
    return 2;
  }
  mode = argv[1];
  nthreads = atoi(argv[2]);
  seconds = atof(argv[3]);

  if(!strcmp(mode, "abi")) {
    for(int i = 0; i < 100000; i++)
      check_syscalls();
    return failures ? 1 : 0;
  }

  pthread_t *threads = calloc(nthreads, sizeof(*threads));
  for(long i = 0; i < nthreads; i++)
    pthread_create(&threads[i], NULL, worker, (void *)i);

  if(!strcmp(mode, "leader-exit")) {
    sleep(1);
    pthread_exit(NULL); /* process exits (0) when the workers finish */
  }
  if(!strcmp(mode, "fork")) {
    sleep(1);
    pid_t p = fork();
    if(p == 0) {
      /* only this thread exists in the child */
      double start = now_s();
      while(now_s() - start < 1)
        check_syscalls();
      _exit(failures ? 1 : 0);
    }
    int st;
    if(waitpid(p, &st, 0) != p || !WIFEXITED(st) || WEXITSTATUS(st) != 0)
      fail("forked child");
  }
  if(!strcmp(mode, "fdexhaust")) {
    sleep(1);
    struct rlimit lim = {64, 64};
    setrlimit(RLIMIT_NOFILE, &lim);
    int fds[64], n = 0;
    while(n < 64 && (fds[n] = open("/dev/null", O_RDONLY)) >= 0)
      n++;
    if(n > 0)
      close(fds[--n]); /* leave exactly one free descriptor */
    sleep(3);
    while(n > 0)
      close(fds[--n]);
    for(int i = 0; i < nthreads; i++)
      pthread_join(threads[i], NULL);
    return failures ? 1 : 3;
  }
  if(!strcmp(mode, "handler")) {
    sleep(1);
    struct sigaction act;
    memset(&act, 0, sizeof(act));
    act.sa_handler = own_handler;
    sigaction(SIGRTMIN + 7, &act, NULL);
  }

  for(int i = 0; i < nthreads; i++)
    pthread_join(threads[i], NULL);
  if(failures)
    fprintf(stderr, "lifecycle_target: %d failures\n", failures);
  return failures ? 1 : 0;
}
