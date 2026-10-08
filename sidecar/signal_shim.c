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

/* LD_PRELOAD shim that interrupts random threads of an unmodified program at
 * arbitrary user-code instructions and forces them off the CPU. This is the
 * in-process counterpart of ptrace_shim and uses the same target selection
 * and log format. x86-64 Linux only.
 *
 * Configured by SIGNAL_SHIM="key=value,...". Every key has a default, so
 * SIGNAL_SHIM may be left unset; loading the library is then enough.
 *
 *   rate=HZ      A helper thread requests about HZ pauses per second (default
 *                1000; 0 disables pausing, e.g. to use only futex delays; the
 *                achieved rate is lower: each request also costs a /proc scan
 *                and waits for the target to be interrupted). Requests
 *                overlap: the helper waits until the handler has started, not
 *                until it finishes.
 *   hold=US      The handler sleeps this long (default 0). Any sleep, even of
 *                0us, blocks the thread so its CPU goes to another runnable
 *                thread; an oversubscribed machine then makes it wait to get
 *                back on. (With hold=0 the sleep is 1ns plus timer slack.)
 *   maxhold=US   Random log-uniform hold in [1, US] instead of a fixed one
 *                (cannot be combined with hold).
 *   mode=yield   Call sched_yield() in the handler instead of sleeping. This
 *                only gives up the CPU if another thread is runnable on it.
 *   runnable=0   Also target threads that are not runnable (default: only
 *                runnable threads, which avoids most interruptions of blocked
 *                syscalls; some, e.g. GASNet's ibv_reg_mr, fail with EINTR).
 *   spinners=PCT Skip threads that were runnable >= PCT% of their last
 *                completed 250ms sampling window (e.g. network pollers, which
 *                would otherwise absorb most pauses). Default 70; 0 disables.
 *   futex_prob=P With probability P, delay around FUTEX_WAKE calls made
 *                through syscall(3) (this is how Realm's doorbells wake
 *                sleeping threads), i.e. right at a synchronization point.
 *                (Old name: futex=P.)
 *   futex_delay=US  Maximum futex delay in microseconds (default 1000).
 *                (Old name: pause=US.)
 *   warmup=N     Do not send signals until the process has N threads other
 *                than the helper (default 8): GASNet's ibv init treats EINTR
 *                as fatal, and Realm's processor threads only start after
 *                network init. This is a heuristic, not a readiness signal.
 *   abandon=MS   Give up on a request whose signal has not been handled after
 *                this long (default 1000), e.g. because the target blocks the
 *                signal or is stopped. A late handler for an abandoned
 *                request does nothing. Requests whose handler has started are
 *                never reclaimed.
 *   log=DIR      Write one line per completed pause to DIR/pauses-HOST-PID.log,
 *                a copy of /proc/PID/maps (maps-HOST-PID.log), and counters at
 *                exit (summary-HOST-PID.txt). Without log, the counters go to
 *                stderr, as with ptrace_shim.
 *   seed=N       RNG seed, decimal or 0x hex (default: random, from
 *                getrandom). The seed is reported in the summary; it does not
 *                replay a schedule, since timing still varies.
 *
 * Limitations: the pause signal (SIGRTMIN+7) must not be used by the program
 * (the shim refuses to start if a handler is already installed); a forked
 * child without exec runs with the shim disabled; futex interception only sees
 * explicit syscall(SYS_futex, FUTEX_WAKE...) calls, not libc-internal ones.
 *
 * Build: cc -O2 -shared -fPIC -Wl,-z,now -o signal_shim.so signal_shim.c -ldl -lpthread
 * -lm
 */

#define _GNU_SOURCE
#include <dirent.h>
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <linux/futex.h>
#include <math.h>
#include <pthread.h>
#include <sched.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/random.h>
#include <sys/syscall.h>
#include <sys/uio.h>
#include <time.h>
#include <ucontext.h>
#include <unistd.h>

#if !defined(__x86_64__) || !defined(__linux__)
#error "signal_shim supports x86-64 Linux only"
#endif

#define MAX_THREADS 4096
#define UTIL_WINDOW_NS 250000000LL
#define MAX_HOLD_US 10000000U /* 10 s */

static double cfg_rate = 1000;
static unsigned cfg_hold_us = 0, cfg_maxhold_us = 0;
static int cfg_yield = 0, cfg_runnable = 1, cfg_warmup = 8;
static double cfg_spinners = 0.70;
static double cfg_futex = 0;
static unsigned cfg_pause_us = 1000;
static int64_t cfg_abandon_ns = 1000000000LL;
static uint64_t cfg_seed;
static char cfg_log[1024];

static int pause_signal;
static pid_t helper_tid;
static uint64_t rng_counter;

typedef long (*syscall_fn)(long, ...);
/* libc's syscall(); resolved before use (see the trampoline below). Not
 * static: the assembly trampoline refers to it by name. */
__attribute__((visibility("hidden"))) syscall_fn shim_real_syscall;

/* Counters (updated atomically), reported at exit. */
static struct {
  long sent, done, abandoned, stale, send_failed, ring_full, handler_replaced;
} counts;

/* One slot per in-flight pause. The state word packs a generation number with
 * the slot state; the signal carries (generation, index), and the handler
 * claims the slot only by atomically moving it from SENT to ENTERED with a
 * matching generation. A late signal for an abandoned or reused request
 * therefore never touches the slot. Only the helper frees slots: DONE slots
 * after logging them, and SENT slots after the abandon timeout. ENTERED slots
 * are never reclaimed. */
#define NUM_SLOTS 1024
enum { SLOT_FREE = 0, SLOT_SENT = 1, SLOT_ENTERED = 2, SLOT_DONE = 3 };
#define WORD(gen, state) (((uint32_t)(gen) << 2) | (uint32_t)(state))
#define WORD_GEN(w) ((w) >> 2)
#define WORD_STATE(w) ((w) & 3)
static struct {
  uint32_t word; /* futex word */
  pid_t tid;
  int64_t t0, t1, t2;
  uint64_t rip;
  int in_syscall; /* 1, 0, or -1 if unknown */
  int64_t hold_ns;
} slots[NUM_SLOTS];

/* Stateless RNG: safe in signal handlers and needs no TLS. */
static uint64_t rand64(void) {
  uint64_t z = cfg_seed +
               __atomic_add_fetch(&rng_counter, 0x9e3779b97f4a7c15ULL, __ATOMIC_RELAXED);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

static double rand01(void) { return (rand64() >> 11) * (1.0 / 9007199254740992.0); }

static int64_t now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

static void sleep_ns(int64_t ns) {
  if (ns < 1) ns = 1;
  struct timespec ts = {ns / 1000000000LL, ns % 1000000000LL};
  while (nanosleep(&ts, &ts) == -1 && errno == EINTR) {
  }
}

/* log-uniform in [1, max_us] microseconds; not for use in the handler */
static unsigned random_us(unsigned max_us) {
  if (max_us <= 1) return 1;
  double us = exp(rand01() * log((double)max_us));
  if (us < 1) us = 1;
  if (us > max_us) us = max_us;
  return (unsigned)us;
}

/* Was the interrupted instruction a syscall (or a restart of one)? Reads the
 * instruction bytes with process_vm_readv, which reports EFAULT instead of
 * faulting if they are not readable. Heuristic: a pc next to a syscall
 * instruction does not prove the thread was blocked in it. */
static int classify_syscall(uint64_t rip) {
  unsigned char bytes[4];
  struct iovec local = {bytes, sizeof(bytes)};
  struct iovec remote = {(void *)(uintptr_t)(rip - 2), sizeof(bytes)};
  long n = shim_real_syscall(SYS_process_vm_readv, (long)getpid(), (long)&local, 1L,
                             (long)&remote, 1L, 0L);
  if (n != (long)sizeof(bytes)) return -1;
  /* syscall is 0f 05: just before rip, or at rip if rewound for restart */
  return ((bytes[0] == 0x0f && bytes[1] == 0x05) ||
          (bytes[2] == 0x0f && bytes[3] == 0x05))
             ? 1
             : 0;
}

static void pause_handler(int sig, siginfo_t *info, void *ctx) {
  (void)sig;
  int saved = errno;
  int64_t t1 = now_ns();
  if (info->si_code != SI_QUEUE || info->si_pid != getpid()) {
    errno = saved;
    return;
  }
  uint64_t v = (uint64_t)(uintptr_t)info->si_value.sival_ptr;
  unsigned i = v & 0xffff;
  uint32_t gen = (uint32_t)(v >> 16);
  if (i >= NUM_SLOTS) {
    errno = saved;
    return;
  }
  uint32_t expected = WORD(gen, SLOT_SENT);
  if (!__atomic_compare_exchange_n(&slots[i].word, &expected, WORD(gen, SLOT_ENTERED), 0,
                                   __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
    /* abandoned or reused request */
    __atomic_add_fetch(&counts.stale, 1, __ATOMIC_RELAXED);
    errno = saved;
    return;
  }
  /* we own the slot until we publish DONE */
  slots[i].t1 = t1;
  uint64_t rip = ((ucontext_t *)ctx)->uc_mcontext.gregs[REG_RIP];
  slots[i].rip = rip;
  slots[i].in_syscall = classify_syscall(rip);
  shim_real_syscall(SYS_futex, (long)&slots[i].word, (long)FUTEX_WAKE, 1L, 0L, 0L, 0L);
  if (cfg_yield)
    sched_yield();
  else
    sleep_ns(slots[i].hold_ns);
  slots[i].t2 = now_ns();
  __atomic_store_n(&slots[i].word, WORD(gen, SLOT_DONE), __ATOMIC_RELEASE);
  errno = saved;
}

/* Fraction of time each thread was runnable over its last completed sampling
 * window, from /proc/self/task/TID/schedstat (on-CPU plus run-queue wait
 * time). Unlike CPU utilization this stays near 1 for a spinner on an
 * oversubscribed machine. Windows are tumbling and only advance when a thread
 * is examined, so a long-idle thread's first new sample can cover a long
 * interval. */
static struct {
  pid_t tid;
  int64_t start_ns, start_busy_ns;
  double util;
} util_table[65536];

static double update_util(pid_t tid, int64_t busy_ns, int64_t now) {
  int i = tid & 0xffff;
  if (util_table[i].tid != tid) {
    util_table[i].tid = tid;
    util_table[i].start_ns = now;
    util_table[i].start_busy_ns = busy_ns;
    util_table[i].util = 0;
  } else if (now - util_table[i].start_ns >= UTIL_WINDOW_NS) {
    util_table[i].util =
        (double)(busy_ns - util_table[i].start_busy_ns) / (now - util_table[i].start_ns);
    util_table[i].start_ns = now;
    util_table[i].start_busy_ns = busy_ns;
  }
  return util_table[i].util;
}

static ssize_t read_file(const char *path, char *buf, size_t size) {
  int fd = open(path, O_RDONLY);
  if (fd < 0) return -1;
  ssize_t len = read(fd, buf, size - 1);
  close(fd);
  if (len >= 0) buf[len] = 0;
  return len;
}

/* Returns candidate threads; *total gets the number of live (non-zombie)
 * threads other than the helper, or -1 if the thread list could not be read. */
static int list_threads(pid_t *tids, int max, int *total) {
  DIR *d = opendir("/proc/self/task");
  *total = -1;
  if (!d) return 0;
  *total = 0;
  int n = 0;
  int64_t now = now_ns();
  struct dirent *e;
  while ((e = readdir(d)) && n < max) {
    pid_t tid = atoi(e->d_name);
    if (tid <= 0 || tid == helper_tid) continue;
    /* Everything read from /proc is advisory: the thread can change state or
     * exit right after we look. Only a positively read zombie/dead state
     * makes a thread not count as live (this count decides whether the
     * program has finished, see exit_if_alone); an unreadable entry (e.g.
     * EMFILE) counts as live but is not a candidate. */
    char path[64], buf[1024];
    snprintf(path, sizeof(path), "/proc/self/task/%d/stat", tid);
    char *p = NULL;
    if (read_file(path, buf, sizeof(buf)) > 0) {
      p = strrchr(buf, ')');
      if (p && (p[1] != ' ' || p[2] == 0)) p = NULL;
    }
    if (p && (p[2] == 'Z' || p[2] == 'X'))
      continue; /* exited (e.g. a main thread that called pthread_exit) */
    (*total)++;
    if (!p || (cfg_runnable && p[2] != 'R')) continue;
    if (cfg_spinners > 0) {
      long long run = 0, wait = 0;
      snprintf(path, sizeof(path), "/proc/self/task/%d/schedstat", tid);
      if (read_file(path, buf, sizeof(buf)) > 0 &&
          sscanf(buf, "%lld %lld", &run, &wait) == 2 &&
          update_util(tid, run + wait, now) >= cfg_spinners)
        continue;
    }
    tids[n++] = tid;
  }
  closedir(d);
  return n;
}

static void copy_maps(void) {
  char dst[1400], host[256] = "unknown", buf[65536];
  gethostname(host, sizeof(host));
  snprintf(dst, sizeof(dst), "%s/maps-%s-%d.log", cfg_log, host, getpid());
  int in = open("/proc/self/maps", O_RDONLY);
  if (in < 0) return;
  int out = open(dst, O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (out >= 0) {
    ssize_t n;
    while ((n = read(in, buf, sizeof(buf))) > 0)
      if (write(out, buf, n) != n) break;
    close(out);
  }
  close(in);
}

static FILE *pause_log;

/* Log and free finished requests; abandon requests never delivered. Only the
 * helper thread calls this. */
static void collect_slots(void) {
  int64_t now = now_ns();
  for (int i = 0; i < NUM_SLOTS; i++) {
    uint32_t w = __atomic_load_n(&slots[i].word, __ATOMIC_ACQUIRE);
    if (WORD_STATE(w) == SLOT_DONE) {
      if (pause_log)
        /* same columns as ptrace_shim: tid t0 t1 t2 t3 rip orig_rax, where
         * t3 (resume) is the handler exit and orig_rax is 0 for a syscall,
         * -1 for user code, and -2 if unknown */
        fprintf(pause_log, "%d %lld %lld %lld %lld %llx %d\n", slots[i].tid,
                (long long)slots[i].t0, (long long)slots[i].t1, (long long)slots[i].t2,
                (long long)slots[i].t2, (unsigned long long)slots[i].rip,
                slots[i].in_syscall == 1 ? 0 : (slots[i].in_syscall == 0 ? -1 : -2));
      __atomic_add_fetch(&counts.done, 1, __ATOMIC_RELAXED);
      __atomic_store_n(&slots[i].word, WORD(WORD_GEN(w), SLOT_FREE), __ATOMIC_RELEASE);
    } else if (WORD_STATE(w) == SLOT_SENT && now - slots[i].t0 >= cfg_abandon_ns) {
      uint32_t expected = w;
      if (__atomic_compare_exchange_n(&slots[i].word, &expected,
                                      WORD(WORD_GEN(w), SLOT_FREE), 0, __ATOMIC_ACQ_REL,
                                      __ATOMIC_ACQUIRE))
        __atomic_add_fetch(&counts.abandoned, 1, __ATOMIC_RELAXED);
    }
  }
  if (pause_log) fflush(pause_log);
}

/* Called with the result of a thread scan: if only the helper is left, the
 * program has finished (all its threads called pthread_exit), and glibc would
 * have called exit(0); do that rather than keep the process alive. */
static void exit_if_alone(int total, int *lonely) {
  if (total == 0) {
    if (++*lonely >= 2) exit(0);
  } else
    *lonely = 0;
}

static void *helper_main(void *arg) {
  (void)arg;
  helper_tid = shim_real_syscall(SYS_gettid);
  pid_t pid = getpid();
  static pid_t tids[MAX_THREADS];
  int total = 0;

  int alone = 0;
  for (;;) {
    list_threads(tids, MAX_THREADS, &total);
    if (total >= cfg_warmup) break;
    exit_if_alone(total, &alone);
    sleep_ns(1000000);
  }

  if (cfg_log[0]) {
    char path[1400], host[256] = "unknown";
    gethostname(host, sizeof(host));
    snprintf(path, sizeof(path), "%s/pauses-%s-%d.log", cfg_log, host, pid);
    pause_log = fopen(path, "w");
    copy_maps();
  }

  unsigned next = 0;
  int lonely = 0;
  for (;;) {
    collect_slots();
    /* stop if the program has taken over the signal */
    {
      struct sigaction cur;
      if (sigaction(pause_signal, NULL, &cur) != 0 || !(cur.sa_flags & SA_SIGINFO) ||
          cur.sa_sigaction != pause_handler) {
        __atomic_store_n(&counts.handler_replaced, 1, __ATOMIC_RELAXED);
        fprintf(stderr, "SIGNAL_SHIM: signal %d handler was replaced; pausing stopped\n",
                pause_signal);
        for (;;) {
          collect_slots();
          sleep_ns(100000000);
        }
      }
    }
    /* exponential inter-arrival times */
    sleep_ns((int64_t)(-log1p(-rand01()) / cfg_rate * 1e9));
    int n = list_threads(tids, MAX_THREADS, &total);
    exit_if_alone(total, &lonely);
    if (n == 0) continue;

    /* find a free slot */
    int i = -1;
    uint32_t w = 0;
    for (unsigned k = 0; k < NUM_SLOTS; k++) {
      unsigned j = (next + k) % NUM_SLOTS;
      w = __atomic_load_n(&slots[j].word, __ATOMIC_ACQUIRE);
      if (WORD_STATE(w) == SLOT_FREE) {
        i = j;
        break;
      }
    }
    if (i < 0) {
      __atomic_add_fetch(&counts.ring_full, 1, __ATOMIC_RELAXED);
      continue;
    }
    next = i + 1;
    uint32_t gen = (WORD_GEN(w) + 1) & 0x3fffffff;
    pid_t tid = tids[rand64() % n];
    unsigned us = cfg_maxhold_us ? random_us(cfg_maxhold_us) : cfg_hold_us;
    slots[i].tid = tid;
    slots[i].hold_ns = (int64_t)us * 1000;
    slots[i].t0 = now_ns();
    __atomic_store_n(&slots[i].word, WORD(gen, SLOT_SENT), __ATOMIC_RELEASE);

    siginfo_t si;
    memset(&si, 0, sizeof(si));
    si.si_signo = pause_signal;
    si.si_code = SI_QUEUE;
    si.si_pid = pid;
    si.si_uid = getuid();
    si.si_value.sival_ptr = (void *)(uintptr_t)(((uint64_t)gen << 16) | (unsigned)i);
    if (shim_real_syscall(SYS_rt_tgsigqueueinfo, (long)pid, (long)tid, (long)pause_signal,
                          (long)&si) != 0) {
      /* thread gone (ESRCH) or signal queue full (EAGAIN): no handler will
       * run for this generation, so the slot can be freed */
      __atomic_add_fetch(&counts.send_failed, 1, __ATOMIC_RELAXED);
      __atomic_store_n(&slots[i].word, WORD(gen, SLOT_FREE), __ATOMIC_RELEASE);
      continue;
    }
    __atomic_add_fetch(&counts.sent, 1, __ATOMIC_RELAXED);
    /* wait until the target has been interrupted (or the abandon timeout,
     * after which collect_slots gives up on it) */
    int64_t deadline = slots[i].t0 + cfg_abandon_ns;
    for (;;) {
      uint32_t cur = __atomic_load_n(&slots[i].word, __ATOMIC_ACQUIRE);
      if (cur != WORD(gen, SLOT_SENT)) break;
      int64_t left = deadline - now_ns();
      if (left <= 0) break;
      if (left > 10000000) left = 10000000;
      struct timespec timeout = {0, left};
      shim_real_syscall(SYS_futex, (long)&slots[i].word, (long)FUTEX_WAIT, (long)cur,
                        (long)&timeout, 0L, 0L);
    }
  }
  return NULL;
}

/* Interposed syscall(3), as an assembly trampoline so that every call except
 * FUTEX_WAKE is forwarded to libc with its argument registers untouched
 * (forwarding a C variadic call would read arguments the caller never
 * passed). FUTEX_WAKE takes only (uaddr, op, val), which the hook forwards. */
__attribute__((visibility("hidden"))) long shim_futex_wake_hook(long number, long uaddr,
                                                                long op, long val);
__attribute__((visibility("hidden"))) void shim_resolve_real_syscall(void);

_Static_assert(SYS_futex == 202, "trampoline assumes SYS_futex == 202");
_Static_assert(FUTEX_WAKE == 1, "trampoline assumes FUTEX_WAKE == 1");
_Static_assert(FUTEX_CMD_MASK == -385, "trampoline assumes FUTEX_CMD_MASK == ~0x180");

__asm__(
    ".pushsection .text\n"
    ".globl syscall\n"
    ".type syscall, @function\n"
    "syscall:\n"
    "  cmpq $202, %rdi\n" /* SYS_futex */
    "  jne 1f\n"
    "  movl %edx, %eax\n"
    "  andl $-385, %eax\n" /* FUTEX_CMD_MASK = ~(PRIVATE | CLOCK_REALTIME) */
    "  cmpl $1, %eax\n"    /* FUTEX_WAKE */
    "  jne 1f\n"
    "  jmp shim_futex_wake_hook\n"
    "1:\n"
    "  cmpq $0, shim_real_syscall(%rip)\n"
    "  jne 2f\n"
    /* not resolved yet (called before our constructor): resolve while
     * preserving the argument registers; 6 pushes + 8 keeps alignment */
    "  pushq %rdi\n  pushq %rsi\n  pushq %rdx\n  pushq %rcx\n  pushq %r8\n  pushq %r9\n"
    "  subq $8, %rsp\n"
    "  call shim_resolve_real_syscall\n"
    "  addq $8, %rsp\n"
    "  popq %r9\n  popq %r8\n  popq %rcx\n  popq %rdx\n  popq %rsi\n  popq %rdi\n"
    "2:\n"
    "  jmp *shim_real_syscall(%rip)\n"
    ".size syscall, .-syscall\n"
    ".popsection\n");

void shim_resolve_real_syscall(void) {
  if (!shim_real_syscall) shim_real_syscall = (syscall_fn)dlsym(RTLD_NEXT, "syscall");
  if (!shim_real_syscall) abort();
}

long shim_futex_wake_hook(long number, long uaddr, long op, long val) {
  int saved = errno;
  if (!shim_real_syscall) shim_resolve_real_syscall();
  if (cfg_futex > 0 && rand01() < cfg_futex)
    sleep_ns((int64_t)random_us(cfg_pause_us) * 1000); /* delay the waker */
  errno = saved;
  long ret = shim_real_syscall(number, uaddr, op, val);
  int after = errno;
  if (cfg_futex > 0 && rand01() < cfg_futex)
    sched_yield(); /* let the wakee run ahead of the waker */
  errno = after;
  return ret;
}

static void disable_in_child(void) {
  /* the helper thread does not survive fork; keep the child unperturbed */
  cfg_rate = 0;
  cfg_futex = 0;
  helper_tid = 0;
}

static void report_summary(void) {
  if (cfg_rate <= 0 || helper_tid == 0) return;
  char line[512];
  snprintf(line, sizeof(line),
           "signal_shim: pid=%d seed=0x%016llx sent=%ld done=%ld abandoned=%ld "
           "stale_signals=%ld send_failed=%ld ring_full=%ld handler_replaced=%ld\n",
           getpid(), (unsigned long long)cfg_seed,
           __atomic_load_n(&counts.sent, __ATOMIC_RELAXED),
           __atomic_load_n(&counts.done, __ATOMIC_RELAXED),
           __atomic_load_n(&counts.abandoned, __ATOMIC_RELAXED),
           __atomic_load_n(&counts.stale, __ATOMIC_RELAXED),
           __atomic_load_n(&counts.send_failed, __ATOMIC_RELAXED),
           __atomic_load_n(&counts.ring_full, __ATOMIC_RELAXED),
           __atomic_load_n(&counts.handler_replaced, __ATOMIC_RELAXED));
  FILE *f = NULL;
  if (cfg_log[0]) {
    char path[1400], host[256] = "unknown";
    gethostname(host, sizeof(host));
    snprintf(path, sizeof(path), "%s/summary-%s-%d.txt", cfg_log, host, getpid());
    f = fopen(path, "w");
  }
  if (f) {
    fputs(line, f);
    fclose(f);
  } else
    fputs(line, stderr); /* no log directory (or it is unwritable) */
}

/* Same seeding as ptrace_shim: the given seed, or 64 random bits. */
static int parse_seed(const char *val, uint64_t *out) {
  char *end;
  errno = 0;
  unsigned long long v = strtoull(val, &end, 0);
  if (end == val || *end || errno || *val == '-') {
    fprintf(stderr, "SIGNAL_SHIM: invalid seed=%s\n", val);
    return 0;
  }
  *out = v;
  return 1;
}

static uint64_t random_seed(void) {
  uint64_t seed;
  if (getrandom(&seed, sizeof(seed), 0) != (ssize_t)sizeof(seed))
    seed = (uint64_t)now_ns() ^ ((uint64_t)getpid() << 32);
  return seed;
}

static int parse_double(const char *key, const char *val, double lo, double hi,
                        double *out) {
  char *end;
  double d = strtod(val, &end);
  if (end == val || *end || !isfinite(d) || d < lo || d > hi) {
    fprintf(stderr, "SIGNAL_SHIM: invalid %s=%s (must be in [%g, %g])\n", key, val, lo,
            hi);
    return 0;
  }
  *out = d;
  return 1;
}

__attribute__((constructor)) static void signal_shim_init(void) {
  shim_resolve_real_syscall();
  const char *env = getenv("SIGNAL_SHIM");
  if (!env) env = ""; /* all defaults */
  char *copy = strdup(env), *saveptr = NULL;
  int ok = 1, have_seed = 0;
  for (char *tok = strtok_r(copy, ",", &saveptr); tok;
       tok = strtok_r(NULL, ",", &saveptr)) {
    char *eq = strchr(tok, '=');
    if (!eq) {
      fprintf(stderr, "SIGNAL_SHIM: malformed entry '%s'\n", tok);
      ok = 0;
      continue;
    }
    *eq = 0;
    const char *val = eq + 1;
    double d = 0;
    if (!strcmp(tok, "rate"))
      ok &= parse_double(tok, val, 0, 1e6, &cfg_rate);
    else if (!strcmp(tok, "hold")) {
      if ((ok &= parse_double(tok, val, 0, MAX_HOLD_US, &d))) cfg_hold_us = (unsigned)d;
    } else if (!strcmp(tok, "maxhold")) {
      if ((ok &= parse_double(tok, val, 1, MAX_HOLD_US, &d)))
        cfg_maxhold_us = (unsigned)d;
    } else if (!strcmp(tok, "mode")) {
      if (!strcmp(val, "yield"))
        cfg_yield = 1;
      else if (!strcmp(val, "sleep"))
        cfg_yield = 0;
      else {
        fprintf(stderr, "SIGNAL_SHIM: invalid mode=%s\n", val);
        ok = 0;
      }
    } else if (!strcmp(tok, "runnable")) {
      if ((ok &= parse_double(tok, val, 0, 1, &d))) cfg_runnable = (int)d;
    } else if (!strcmp(tok, "spinners")) {
      if ((ok &= parse_double(tok, val, 0, 100, &d))) cfg_spinners = d / 100.0;
    } else if (!strcmp(tok, "futex_prob") || !strcmp(tok, "futex"))
      ok &= parse_double(tok, val, 0, 1, &cfg_futex);
    else if (!strcmp(tok, "futex_delay") || !strcmp(tok, "pause")) {
      if ((ok &= parse_double(tok, val, 1, MAX_HOLD_US, &d))) cfg_pause_us = (unsigned)d;
    } else if (!strcmp(tok, "warmup")) {
      if ((ok &= parse_double(tok, val, 0, 1e6, &d))) cfg_warmup = (int)d;
    } else if (!strcmp(tok, "abandon")) {
      if ((ok &= parse_double(tok, val, 1, 1e7, &d))) cfg_abandon_ns = (int64_t)(d * 1e6);
    } else if (!strcmp(tok, "log"))
      snprintf(cfg_log, sizeof(cfg_log), "%s", val);
    else if (!strcmp(tok, "seed"))
      ok &= (have_seed = parse_seed(val, &cfg_seed));
    else {
      fprintf(stderr, "SIGNAL_SHIM: unknown key '%s'\n", tok);
      ok = 0;
    }
  }
  free(copy);
  if (cfg_hold_us > 0 && cfg_maxhold_us > 0) {
    fprintf(stderr, "SIGNAL_SHIM: hold and maxhold are mutually exclusive\n");
    ok = 0;
  }
  if (!ok) {
    /* refuse to run a misconfigured experiment silently */
    fprintf(stderr, "SIGNAL_SHIM: invalid configuration, aborting\n");
    abort();
  }
  if (!have_seed) cfg_seed = random_seed();
  pthread_atfork(NULL, NULL, disable_in_child);
  atexit(report_summary);

  if (cfg_rate > 0) {
    pause_signal = SIGRTMIN + 7;
    if (pause_signal > SIGRTMAX) {
      fprintf(stderr, "SIGNAL_SHIM: SIGRTMIN+7 exceeds SIGRTMAX\n");
      abort();
    }
    struct sigaction old;
    if (sigaction(pause_signal, NULL, &old) != 0 ||
        ((old.sa_flags & SA_SIGINFO) ? (old.sa_sigaction != NULL)
                                     : (old.sa_handler != SIG_DFL))) {
      fprintf(stderr, "SIGNAL_SHIM: signal %d is already in use\n", pause_signal);
      abort();
    }
    struct sigaction act;
    memset(&act, 0, sizeof(act));
    act.sa_sigaction = pause_handler;
    act.sa_flags = SA_RESTART | SA_SIGINFO;
    sigemptyset(&act.sa_mask);
    if (sigaction(pause_signal, &act, NULL) != 0) {
      perror("SIGNAL_SHIM: sigaction");
      abort();
    }

    pthread_t helper;
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
    int err = pthread_create(&helper, &attr, helper_main, NULL);
    pthread_attr_destroy(&attr);
    if (err) {
      fprintf(stderr, "SIGNAL_SHIM: pthread_create failed: %s\n", strerror(err));
      abort();
    }
  }
}
