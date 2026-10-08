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

/* Sidecar that interrupts random threads of a child process at arbitrary
 * user-code instructions using ptrace, without modifying the program.
 *
 *   ptrace_shim [options] -- program args...
 *
 * The program runs as a child and is NOT traced in between pauses. Each pause
 * attaches to a single thread, stops it, optionally holds it, and fully
 * detaches, so signals and debuggers behave normally the rest of the time:
 *
 *   PTRACE_SEIZE -> PTRACE_INTERRUPT -> waitpid -> [hold] -> PTRACE_DETACH
 *
 * A stopped thread is off the run queue; on an oversubscribed machine another
 * runnable thread takes its CPU, and when it is detached it has to compete to
 * get back on, so even a zero-length hold forces a context switch at the
 * point where it was interrupted.
 *
 * Options:
 *   -r HZ     pauses per second for the whole process (default 1000); 0 runs
 *             the program without pausing it
 *   -H US     fixed hold time in microseconds (default 0)
 *   -M US     random log-uniform hold in [1, US] microseconds (cannot be
 *             combined with -H)
 *   -a        consider all threads, not only runnable ones (state R)
 *   -x PCT    skip threads that were runnable >= PCT% of the last 250ms
 *             (spinning pollers); 0 disables (default 70)
 *   -w N      warmup: wait until the program has N threads (default 8)
 *   -W SEC    ... or until SEC seconds have passed (default 10)
 *   -s SEED   random seed, decimal or 0x hex (default: random, from
 *             getrandom); reported in the summary
 *   -T MS     give up waiting for a thread to stop after MS ms (default 1000);
 *             it stays attached on a pending list and is detached as soon as
 *             it does stop
 *   -l DIR    write a per-pause log, a /proc/PID/maps snapshot and a summary
 *             into DIR (without -l the summary goes to stderr)
 *
 * The program is started (exec'd) before warmup begins, so the warmup and the
 * maps snapshot always see it; if it cannot be executed, ptrace_shim exits
 * with status 127. Otherwise it exits with the program's status (128 + the
 * signal number if the program was killed).
 *
 * Numeric options are checked like signal_shim's settings: a malformed or
 * out-of-range value is an error, not silently 0.
 *
 * Stops are classified: our interrupt stop proceeds normally; a job-control
 * group-stop is detached without injecting anything, so the stop persists; a
 * signal-delivery-stop hands its signal back on detach. A thread is never
 * detached unless it is in a ptrace-stop. Threads still attached when the
 * program exits are detached by the kernel when this tracer exits.
 *
 * Pausing only runnable threads avoids interrupting blocked system calls:
 * although ptrace restarts most of them transparently, some drivers (e.g.
 * ibv_reg_mr) fail with EINTR whenever any signal-like wakeup is pending.
 */

#define _GNU_SOURCE
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ptrace.h>
#include <sys/random.h>
#include <sys/types.h>
#include <sys/user.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#define MAX_THREADS 4096
#define MAX_HOLD_US 10000000 /* 10 s, as in signal_shim */

static pid_t child = -1;
static int child_status = -1; /* wait status once the child has been reaped */

static uint64_t rng_state;

static struct {
  long pauses, stopped, in_syscall, aborted_signal, group_stop, seize_fail, vanished;
  long interrupt_fail, stop_timeout, getregs_fail, detach_fail, pending_released, foreign;
  double stop_latency_sum;
} stats;

static uint64_t rand64(void) {
  uint64_t z = (rng_state += 0x9e3779b97f4a7c15ULL);
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
  if (ns <= 0) return;
  struct timespec ts = {ns / 1000000000LL, ns % 1000000000LL};
  while (clock_nanosleep(CLOCK_MONOTONIC, 0, &ts, &ts) == EINTR) {
  }
}

static void forward_signal(int sig) {
  int saved = errno;
  if (child > 0) kill(child, sig);
  errno = saved;
}

/* Per-thread fraction of time runnable (on a CPU or waiting in the run queue)
 * over the last window, used to skip threads that spin (e.g. network
 * pollers): they are always runnable, so a uniform choice among runnable
 * threads would mostly pause them, and pausing a poller says little about the
 * interleaving of the rest of the program. Unlike CPU utilization, this stays
 * near 1 for a spinner even when the machine is oversubscribed. */
#define UTIL_WINDOW_NS 250000000LL
static struct {
  pid_t tid;
  int64_t start_ns;
  int64_t start_busy_ns;
  double util;
} util_table[65536];
static int last_spinners;

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

/* on-CPU plus run-queue wait time in ns, from /proc/PID/task/TID/schedstat */
static int64_t read_busy_ns(pid_t pid, pid_t tid) {
  char path[64], buf[128];
  snprintf(path, sizeof(path), "/proc/%d/task/%d/schedstat", pid, tid);
  int fd = open(path, O_RDONLY);
  if (fd < 0) return -1;
  ssize_t len = read(fd, buf, sizeof(buf) - 1);
  close(fd);
  if (len <= 0) return -1;
  buf[len] = 0;
  long long run = 0, wait = 0;
  if (sscanf(buf, "%lld %lld", &run, &wait) != 2) return -1;
  return run + wait;
}

/* Returns the number of candidate threads written to tids; *total gets the
 * number of threads in the process. Candidates must be runnable (unless
 * runnable_only is 0) and runnable less than max_util of the time (unless
 * max_util is 0). */
static int list_threads(pid_t pid, pid_t *tids, int max, int runnable_only,
                        double max_util, int *total) {
  char path[64];
  snprintf(path, sizeof(path), "/proc/%d/task", pid);
  DIR *d = opendir(path);
  *total = 0;
  if (!d) return 0;
  int n = 0, spinners = 0;
  int64_t now = now_ns();
  struct dirent *e;
  while ((e = readdir(d)) && n < max) {
    pid_t tid = atoi(e->d_name);
    if (tid <= 0) continue;
    (*total)++;
    if (runnable_only || max_util > 0) {
      char buf[1024];
      snprintf(path, sizeof(path), "/proc/%d/task/%d/stat", pid, tid);
      int fd = open(path, O_RDONLY);
      if (fd < 0) continue;
      ssize_t len = read(fd, buf, sizeof(buf) - 1);
      close(fd);
      if (len <= 0) continue;
      buf[len] = 0;
      char *p = strrchr(buf, ')');
      if (!p || p[1] != ' ') continue;
      char state = p[2];
      if (runnable_only && state != 'R') continue;
      if (max_util > 0) {
        int64_t busy = read_busy_ns(pid, tid);
        if (busy >= 0 && update_util(tid, busy, now) >= max_util) {
          spinners++;
          continue;
        }
      }
    }
    tids[n++] = tid;
  }
  closedir(d);
  last_spinners = spinners;
  return n;
}

/* Threads that are attached but have not (yet) reported a stop we could
 * detach from, e.g. because they did not stop before the -T deadline, and
 * threads that died while attached: a traced thread that dies stays a zombie
 * until its tracer reaps it, and until then its process cannot be reaped
 * either (e.g. when another thread calls exec, or the program exits, while
 * we hold a thread). */
#define MAX_PENDING 1024
static pid_t pending[MAX_PENDING];
static int num_pending;
static int64_t stop_timeout_ns = 1000000000LL;

static int is_pending(pid_t tid) {
  for (int i = 0; i < num_pending; i++)
    if (pending[i] == tid) return 1;
  return 0;
}

static void add_pending(pid_t tid) {
  if (!is_pending(tid) && num_pending < MAX_PENDING) pending[num_pending++] = tid;
}

/* A ptrace request on an attached thread failed: keep it pending, so that we
 * detach it once it stops or reap it once it has died (ESRCH). */
static void request_failed(pid_t tid, long *counter) {
  if (errno == ESRCH)
    stats.vanished++;
  else
    (*counter)++;
  add_pending(tid);
}

/* Detach from a tracee that is in a ptrace-stop described by status st. */
static void detach_from_stop(pid_t tid, int st) {
  int sig = 0;
  if ((unsigned)st >> 16 == PTRACE_EVENT_STOP) {
    /* interrupt stop (SIGTRAP) or group-stop (SIGSTOP etc.): inject nothing;
     * a group-stopped thread stays stopped after detach */
    if (WSTOPSIG(st) != SIGTRAP) stats.group_stop++;
  } else {
    /* signal-delivery-stop: hand the signal back */
    sig = WSTOPSIG(st);
    stats.aborted_signal++;
  }
  if (ptrace(PTRACE_DETACH, tid, 0, sig) == -1) request_failed(tid, &stats.detach_fail);
}

/* Poll pending tracees; detach any that have stopped, forget any that died. */
static void service_pending(void) {
  for (int i = 0; i < num_pending;) {
    pid_t tid = pending[i];
    int st;
    pid_t r = waitpid(tid, &st, WNOHANG | __WALL);
    int done = 0;
    if (r == -1) {
      done = (errno == ECHILD); /* no longer our tracee */
    } else if (r == tid) {
      if (WIFEXITED(st) || WIFSIGNALED(st)) {
        if (tid == child) child_status = st;
        done = 1;
      } else if (WIFSTOPPED(st)) {
        num_pending--;
        pending[i] = pending[num_pending];
        detach_from_stop(tid, st); /* may re-add on failure */
        stats.pending_released++;
        continue;
      }
    }
    if (done) {
      num_pending--;
      pending[i] = pending[num_pending];
    } else
      i++;
  }
}

static void reap_child_nonblocking(void) {
  if (child_status >= 0 || is_pending(child))
    return; /* a pending leader's statuses are handled by service_pending */
  int st;
  pid_t r = waitpid(child, &st, WNOHANG);
  if (r == child && (WIFEXITED(st) || WIFSIGNALED(st))) child_status = st;
}

/* Wait for a status change of tid until deadline. Returns 1 with *st set, 0
 * on timeout, -1 if tid is no longer our tracee. SIGCHLD is blocked in this
 * process, so sigtimedwait wakes us when a tracee stops or exits. */
static int wait_status(pid_t tid, int *st, int64_t deadline) {
  sigset_t chld;
  sigemptyset(&chld);
  sigaddset(&chld, SIGCHLD);
  for (;;) {
    pid_t r = waitpid(tid, st, WNOHANG | __WALL);
    if (r == tid) return 1;
    if (r == -1 && errno != EINTR) return -1;
    int64_t left = deadline - now_ns();
    if (left <= 0) return 0;
    struct timespec ts = {left / 1000000000LL, left % 1000000000LL};
    sigtimedwait(&chld, NULL, &ts);
  }
}

/* Does tid belong to our child's thread group? Only reliable while tid is
 * stopped under our ptrace (a stopped thread cannot exit, so its TID cannot
 * be recycled); before that, /proc is only advisory. */
static int belongs_to_child(pid_t tid) {
  char path[64], buf[4096];
  snprintf(path, sizeof(path), "/proc/%d/status", tid);
  int fd = open(path, O_RDONLY);
  if (fd < 0) return 0;
  ssize_t len = read(fd, buf, sizeof(buf) - 1);
  close(fd);
  if (len <= 0) return 0;
  buf[len] = 0;
  char *p = strstr(buf, "\nTgid:");
  return p && atoi(p + 6) == child;
}

static void pause_thread(pid_t tid, int64_t hold, FILE *log) {
  int64_t t0 = now_ns();
  stats.pauses++;
  if (ptrace(PTRACE_SEIZE, tid, 0, 0) == -1) {
    /* EPERM if e.g. gdb is attached, ESRCH if the thread exited */
    stats.seize_fail++;
    return;
  }
  if (ptrace(PTRACE_INTERRUPT, tid, 0, 0) == -1) {
    /* attached but not stopping: can only detach once it stops */
    request_failed(tid, &stats.interrupt_fail);
    return;
  }

  int st;
  int r = wait_status(tid, &st, t0 + stop_timeout_ns);
  if (r < 0) {
    stats.vanished++;
    return;
  }
  if (r == 0) {
    stats.stop_timeout++;
    add_pending(tid);
    return;
  }
  if (WIFEXITED(st) || WIFSIGNALED(st)) {
    if (tid == child) child_status = st;
    stats.vanished++;
    return;
  }
  if (!WIFSTOPPED(st)) {
    stats.stop_timeout++;
    add_pending(tid);
    return;
  }
  if (!((unsigned)st >> 16 == PTRACE_EVENT_STOP && WSTOPSIG(st) == SIGTRAP)) {
    /* a signal or a job-control stop got there first: release it unchanged
     * rather than hold anything up */
    detach_from_stop(tid, st);
    return;
  }

  /* The TID came from /proc and could have been recycled by another process
   * of the same user between the listing and PTRACE_SEIZE. Now that it is
   * stopped we can check; if it is not ours, let it go at once. (The brief
   * interrupt itself cannot be avoided without pidfds for threads, which
   * need kernel >= 6.9.) */
  if (!belongs_to_child(tid)) {
    stats.foreign++;
    if (ptrace(PTRACE_DETACH, tid, 0, 0) == -1) request_failed(tid, &stats.detach_fail);
    return;
  }

  int64_t t1 = now_ns();
  struct user_regs_struct regs;
  memset(&regs, 0, sizeof(regs));
  int regs_ok = (ptrace(PTRACE_GETREGS, tid, 0, &regs) == 0);
  if (!regs_ok) stats.getregs_fail++;
  sleep_ns(hold);
  int64_t t2 = now_ns();
  if (ptrace(PTRACE_DETACH, tid, 0, 0) == -1) {
    request_failed(tid, &stats.detach_fail);
    return;
  }
  int64_t t3 = now_ns();

  stats.stopped++;
  stats.stop_latency_sum += (t1 - t0) * 1e-3;
  if (!regs_ok) return;
  /* orig_rax is the syscall number (>= 0) if the thread was stopped in (or on
   * the way out of) a system call; if it was interrupted in user code by a
   * hardware interrupt (e.g. the IPI that delivers our stop, or the timer) it
   * holds the bitwise NOT of the interrupt vector, i.e. a negative value */
  if ((long long)regs.orig_rax >= 0) stats.in_syscall++;
  if (log)
    fprintf(log, "%d %lld %lld %lld %lld %llx %lld\n", tid, (long long)t0, (long long)t1,
            (long long)t2, (long long)t3, (unsigned long long)regs.rip,
            (long long)regs.orig_rax);
}

static void copy_maps(const char *dir) {
  char src[64], dst[4096], host[256] = "unknown", buf[65536];
  gethostname(host, sizeof(host));
  snprintf(src, sizeof(src), "/proc/%d/maps", child);
  snprintf(dst, sizeof(dst), "%s/maps-%s-%d.log", dir, host, child);
  int in = open(src, O_RDONLY);
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

static void usage(void) {
  fprintf(stderr,
          "usage: ptrace_shim [-r HZ] [-H US | -M US] [-a] [-x PCT] [-w N] [-W SEC] "
          "[-s SEED] [-T MS] [-l DIR] -- program args...\n");
  exit(2);
}

/* A number in [lo, hi]; anything else is a usage error. */
static double parse_num(int opt, const char *val, double lo, double hi) {
  char *end;
  double d = strtod(val, &end);
  if (end == val || *end || !isfinite(d) || d < lo || d > hi) {
    fprintf(stderr, "ptrace_shim: invalid -%c %s (must be in [%g, %g])\n", opt, val, lo,
            hi);
    exit(2);
  }
  return d;
}

/* Same seeding as signal_shim: the given seed, or 64 random bits. */
static uint64_t parse_seed(const char *val) {
  char *end;
  errno = 0;
  unsigned long long v = strtoull(val, &end, 0);
  if (end == val || *end || errno || *val == '-') {
    fprintf(stderr, "ptrace_shim: invalid -s %s\n", val);
    exit(2);
  }
  return v;
}

static uint64_t random_seed(void) {
  uint64_t seed;
  if (getrandom(&seed, sizeof(seed), 0) != (ssize_t)sizeof(seed))
    seed = (uint64_t)now_ns() ^ ((uint64_t)getpid() << 32);
  return seed;
}

int main(int argc, char **argv) {
  double rate = 1000;
  int64_t fixed_hold_us = 0, max_hold_us = 0;
  int runnable_only = 1, warmup_threads = 8;
  double max_util = 0.70;
  double warmup_s = 10;
  const char *log_dir = NULL;
  int have_seed = 0;

  int opt;
  while ((opt = getopt(argc, argv, "+r:H:M:ax:w:W:s:T:l:")) != -1) {
    switch (opt) {
      /* same ranges as the corresponding signal_shim settings */
      case 'r':
        rate = parse_num(opt, optarg, 0, 1e6);
        break;
      case 'H':
        fixed_hold_us = (int64_t)parse_num(opt, optarg, 0, MAX_HOLD_US);
        break;
      case 'M':
        max_hold_us = (int64_t)parse_num(opt, optarg, 1, MAX_HOLD_US);
        break;
      case 'a':
        runnable_only = 0;
        break;
      case 'x':
        max_util = parse_num(opt, optarg, 0, 100) / 100.0;
        break;
      case 'w':
        warmup_threads = (int)parse_num(opt, optarg, 0, 1e6);
        break;
      case 'W':
        warmup_s = parse_num(opt, optarg, 0, 1e6);
        break;
      case 's':
        rng_state = parse_seed(optarg);
        have_seed = 1;
        break;
      case 'T':
        stop_timeout_ns = (int64_t)(parse_num(opt, optarg, 1, 1e7) * 1e6);
        break;
      case 'l':
        log_dir = optarg;
        break;
      default:
        usage();
    }
  }
  if (optind < argc && !strcmp(argv[optind], "--")) optind++;
  if (optind >= argc) usage();
  if (fixed_hold_us > 0 && max_hold_us > 0) {
    fprintf(stderr, "ptrace_shim: -H and -M are mutually exclusive\n");
    usage();
  }
  if (!have_seed) rng_state = random_seed();
  const uint64_t seed = rng_state;

  /* The child reports on a close-on-exec pipe: a successful exec closes the
   * write end (we read EOF); a failed one writes errno. Waiting for this
   * means the warmup and the maps snapshot see the program, not a forked copy
   * of this sidecar. */
  int exec_pipe[2];
  if (pipe2(exec_pipe, O_CLOEXEC) != 0) {
    perror("ptrace_shim: pipe2");
    return 1;
  }
  child = fork();
  if (child < 0) {
    perror("ptrace_shim: fork");
    return 1;
  }
  if (child == 0) {
    close(exec_pipe[0]);
    execvp(argv[optind], argv + optind);
    int err = errno;
    ssize_t written = write(exec_pipe[1], &err, sizeof(err));
    (void)written; /* nothing more to do if even this fails */
    _exit(127);
  }
  close(exec_pipe[1]);
  int exec_errno;
  ssize_t got;
  do got = read(exec_pipe[0], &exec_errno, sizeof(exec_errno));
  while (got == -1 && errno == EINTR);
  close(exec_pipe[0]);
  if (got > 0) {
    fprintf(stderr, "ptrace_shim: cannot execute %s: %s\n", argv[optind],
            got == sizeof(exec_errno) ? strerror(exec_errno) : "unknown error");
    waitpid(child, NULL, 0);
    return 127;
  }
  if (got < 0) /* cannot tell; carry on as if the exec succeeded */
    perror("ptrace_shim: waiting for exec");

  sigset_t chld;
  sigemptyset(&chld);
  sigaddset(&chld, SIGCHLD);
  sigprocmask(SIG_BLOCK, &chld, NULL);

  struct sigaction act;
  memset(&act, 0, sizeof(act));
  act.sa_handler = forward_signal;
  act.sa_flags = SA_RESTART;
  sigemptyset(&act.sa_mask);
  int fwd[] = {SIGTERM, SIGINT, SIGHUP, SIGQUIT, SIGUSR1, SIGUSR2, SIGABRT};
  for (size_t i = 0; i < sizeof(fwd) / sizeof(fwd[0]); i++) sigaction(fwd[i], &act, NULL);

  FILE *pause_log = NULL;
  if (log_dir) {
    char path[4096], host[256] = "unknown";
    gethostname(host, sizeof(host));
    snprintf(path, sizeof(path), "%s/pauses-%s-%d.log", log_dir, host, child);
    pause_log = fopen(path, "w");
  }

  static pid_t tids[MAX_THREADS];
  int total = 0;
  int64_t deadline = now_ns() + (int64_t)(warmup_s * 1e9);
  while (child_status < 0) {
    list_threads(child, tids, MAX_THREADS, 0, 0, &total);
    if (total >= warmup_threads || now_ns() >= deadline) break;
    reap_child_nonblocking();
    sleep_ns(1000000);
  }
  if (log_dir && child_status < 0) copy_maps(log_dir);

  if (rate == 0) {
    /* no pauses: just wait for the program (forwarded signals restart this) */
    while (child_status < 0) {
      int st;
      pid_t r = waitpid(child, &st, 0);
      if (r == child && (WIFEXITED(st) || WIFSIGNALED(st)))
        child_status = st;
      else if (r == -1 && errno != EINTR) {
        perror("ptrace_shim: waitpid");
        return 1;
      }
    }
  }

  while (child_status < 0) {
    /* exponential inter-arrival times, in slices so that a low rate does
     * not delay noticing that the program has exited */
    int64_t until = now_ns() + (int64_t)(-log1p(-rand01()) / rate * 1e9);
    for (;;) {
      service_pending();
      reap_child_nonblocking();
      int64_t left = until - now_ns();
      if (child_status >= 0 || left <= 0) break;
      sleep_ns(left < 10000000 ? left : 10000000);
    }
    if (child_status >= 0) break;
    int n = list_threads(child, tids, MAX_THREADS, runnable_only, max_util, &total);
    if (n == 0) continue;
    int64_t hold = fixed_hold_us * 1000;
    if (max_hold_us > 1) {
      double us = exp(rand01() * log((double)max_hold_us));
      if (us > max_hold_us) us = max_hold_us;
      hold = (int64_t)(us * 1000);
    } else if (max_hold_us == 1)
      hold = 1000;
    pid_t tid = tids[rand64() % n];
    if (is_pending(tid)) continue;
    pause_thread(tid, hold, pause_log);
  }

  if (pause_log) fclose(pause_log);
  FILE *summary = stderr;
  if (log_dir) {
    char path[4096], host[256] = "unknown";
    gethostname(host, sizeof(host));
    snprintf(path, sizeof(path), "%s/summary-%s-%d.txt", log_dir, host, child);
    summary = fopen(path, "w");
    if (!summary) summary = stderr;
  }
  fprintf(summary,
          "ptrace_shim: pid=%d seed=0x%016llx pauses=%ld stopped=%ld in_syscall=%ld "
          "aborted_signal=%ld group_stop=%ld seize_fail=%ld vanished=%ld "
          "interrupt_fail=%ld stop_timeout=%ld getregs_fail=%ld detach_fail=%ld "
          "pending_released=%ld still_pending=%d foreign=%ld mean_stop_latency_us=%.1f "
          "spinners=%d\n",
          child, (unsigned long long)seed, stats.pauses, stats.stopped, stats.in_syscall,
          stats.aborted_signal, stats.group_stop, stats.seize_fail, stats.vanished,
          stats.interrupt_fail, stats.stop_timeout, stats.getregs_fail, stats.detach_fail,
          stats.pending_released, num_pending, stats.foreign,
          stats.stopped ? stats.stop_latency_sum / stats.stopped : 0.0, last_spinners);
  if (summary != stderr) fclose(summary);

  if (WIFEXITED(child_status)) return WEXITSTATUS(child_status);
  return 128 + WTERMSIG(child_status);
}
