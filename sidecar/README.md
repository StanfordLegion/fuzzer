# Noise-Generating Shims/Sidecars

This directory contains noise-generating shims/sidecars for use with the fuzzer.
The goal is to generate more *interesting* thread interleavings more frequently,
because (even when the machine is oversubscribed) the default Linux scheduler
will only swap threads at the scheduler's quantum (around 3ms on typical
machines).

There are two variants:

 * `signal_shim`: an `LD_PRELOAD` library shim that interrupts threads via signals.
 * `ptrace_shim`: a sidecar that attaches and stops/starts threads via ptrace.

# Contents

| File | What it is |
|---|---|
| `signal_shim.c` / `.so` | **signal shim**: when loaded via `LD_PRELOAD`, interrupts random threads of the program it is loaded into and makes them sleep (see "Signal design overview"). Optionally delays thread wake-ups (`FUTEX_WAKE`). x86-64 Linux only. |
| `ptrace_shim.c` / `ptrace_shim` | **ptrace sidecar**: a separate process that launches the program and pauses its threads from outside with ptrace (see "Ptrace design overview"). |
| `lifecycle_target.c` | a test program to ensure the shims handle various situations safely, to avoid false positives. |
| `spinners.c` / `spinners` | microbenchmark: N threads spin reading `CLOCK_MONOTONIC` and log every gap in their own progress. |
| `analyze_pauses.py` | An analysis script that parses the `spinners` gap file (and optionally the shim pause log) and reports whether the thread was on a CPU when interrupted, how long it took to stop, and how long until it ran again. |
| `pause_sites.py` | A script that parses the logs generated from either shim and reports the user-code vs. system-call split, each library's share and its number of distinct pause sites, how much the two busiest threads take, and the top functions, optionally. |

# Signal shim

## Signal design overview

The shim is a shared library added to each rank with `LD_PRELOAD`.
Terminology: the **helper** is a thread the shim adds to the program; a
**target** is the program thread being paused.

1. **Start-up.** The library's constructor reads `SIGNAL_SHIM` (if set), installs a
   handler for the pause signal (`SIGRTMIN+7`), and starts the helper thread. It
   aborts if the configuration is invalid or the signal already has a handler.
   * If `rate=0` then no attempt is made to install the handler.
   * If any values specified in `SIGNAL_SHIM` are invalid, the shim calls `abort()`.
2. **Warmup.** The helper waits until the process has `warmup` live threads
   besides itself (Realm's processor threads appear only after network
   initialization, which must not be interrupted).
3. **Request loop.** Repeatedly, the helper:
   1. Sleeps for a random, exponentially distributed interval with mean
      1/`rate`.
   2. Lists the program's threads under `/proc/self/task` and keeps the
      candidates: by default threads that are runnable (state `R`) and not
      spinning (see `spinners`).
   3. Picks one candidate uniformly at random and sends it the pause signal
      with `rt_tgsigqueueinfo`. That call can only reach threads of this
      process (i.e., there is no risk of sending signals to other processes).
   4. Waits until the target has *entered* the handler (not until it
      finishes), or gives up after `abandon` ms.
4. **The pause.** The kernel delivers the signal at whatever instruction the
   target is executing, and the target itself runs the handler. The handler
   records the time and the interrupted instruction address, then calls
   `nanosleep(hold)`. That is a real blocking sleep, not a busy-wait: the
   target leaves the CPU, and another runnable thread can use it. When the
   sleep ends (and, on a busy node, when the scheduler next runs it), the
   target returns from the handler and continues exactly where it was
   interrupted. The helper never sleeps on a target's behalf.
5. **Pause slots.** Several pauses can be in flight at once, so the helper
   keeps a fixed table of 1024 request records ("slots"). Each holds one
   request's target, timestamps and hold time, plus one atomic word combining
   a generation number with a state: FREE -> SENT -> ENTERED -> DONE.
   * The signal carries the slot index and generation.
   * The handler may use a slot only if it can atomically change it from SENT
     to ENTERED with the generation it was given.
   * A signal that arrives late (for example, after being blocked) for a
     request the helper has already abandoned therefore does nothing.
   * A slot whose handler has started is never reused until that handler
     finishes.
   * The helper logs DONE slots and frees them.
6. **Futex delays (optional, `futex_prob`).** The shim also replaces
   `syscall()`. Realm wakes sleeping threads with
   `syscall(SYS_futex, ..., FUTEX_WAKE, ...)`. With probability `futex_prob`
   the waker is delayed before the wake-up (up to `futex_delay` us), and with
   the same probability it yields after it, so the woken thread can run
   first. Every other `syscall()` call goes straight to libc through an
   assembly trampoline that leaves the arguments untouched. Notes:
   * To delay futexes without also sending signals, set `rate=0`.
   * Futex delays do NOT obey the same warmup as signals.

7. **Exit.** If every program thread has exited (for example, `main` called
   `pthread_exit`), the helper calls `exit(0)`, as glibc would. At exit, if
   `rate > 0` and assuming the program does not exit abnormally (abort, killed,
   or `_exit`), counters are written to `DIR/summary-HOST-PID.txt` (if
   `log` is set; or if unset, just the summary is printed to stderr).

## Signal settings

To configure the shim, set the environment variable `SIGNAL_SHIM`. Arguments are
comma-separated key-value pairs (i.e., `"key=value,..."`).

| Key | Default | Meaning |
|---|---|---|
| `rate=HZ` | 1000 | Requested pauses per second per process. The achieved rate is lower, since each request also costs a `/proc` scan and waits until the target is interrupted. Set a value of `0` to disable pauses entirely. |
| `hold=US` | 0 | How long the target sleeps in the handler (`nanosleep`, a blocking sleep). Any value, even 0, makes the target leave the CPU (0 means a 1 ns request, which the kernel rounds up to its ~50 us timer slack). Max 10 s. |
| `maxhold=US` | — | Instead of a fixed `hold`, draw each pause's hold log-uniformly from \[1, US]. Setting both `hold` (non-zero) and `maxhold` is an error. |
| `mode=yield` | `sleep` | The target calls `sched_yield()` instead of sleeping. That only gives up the CPU if another thread is waiting for that same CPU. |
| `runnable=0` | 1 | Also target threads that are not runnable (blocked in a syscall). By default, only runnable threads are targeted, because interrupting a blocked syscall can make it fail with EINTR. Because checking runnable threads is not atomic with sending signals, there is still a small chance of sending a signal to a blocked thread even with `runnable=1`. |
| `spinners=PCT` | 70 | Skip threads that were runnable at least PCT% of their last 250 ms sampling window (from `/proc/.../schedstat`). Realm's background workers and the GASNet/ibverbs pollers busy-poll, so they are always runnable; without this filter they absorb most pauses (in measurements, >99% of stops landed in polling and runtime code rather than Legion). Pausing a poller probably mostly delays message handling (not measured). The filter steers pauses toward threads doing other work. Set a value of `0` to disable this check. |
| `futex_prob=P` | 0 (off) | Probability of delaying a `FUTEX_WAKE` (see step 6 above). |
| `futex_delay=US` | 1000 | Maximum futex delay (log-uniform). |
| `warmup=N` | 8 | No threads are paused until this number of live non-helper threads are active. A heuristic for "network initialization is done", not a readiness signal; waits indefinitely. |
| `abandon=MS` | 1000 | Give up on a request whose target has not entered the handler after this long (signal blocked, target stopped). A late handler then does nothing. |
| `log=DIR` | — | Save a per-pause log (`pauses-HOST-PID.log`: target, timestamps, interrupted address, syscall-or-not), a copy of `/proc/PID/maps` (as `maps-HOST-PID.log`), and counters at exit (`summary-HOST-PID.txt`: sent, done, abandoned, stale signals, send failures, ring full, handler replaced). If unspecified the summary is printed to stderr. |
| `seed=N` | — | RNG seed. If not specified, the seed is generated by obtaining 64 bits from `getrandom(2)` or by mixing the PID with a timestamp as a fallback. |

Limitations:
* The pause signal (`SIGRTMIN+7`) must be free. The shim aborts at start-up if
  a handler is already installed, and stops pausing (logging
  `handler_replaced`) if the program installs its own later.
* A child forked without exec runs with the shim disabled.
* Realm user-level threads are paused together with the kernel thread hosting
  them. There is no dedicated test with small user stacks or nested
  `SIGUSR1`.
* Futex interception only sees explicit `syscall(SYS_futex, FUTEX_WAKE...)`
  calls, not libc-internal ones.

# Ptrace sidecar

## Ptrace design overview

`ptrace_shim [options] -- program args...` runs the program as its child and
pauses the child's threads from outside, with the same target selection as the
signal shim. The sidecar fully detaches between pauses, to avoid interfering
with gdb and the like. Each pause:

1. Lists the child's threads in `/proc/PID/task` and picks a candidate
   (runnable, not spinning).
2. Attaches to that one thread (`PTRACE_SEIZE`) and asks the kernel to stop it
   (`PTRACE_INTERRUPT`). The thread stops at its current instruction, leaves the
   CPU, and stays off it while stopped.
3. Waits for the stop (at most `-T` ms), then checks that the stopped thread
   really belongs to the child. (After a thread exits, its thread ID is
   available to be reused by other processes, and we want to avoid attaching to
   other processes' threads. If we do by accident anyway we detach as soon as
   possible.)
4. Holds the thread stopped for the hold time (specified by either `-H` or `-M`,
   if set), then detaches (`PTRACE_DETACH`), which lets it run again. **Note:**
   because the sidecar waits for the thread, threads are attached to in
   sequence, not concurrently, in contrast to the signal shim.

If something else stops the thread before our interrupt does, ptrace reports a
different kind of stop, which the sidecar handles as follows (the latter two are
counted in the summary as `group_stop` and `aborted_signal`):
* **Interrupt stop:** this is the normal case. Hold, log, detach.
* **Job-control group-stop:** someone sent SIGSTOP (e.g., Ctrl-Z). Detach
  without resuming to preserve the requested stop.
* **Signal-delivery-stop:** an unrelated signal arrived while attached, so hand
  it back on detach to make sure the signal isn't lost.

A thread is never detached unless it is in a ptrace-stop. Threads that don't
stop by the deadline, or fail to detach, stay on a pending list and are detached
as soon as they do stop. If the sidecar is killed, the kernel detaches
everything. If a debugger is already attached, `PTRACE_SEIZE` fails and the
thread is skipped. A debugger trying to attach during one of the short
attachments can fail; there is no quiesce control yet.

`ptrace_shim` sets one of the following return codes:
* 1 if pipe or fork fail.
* 2 if argument parsing fails.
* 127 if the program cannot be started.
* `N` + 128 if the application is killed by a signal `N`.
* The application's return code, otherwise.

## Ptrace settings

| Option | Default | Meaning | Shim equivalent |
|---|---|---|---|
| `-r HZ` | 1000 | Requested pauses per second for the whole process, or `0` to disable pauses entirely | `rate` |
| `-H US` | 0 | Fixed hold: how long the target stays stopped | `hold` |
| `-M US` | — | Random log-uniform hold in \[1, US]; exclusive with `-H` | `maxhold` |
| `-a` | off | Also target non-runnable threads | `runnable=0` |
| `-x PCT` | 70 | Skip threads runnable >= PCT% of their last 250 ms window (same as signal shim), or `0` to disable the check | `spinners` |
| `-w N` | 8 | Warmup: wait until the program has N threads... | `warmup` |
| `-W SEC` | 10 | ...or until SEC seconds have passed, then start anyway | none (the shim waits indefinitely) |
| `-T MS` | 1000 | Stop deadline: a thread that hasn't stopped by then goes on the pending list | `abandon` (similar role) |
| `-s SEED` | random | RNG seed (same as the signal shim) | `seed` |
| `-l DIR` | — | Log directory (same format as signal shim) | `log` |

There is no futex-delay or yield mode in `ptrace_shim`.

# Testing

The `lifecycle_target` is a self-contained program that runs a variety of
scenarios to ensure that each of the shims are safe across a broad array of
situations. The scenarios include:

- `spin`: workers spin and repeatedly call `syscall(SYS_gettid)` and `syscall(SYS_futex, FUTEX_WAKE)`, checking each return value and `errno`. This catches a pause that corrupts registers or `errno`.
- `abi`: one thread, 100,000 rounds of `syscall()` checks:
  - `gettid` matches, with `errno` untouched.
  - An invalid number returns -1 / `ENOSYS`.
  - `FUTEX_WAKE_PRIVATE` succeeds with `errno` untouched, which tests the futex-flag masking.
  - `FUTEX_WAKE` on `NULL` returns -1 / `EFAULT`.
- `block`: workers block the pause signal for the first half, then unblock it, so signals are delivered late.
- `leader-exit`: the main thread calls `pthread_exit` after 1 s while workers keep running. The process must survive and exit 0.
- `exec`: a worker execs `/bin/true` after 1 s, replacing the process mid-run.
- `fork`: forks after 1 s; the child (with the shim disabled) runs the checks for 1 s, and the parent checks its exit status.
- `handler`: the program installs its own handler for the pause signal after 1 s.
- `fdexhaust`: after 1 s, uses up all file descriptors but one for 3 s while workers run, then exits with 3. A shim that misreads `/proc` as "all threads gone" would end it early with 0.
- Tests beyond the defined scenarios include: bad options, seeds, a held thread dying, the tracer being killed, and `SIGSTOP`/`SIGCONT`.

# Microbenchmark

`spinners NTHREADS SECONDS GAP_US OUTFILE [NCPUS]` runs a set of `NTHREADS`
threads that spin continuously for `SECONDS` seconds while tracking the time
elapsed since the last iteration (if the gap is larger than `GAP_US`). When
interrupted, this allows the benchmark to effectively measure the start time and
duration of any pauses (whether from our shims or normal OS preemption), which
the benchmark then logs at the end to `OUTFILE`. The benchmark can optionally be
pinned to the first `NCPUS` cores to avoid competition with the helper process
thread.

Notes:
* The benchmark must be run with `spinners=0` (signals) or `-x 0` (ptrace)
  because otherwise all threads will be detected as spinning, and no interrupts
  will be issued after 250 ms. (Within the first 250 ms the runnable statistics
  are still being computed, and thus count as zero.)
* Set `warmup` or `-w` to at most `NTHREADS+1` to ensure the warmup check
  completes successfully; otherwise the shim will not start pausing until the
  maximum warmup timer (ptrace `-W`) expires (or never if under the signal
  shim).

# Tools

Two tools are available:

`analyze_pauses.py GAP_LOG SECONDS NTHREADS [PAUSE_LOG]` reads the spinners gap
file `GAP_LOG` and, optionally, one pause log (`PAUSE_LOG`). A gap is any
stretch longer than `SECONDS` in which a spinning thread made no progress. The
pause log's timestamps mean the following:

|             | t0             | t1              | t2                      | t3              |
|-------------|----------------|-----------------|-------------------------|-----------------|
| ptrace      | attach started | stop observed   | hold ended              | detach returned |
| signal shim | signal sent    | handler entered | handler about to return | same as t2      |

The report includes, in order:
1. **Gaps (always shown):** gaps per second per thread, and gap length percentiles
   (p10/p50/p90/p99, µs). This counts every gap, from our pauses and from
   ordinary preemption alike. Without a pause log this is the only line.
2. **Pauses:** the number of pauses, the rate, and the share "matched to a gap". A
   pause matches when `t1` falls inside one of that thread's gaps.
   - A pause can fail to match if it was too short to register as a gap.
   - It also fails to match if it hit a thread that spinners doesn't monitor,
     such as the main thread.
3. **On CPU when interrupted:** the share of matched pauses whose gap began
   after t0, meaning the thread was running when we asked and the pause took it
   off the CPU. Stop latency is the time from t0 to the gap's start.
4. **Hold:** `t2−t1`. This differs between the tools:
   - For ptrace it is just the sidecar's hold.
   - For the signal shim it includes the time to get back onto a CPU after the
     sleep, because the thread itself measures it.
5. **Release to running again:** from `t3` to the gap's end.
   - For ptrace this is the time the thread waited to be scheduled after being
     released.
   - For the signal shim it is near zero, since that wait already appears under
     Hold.
6. **Resumed within 100 µs / waited more than 1 ms:** the short and long ends of
   line 5. A wait over 1 ms means the thread lost its CPU and had to wait for a
   turn.
7. **Total time off CPU per pause:** the full gap length. This is the one number
   directly comparable between the two tools.

`pause_sites.py [--functions N] DIR [DIR...]` analyzes the shim logs directory
and reports the user-code vs. system-call split, each library's share and its
number of distinct pause sites, and how much the two busiest threads take.
Optionally, the top `N` functions are also reported.
