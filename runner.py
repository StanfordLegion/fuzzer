#!/usr/bin/env python3

# Copyright 2024 Stanford University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import argparse, glob, multiprocessing, os, queue, shlex, shutil, subprocess, tempfile


def run_spy(tid, tnum, spy, logfiles, verbose):
    cmd = [
        "pypy3",
        spy,
        "--logical",
        "--physical",
        "--cycle",
        # "--sanity", # FIXME: This breaks on several test cases.
        "--leaks",
        # "--geometry", # FIXME: This is *very* slow.
        "--assert-error",
        "--assert-warning",
    ] + logfiles
    if verbose >= 3:
        print(f"[{tid} of {tnum}]: Running {shlex.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode == 0:
        return
    return proc


def run_fuzzer(tid, tnum, seed, num_ops, fuzzer_exe, spy, verbose, skip=None):
    cmd = [fuzzer_exe, "-fuzz:seed", str(seed), "-fuzz:ops", str(num_ops)]
    if skip is not None:
        cmd.extend(["-fuzz:skip", str(skip)])
    if spy:
        log_dir = tempfile.mkdtemp(dir=os.getcwd())
        cmd.extend(
            ["-level", "legion_spy=2", "-logfile", os.path.join(log_dir, "spy_%.log")]
        )
    else:
        cmd.extend(["-level", "4"])
    if verbose >= 3:
        print(f"[{tid} of {tnum}]: Running {shlex.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode == 0:
        spy_proc = None
        if spy:
            spy_logs = glob.glob(os.path.join(log_dir, "spy_*.log"))
            spy_proc = run_spy(tid, tnum, spy, spy_logs, verbose)
            shutil.rmtree(log_dir)
        if spy_proc is None:
            return
        return (proc, spy_proc)
    if spy:
        shutil.rmtree(log_dir)
    return (proc, None)


def bisect_start(tid, tnum, seed, num_ops, fuzzer_exe, spy, verbose):
    if verbose >= 2:
        print(f"[{tid} of {tnum}]: Bisecting {num_ops} ops at seed {seed}")
    good = num_ops
    bad = 0
    last_failure = None
    while bad + 1 < good:
        check = (good + bad) // 2
        if verbose >= 2:
            print(
                f"[{tid} of {tnum}]: Testing {num_ops} ops (skipping {check}) at seed {seed}"
            )
        proc = run_fuzzer(
            tid, tnum, seed, num_ops, fuzzer_exe, spy, verbose, skip=check
        )
        if proc is None:
            good = check
        else:
            bad = check
            last_failure = proc
    return bad, last_failure


def bisect_stop(tid, tnum, seed, num_ops, fuzzer_exe, spy, verbose):
    if verbose >= 2:
        print(f"[{tid} of {tnum}]: Bisecting {num_ops} ops at seed {seed}")
    good = 0
    bad = num_ops
    last_failure = None
    while good + 1 < bad:
        check = (good + bad) // 2
        if verbose >= 2:
            print(f"[{tid} of {tnum}]: Testing {check} ops at seed {seed}")
        proc = run_fuzzer(tid, tnum, seed, check, fuzzer_exe, spy, verbose)
        if proc is None:
            good = check
        else:
            bad = check
            last_failure = proc
    return bad, last_failure


def fuzz(tid, tnum, seed, num_ops, fuzzer_exe, spy, verbose):
    if verbose >= 2:
        print(f"[{tid} of {tnum}]: Testing {num_ops} ops at seed {seed}")
    proc = run_fuzzer(tid, tnum, seed, num_ops, fuzzer_exe, spy, verbose)
    if proc is None:
        return
    if verbose >= 1:
        print(f"[{tid} of {tnum}]: Found failure: {shlex.join(proc[0].args)}")
    stop, stop_proc = bisect_stop(tid, tnum, seed, num_ops, fuzzer_exe, spy, verbose)
    start, start_proc = bisect_start(tid, tnum, seed, stop, fuzzer_exe, spy, verbose)
    proc = start_proc or stop_proc or proc
    if verbose >= 1:
        print(f"[{tid} of {tnum}]: Shortest failure: {shlex.join(proc[0].args)}")
    return proc


def run_test(
    tid,
    tnum,
    num_tests,
    num_ops,
    base_seed,
    fuzzer_exe,
    spy,
    verbose,
):
    if verbose >= 1:
        print(f"[{tid} of {tnum}]: Starting search")
    initial_seed = base_seed + tid
    failed_procs = []
    for test_index in range(num_tests):
        seed = initial_seed + test_index * tnum
        proc = fuzz(tid, tnum, seed, num_ops, fuzzer_exe, spy, verbose)
        if proc:
            failed_procs.append(proc)
    return failed_procs


def report_failure(proc):
    fuzz_proc, spy_proc = proc
    print(f"Captured failure: {shlex.join(fuzz_proc.args)}")
    if spy_proc is not None:
        print(spy_proc.stderr.decode("utf-8"))
    else:
        print(fuzz_proc.stderr.decode("utf-8"))
    print()


def run_tests(thread_count, num_tests, num_ops, base_seed, fuzzer_exe, spy, verbose):
    if thread_count is None:
        thread_count = os.cpu_count()

    thread_pool = multiprocessing.Pool(thread_count)

    result_queue = queue.Queue()
    num_queued = 0
    for tid in range(thread_count):
        num_queued += 1

        def callback(r):
            result_queue.put(r)

        def error_callback(e):
            print("ERROR CALLBACK", e)
            raise e

        thread_pool.apply_async(
            run_test,
            (
                tid,
                thread_count,
                num_tests,
                num_ops,
                base_seed,
                fuzzer_exe,
                spy,
                verbose,
            ),
            callback=callback,
            error_callback=error_callback,
        )

    thread_pool.close()

    num_remaining = num_queued
    try:
        while num_remaining > 0:
            failed_procs = result_queue.get()
            for proc in failed_procs:
                report_failure(proc)
            num_remaining -= 1
        thread_pool.join()
    except KeyboardInterrupt:
        thread_pool.terminate()
        raise


def driver():
    parser = argparse.ArgumentParser(description="Fuzzer test harness")
    parser.add_argument(
        "-j",
        nargs="?",
        type=int,
        help="number threads used to test",
        dest="thread_count",
    )
    parser.add_argument(
        "-n",
        "--tests",
        type=int,
        default=10,
        dest="num_tests",
        help="number of tests to run per seed",
    )
    parser.add_argument(
        "-o",
        "--ops",
        type=int,
        default=100,
        dest="num_ops",
        help="number of operations to run per test",
    )
    parser.add_argument(
        "-s", "--seed", type=int, default=0, dest="base_seed", help="base seed to use"
    )
    parser.add_argument(
        "--fuzzer",
        required=True,
        dest="fuzzer_exe",
        help="location of fuzzer executable",
    )
    parser.add_argument("--spy", help="location of Legion Spy script (optional)")
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        dest="verbose",
        help="enable verbose output",
    )
    args = parser.parse_args()
    run_tests(**vars(args))


if __name__ == "__main__":
    driver()
