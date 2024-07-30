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

import argparse, glob, multiprocessing, os, queue, shlex, shutil, subprocess, sys, tempfile


def prefix(tid, tnum):
    digits = len(str(tnum))
    return f"[{tid:{digits}} of {tnum}]:"


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
        print(f"{prefix(tid, tnum)} Running {shlex.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode == 0:
        return
    return proc


def run_fuzzer(
    tid, tnum, seed, num_ops, extra_args, fuzzer, launcher, spy, verbose, skip=None
):
    cmd = []
    if launcher:
        cmd.extend(shlex.split(launcher))
    cmd.extend([fuzzer, "-fuzz:seed", str(seed), "-fuzz:ops", str(num_ops)])
    if skip is not None:
        cmd.extend(["-fuzz:skip", str(skip)])
    cmd.extend(arg for extra in extra_args for arg in shlex.split(extra))
    if spy:
        log_dir = tempfile.mkdtemp(dir=os.getcwd())
        cmd.extend(
            ["-level", "legion_spy=2", "-logfile", os.path.join(log_dir, "spy_%.log")]
        )
    else:
        cmd.extend(["-level", "4"])
    if verbose >= 3:
        print(f"{prefix(tid, tnum)} Running {shlex.join(cmd)}", flush=True)
    try:
        proc = subprocess.run(cmd, capture_output=True)
        if proc.returncode == 0:
            spy_proc = None
            if spy:
                spy_logs = glob.glob(os.path.join(log_dir, "spy_*.log"))
                spy_proc = run_spy(tid, tnum, spy, spy_logs, verbose)
            if spy_proc is None:
                return
            return (proc, spy_proc)
        return (proc, None)
    finally:
        if spy:
            shutil.rmtree(log_dir)


def bisect_start(tid, tnum, seed, num_ops, extra_args, fuzzer, launcher, spy, verbose):
    if verbose >= 2:
        print(f"{prefix(tid, tnum)} Bisecting {num_ops} ops at seed {seed}", flush=True)
    good = num_ops
    bad = 0
    last_failure = None
    while bad + 1 < good:
        check = (good + bad) // 2
        if verbose >= 2:
            print(
                f"{prefix(tid, tnum)} Testing {num_ops} ops (skipping {check}) at seed {seed}",
                flush=True,
            )
        proc = run_fuzzer(
            tid,
            tnum,
            seed,
            num_ops,
            extra_args,
            fuzzer,
            launcher,
            spy,
            verbose,
            skip=check,
        )
        if proc is None:
            good = check
        else:
            bad = check
            last_failure = proc
    return bad, last_failure


def bisect_stop(tid, tnum, seed, num_ops, extra_args, fuzzer, launcher, spy, verbose):
    if verbose >= 2:
        print(f"{prefix(tid, tnum)} Bisecting {num_ops} ops at seed {seed}", flush=True)
    good = 0
    bad = num_ops
    last_failure = None
    while good + 1 < bad:
        check = (good + bad) // 2
        if verbose >= 2:
            print(f"{prefix(tid, tnum)} Testing {check} ops at seed {seed}", flush=True)
        proc = run_fuzzer(
            tid, tnum, seed, check, extra_args, fuzzer, launcher, spy, verbose
        )
        if proc is None:
            good = check
        else:
            bad = check
            last_failure = proc
    return bad, last_failure


def fuzz(tid, tnum, seed, num_ops, extra_args, fuzzer, launcher, spy, verbose):
    if verbose >= 2:
        print(f"{prefix(tid, tnum)} Testing {num_ops} ops at seed {seed}", flush=True)
    proc = run_fuzzer(
        tid, tnum, seed, num_ops, extra_args, fuzzer, launcher, spy, verbose
    )
    if proc is None:
        return
    if verbose >= 1:
        print(
            f"{prefix(tid, tnum)} Found failure: {shlex.join(proc[0].args)}", flush=True
        )
    stop, stop_proc = bisect_stop(
        tid, tnum, seed, num_ops, extra_args, fuzzer, launcher, spy, verbose
    )
    start, start_proc = bisect_start(
        tid, tnum, seed, stop, extra_args, fuzzer, launcher, spy, verbose
    )
    proc = start_proc or stop_proc or proc
    if verbose >= 1:
        print(
            f"{prefix(tid, tnum)} Shortest failure: {shlex.join(proc[0].args)}",
            flush=True,
        )
    return proc


def report_failure(proc):
    fuzz_proc, spy_proc = proc
    print(f"Captured failure: {shlex.join(fuzz_proc.args)}")
    if spy_proc is not None:
        # Legion Spy usually spews a lot, so we only want to capture the errors.
        lines = spy_proc.stdout.decode("utf-8").splitlines()
        for line in lines:
            if ("ERROR" in line) or ("WARNING" in line):
                print(line)
        print(spy_proc.stderr.decode("utf-8"))
    else:
        if fuzz_proc.stderr:
            print(fuzz_proc.stderr.decode("utf-8"))
        else:
            print(fuzz_proc.stdout.decode("utf-8"))
    print(flush=True)


def run_tests(
    thread_count,
    num_tests,
    num_ops,
    base_seed,
    extra_args,
    fuzzer,
    launcher,
    spy,
    verbose,
):
    thread_pool = multiprocessing.Pool(thread_count)

    result_queue = queue.Queue()
    num_queued = 0
    for tid in range(num_tests):
        num_queued += 1

        seed = base_seed + tid

        def callback(r):
            result_queue.put(r)

        def error_callback(e):
            print("ERROR CALLBACK", e)
            raise e

        thread_pool.apply_async(
            fuzz,
            (
                tid + 1,
                num_tests,
                seed,
                num_ops,
                extra_args,
                fuzzer,
                launcher,
                spy,
                verbose,
            ),
            callback=callback,
            error_callback=error_callback,
        )

    thread_pool.close()

    num_remaining = num_queued
    num_failed = 0
    try:
        while num_remaining > 0:
            proc = result_queue.get()
            if proc:
                report_failure(proc)
                num_failed += 1
            num_remaining -= 1
        thread_pool.join()
    except KeyboardInterrupt:
        thread_pool.terminate()
        raise

    print(f"Found {num_failed} failures", flush=True)
    if num_failed > 0:
        sys.exit(1)


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
        "--extra",
        action="append",
        default=[],
        dest="extra_args",
        help="extra arguments for fuzzer",
    )
    parser.add_argument(
        "--fuzzer",
        required=True,
        dest="fuzzer",
        help="location of fuzzer executable",
    )
    parser.add_argument("--launcher", help="launcher command")
    parser.add_argument("--spy", help="location of Legion Spy script")
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
