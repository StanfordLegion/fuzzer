#!/usr/bin/env python3

# Copyright 2026 Stanford University
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

from __future__ import annotations
import argparse, dataclasses, glob, multiprocessing, os, queue, shlex, shutil, subprocess, sys, tempfile


@dataclasses.dataclass
class FuzzArgs:
    tid: int
    tnum: int
    seed: int
    num_ops: int
    gpus_per_task: int
    gpus_per_node: int
    extra_args: list[str]
    fuzzer: str
    launcher: str
    spy: str
    verbose: int
    skip: int | None = None


def prefix(args):
    digits = len(str(args.tnum))
    return f"[{args.tid + 1:{digits}} of {args.tnum}]:"


def pr(args, message):
    print(f"{prefix(args)} {message}", flush=True)


def cuda_visible_devices(args):
    first_gpu = (args.tid * args.gpus_per_task) % args.gpus_per_node
    return ",".join(map(str, range(first_gpu, first_gpu + args.gpus_per_task)))


def run_spy(args, logfiles):
    cmd = [
        "pypy3",
        args.spy,
        "--logical",
        "--physical",
        "--cycle",
        # "--sanity", # FIXME: This breaks on several test cases.
        "--leaks",
        # "--geometry", # FIXME: This is *very* slow.
        "--assert-error",
        "--assert-warning",
    ] + logfiles
    if args.verbose >= 3:
        pr(args, f"Running {shlex.join(cmd)}")
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if proc.returncode == 0:
        return
    return proc


def run_fuzzer(args):
    cmd = []
    if args.launcher:
        cmd.extend(shlex.split(args.launcher))
        if args.gpus_per_task is not None:
            # SLURM doesn't like to play nice with setting
            # CUDA_VISIBLE_DEVICES from the outside environment, so we
            # need to do it within the command line.
            cmd.extend(
                ["/usr/bin/env", f"CUDA_VISIBLE_DEVICES={cuda_visible_devices(args)}"]
            )
    cmd.extend(
        [args.fuzzer, "-fuzz:seed", str(args.seed), "-fuzz:ops", str(args.num_ops)]
    )
    if args.skip is not None:
        cmd.extend(["-fuzz:skip", str(args.skip)])
    cmd.extend(arg for extra in args.extra_args for arg in shlex.split(extra))
    if args.gpus_per_task is not None:
        cmd.extend(["-ll:gpu", str(args.gpus_per_task)])
    if args.spy:
        log_dir = tempfile.mkdtemp(dir=os.getcwd())
        cmd.extend(
            ["-level", "legion_spy=2", "-logfile", os.path.join(log_dir, "spy_%.log")]
        )
    else:
        cmd.extend(["-level", "4"])

    env = {}
    if args.gpus_per_task is not None and not args.launcher:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices(args)

    if env:
        env = {**dict(os.environ.items()), **env}
    else:
        env = None

    if args.verbose >= 3:
        pr(args, f"Running {shlex.join(cmd)}")
    if args.verbose >= 4:
        pr(args, f"Environment {env}")
    try:
        proc = subprocess.run(
            cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        if proc.returncode == 0:
            spy_proc = None
            if args.spy:
                spy_logs = glob.glob(os.path.join(log_dir, "spy_*.log"))
                spy_proc = run_spy(args, spy_logs)
            if spy_proc is None:
                return
            return (proc, spy_proc)
        return (proc, None)
    finally:
        if args.spy:
            shutil.rmtree(log_dir)


def bisect_start(args):
    if args.verbose >= 2:
        pr(args, f"Bisecting {args.num_ops} ops at seed {args.seed}")
    good = args.num_ops
    bad = 0
    last_failure = None
    while bad + 1 < good:
        check = (good + bad) // 2
        if args.verbose >= 2:
            pr(
                args,
                f"Testing {args.num_ops} ops (skipping {check}) at seed {args.seed}",
            )
        args = dataclasses.replace(args, skip=check)
        proc = run_fuzzer(args)
        if proc is None:
            good = check
        else:
            bad = check
            last_failure = proc
    return bad, last_failure


def bisect_stop(args):
    if args.verbose >= 2:
        pr(f"Bisecting {args.num_ops} ops at seed {args.seed}")
    good = 0
    bad = args.num_ops
    last_failure = None
    while good + 1 < bad:
        check = (good + bad) // 2
        if args.verbose >= 2:
            pr(args, f"Testing {check} ops at seed {args.seed}")
        args = dataclasses.replace(args, num_ops=check)
        proc = run_fuzzer(args)
        if proc is None:
            good = check
        else:
            bad = check
            last_failure = proc
    return bad, last_failure


def fuzz(args):
    if args.verbose >= 2:
        pr(args, f"Testing {args.num_ops} ops at seed {args.seed}")
    proc = run_fuzzer(args)
    if proc is None:
        return
    if args.verbose >= 1:
        pr(args, f"Found failure: {shlex.join(proc[0].args)}")
    stop, stop_proc = bisect_stop(args)
    args = dataclasses.replace(args, num_ops=stop)
    start, start_proc = bisect_start(args)
    proc = start_proc or stop_proc or proc
    if args.verbose >= 1:
        pr(args, f"Shortest failure: {shlex.join(proc[0].args)}")
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
    gpus_per_task,
    gpus_per_node,
    extra_args,
    fuzzer,
    launcher,
    spy,
    verbose,
):
    assert (gpus_per_task is None) == (gpus_per_node is None)
    if gpus_per_task is not None:
        assert gpus_per_task > 0
        assert gpus_per_node > 0
        assert gpus_per_node % gpus_per_task == 0

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
                FuzzArgs(
                    tid=tid,
                    tnum=num_tests,
                    seed=seed,
                    num_ops=num_ops,
                    gpus_per_task=gpus_per_task,
                    gpus_per_node=gpus_per_node,
                    extra_args=extra_args,
                    fuzzer=fuzzer,
                    launcher=launcher,
                    spy=spy,
                    verbose=verbose,
                ),
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
        "--gpus-per-task",
        type=int,
        required=False,
        dest="gpus_per_task",
        help="number of GPUs to assign per task",
    )
    parser.add_argument(
        "--gpus-per-node",
        type=int,
        required=False,
        dest="gpus_per_node",
        help="total number of GPUs available per node",
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
