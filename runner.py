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

import argparse, multiprocessing, os, queue, shlex, subprocess

def run_fuzzer(seed, num_ops, fuzzer_exe, skip=None):
    cmd = [fuzzer_exe, '-fuzz:seed', str(seed), '-fuzz:ops', str(num_ops)]
    if skip is not None:
        cmd.extend(['-fuzz:skip', str(skip)])
    cmd.extend(['-level', '4'])
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode == 0:
        return
    return proc

def bisect_start(thread_index, thread_count, seed, num_ops, fuzzer_exe, verbose):
    if verbose >= 2:
        print(f'[{thread_index} of {thread_count}]: Bisecting {num_ops} ops at seed {seed} to find shortest failure')
    good = num_ops
    bad = 0
    last_failure = None
    while bad + 1 < good:
        check = (good + bad)//2
        if verbose >= 2:
            print(f'[{thread_index} of {thread_count}]: Testing {num_ops} ops (skipping {check}) at seed {seed}')
        proc = run_fuzzer(seed, num_ops, fuzzer_exe, skip=check)
        if proc is None:
            good = check
        else:
            bad = check
            last_failure = proc
    return bad, last_failure

def bisect_stop(thread_index, thread_count, seed, num_ops, fuzzer_exe, verbose):
    if verbose >= 2:
        print(f'[{thread_index} of {thread_count}]: Bisecting {num_ops} ops at seed {seed} to find shortest failure')
    good = 0
    bad = num_ops
    last_failure = None
    while good + 1 < bad:
        check = (good + bad)//2
        if verbose >= 2:
            print(f'[{thread_index} of {thread_count}]: Testing {check} ops at seed {seed}')
        proc = run_fuzzer(seed, check, fuzzer_exe)
        if proc is None:
            good = check
        else:
            bad = check
            last_failure = proc
    return bad, last_failure

def fuzz(thread_index, thread_count, seed, num_ops, fuzzer_exe, verbose):
    if verbose >= 2:
        print(f'[{thread_index} of {thread_count}]: Testing {num_ops} ops at seed {seed}')
    proc = run_fuzzer(seed, num_ops, fuzzer_exe)
    if proc is None:
        return
    if verbose >= 1:
        print(f'[{thread_index} of {thread_count}]: Found failure: {shlex.join(proc.args)}')
    stop, stop_proc = bisect_stop(thread_index, thread_count, seed, num_ops, fuzzer_exe, verbose)
    start, start_proc = bisect_start(thread_index, thread_count, seed, stop, fuzzer_exe, verbose)
    proc = start_proc or stop_proc or proc
    if verbose >= 1:
        print(f'[{thread_index} of {thread_count}]: Shortest failure for this seed: {shlex.join(proc.args)}')
    return proc

def run_test(thread_index, thread_count, num_tests, num_ops, base_seed, fuzzer_exe, spy_script, verbose):
    if verbose >= 1:
        print(f'[{thread_index} of {thread_count}]: Starting search')
    initial_seed = base_seed + thread_index
    failed_procs = []
    for test_index in range(num_tests):
        seed = initial_seed + test_index * thread_count
        proc = fuzz(thread_index, thread_count, seed, num_ops, fuzzer_exe, verbose)
        if proc:
            failed_procs.append(proc)
    return failed_procs

def report_failure(proc):
    print(f'Captured failure: {shlex.join(proc.args)}')
    print(proc.stderr.decode('utf-8'))
    print()

def run_tests(thread_count, num_tests, num_ops, base_seed, fuzzer_exe, spy_script, verbose):
    if thread_count is None:
        thread_count = os.cpu_count()

    thread_pool = multiprocessing.Pool(thread_count)

    result_queue = queue.Queue()
    num_queued = 0
    for thread_index in range(thread_count):
        num_queued += 1
        def callback(r):
            result_queue.put(r)
        def error_callback(e):
            print('ERROR CALLBACK', e)
            raise e
        thread_pool.apply_async(
            run_test,
            (thread_index, thread_count, num_tests, num_ops, base_seed, fuzzer_exe, spy_script, verbose),
            callback=callback,
            error_callback=error_callback)

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
    parser = argparse.ArgumentParser(description='Fuzzer test harness')
    parser.add_argument('-j',
                        nargs='?',
                        type=int,
                        help='number threads used to test',
                        dest='thread_count')
    parser.add_argument('-n', '--tests',
                        type=int,
                        default=10,
                        dest='num_tests',
                        help='number of tests to run per seed')
    parser.add_argument('-o', '--ops',
                        type=int,
                        default=100,
                        dest='num_ops',
                        help='number of operations to run per test')
    parser.add_argument('-s', '--seed',
                        type=int,
                        default=0,
                        dest='base_seed',
                        help='base seed to use')
    parser.add_argument('--fuzzer',
                        required=True,
                        dest='fuzzer_exe',
                        help='location of fuzzer executable')
    parser.add_argument('--spy',
                        dest='spy_script',
                        help='location of Legion Spy script (optional)')
    parser.add_argument('-v', '--verbose',
                        action='count',
                        default=0,
                        dest='verbose',
                        help='enable verbose output')
    args = parser.parse_args()
    run_tests(**vars(args))

if __name__ == '__main__':
    driver()
