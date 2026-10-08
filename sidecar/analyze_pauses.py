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

# Match a pause log from either tool (signal_shim log=DIR or ptrace_shim -l
# DIR) against the gap file written by the spinners microbenchmark.
#
#   analyze_pauses.py GAPS SECONDS NTHREADS [PAUSE_LOG]
#
# GAPS is the spinners OUTFILE; SECONDS and NTHREADS must be the values the
# spinners run used (they only turn counts into rates); PAUSE_LOG is one
# pauses-HOST-PID.log. Without a pause log, only the gaps are summarized.
#
# For each pause, the spinner's own gap covering the moment it was stopped
# tells us whether it was on a CPU when interrupted, how long it took to stop,
# and how long after its release it got back on a CPU. For signal_shim the
# release (t3) is the handler's exit, so the wait to get back on a CPU is
# already part of the hold (t2 - t1) and the release-to-running time is ~0.

import bisect, collections, sys


def pct(xs, ps=(10, 50, 90, 99)):
    if not xs:
        return "n/a"
    xs = sorted(xs)
    return " ".join(f"p{p}={xs[min(len(xs) - 1, len(xs) * p // 100)]:.0f}" for p in ps)


def main():
    if len(sys.argv) not in (4, 5):
        sys.exit("usage: analyze_pauses.py GAPS SECONDS NTHREADS [PAUSE_LOG]")
    gaps_file, seconds, nthreads = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
    pause_file = sys.argv[4] if len(sys.argv) > 4 else None

    gaps = collections.defaultdict(list)
    for line in open(gaps_file):
        tid, s, e = map(int, line.split())
        gaps[tid].append((s, e))
    for g in gaps.values():
        g.sort()
    all_gaps = [e - s for g in gaps.values() for s, e in g]
    print(
        f"  natural+induced gaps: {len(all_gaps) / seconds / nthreads:.0f}/s/thread, "
        f"length us: {pct([x / 1e3 for x in all_gaps])}"
    )
    if not pause_file:
        return

    # columns: tid t0 t1 t2 t3 rip orig_rax (times in ns, CLOCK_MONOTONIC)
    pauses = [list(map(int, line.split()[:5])) for line in open(pause_file)]
    matched = on_cpu = 0
    stop_lat, resume_lat, gap_len, hold = [], [], [], []
    for tid, t0, t1, t2, t3 in pauses:
        g = gaps.get(tid, [])
        i = bisect.bisect_right(g, (t1, float("inf"))) - 1
        if i < 0 or not (g[i][0] <= t1 <= g[i][1]):
            continue
        s, e = g[i]
        matched += 1
        hold.append((t2 - t1) / 1e3)
        gap_len.append((e - s) / 1e3)
        resume_lat.append((e - t3) / 1e3)
        if s >= t0:
            on_cpu += 1
            stop_lat.append((s - t0) / 1e3)
    n = len(pauses)
    print(
        f"  pauses: {n} ({n / seconds:.0f}/s), matched to a gap: {matched} ({100 * matched / max(n, 1):.0f}%)"
    )
    print(
        f"  on CPU when interrupted: {100 * on_cpu / max(matched, 1):.0f}%, stop latency us: {pct(stop_lat)}"
    )
    print(f"  hold us: {pct(hold)}")
    print(f"  release -> running again us: {pct(resume_lat)}")
    print(
        f"  resumed within 100us: {100 * sum(x < 100 for x in resume_lat) / max(matched, 1):.0f}%, "
        f"waited >1ms: {100 * sum(x > 1000 for x in resume_lat) / max(matched, 1):.0f}%"
    )
    print(f"  total time off CPU per pause us: {pct(gap_len)}")


if __name__ == "__main__":
    main()
