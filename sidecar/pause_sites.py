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

# Where did the pauses land? Reads the log directories written by
# signal_shim (log=DIR) or ptrace_shim (-l DIR) for any program.
#
#   pause_sites.py [--functions N] DIR [DIR...]
#
# Each pauses-HOST-PID.log is matched with its maps-HOST-PID[.log] snapshot.
# Reports the share of pauses in user code (vs. a system call), the libraries
# the user-code pauses landed in with the number of distinct sites (library,
# file offset) per library, how concentrated pauses were on a few threads,
# and optionally the top N functions (via addr2line).
#
# Log columns: tid t0 t1 t2 t3 rip syscall. The last column is the real
# orig_rax for ptrace_shim (>= 0: in a system call); signal_shim writes 0 for
# a system call, -1 for user code and -2 if unknown.

import argparse, bisect, collections, glob, os, statistics, subprocess


def load_maps(path):
    ranges = []
    for line in open(path):
        parts = line.split()
        lo, hi = (int(x, 16) for x in parts[0].split("-"))
        ranges.append(
            (lo, hi, int(parts[2], 16), parts[5] if len(parts) > 5 else "[anon]")
        )
    ranges.sort()
    return ranges


def lookup(ranges, addr):
    """(library path, file offset) of addr, or None."""
    i = bisect.bisect_right(ranges, (addr, float("inf"))) - 1
    if i >= 0 and ranges[i][0] <= addr < ranges[i][1]:
        lo, _, offset, name = ranges[i]
        return name, addr - lo + offset
    return None


def find_maps(log):
    base = log[: -len(".log")].replace("pauses-", "maps-", 1)
    for path in (base + ".log", base):
        if os.path.exists(path):
            return path
    return None


def symbolize(sites):
    """{(library, offset): count} -> {(library basename, function): count}"""
    by_lib = collections.defaultdict(list)
    for lib, off in sites:
        by_lib[lib].append(off)
    funcs = collections.Counter()
    for lib, offs in by_lib.items():
        names = ["??"] * len(offs)
        if lib.startswith("/") and os.path.exists(lib):
            proc = subprocess.run(
                ["addr2line", "-f", "-C", "-e", lib],
                input="\n".join(hex(o) for o in offs),
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            got = proc.stdout.splitlines()[0::2]
            names = got + names[len(got) :]
        for off, fn in zip(offs, names):
            funcs[(os.path.basename(lib), fn)] += sites[(lib, off)]
    return funcs


def main():
    ap = argparse.ArgumentParser(description="Where did the pauses land?")
    ap.add_argument("dirs", nargs="+")
    ap.add_argument(
        "--functions", type=int, default=0, metavar="N", help="list the top N functions"
    )
    ap.add_argument(
        "--libraries",
        type=int,
        default=10,
        metavar="N",
        help="list the top N libraries",
    )
    args = ap.parse_args()

    stops = user = no_maps = malformed = 0
    sites = collections.Counter()  # (library, offset) -> user-code stops
    top2 = []
    for d in args.dirs:
        for log in glob.glob(os.path.join(d, "pauses-*.log")):
            maps = find_maps(log)
            ranges = load_maps(maps) if maps else []
            per_tid = collections.Counter()
            for line in open(log, errors="replace"):
                f = line.split()
                try:
                    if len(f) != 7:
                        raise ValueError
                    rip, syscall = int(f[5], 16), int(f[6])
                except ValueError:
                    malformed += 1  # e.g. the last line of a killed process
                    continue
                stops += 1
                per_tid[f[0]] += 1
                if syscall >= 0:
                    continue  # in a system call
                user += 1
                site = lookup(ranges, rip) if ranges else None
                if site is None:
                    no_maps += 1
                    site = ("[unknown]", 0)
                sites[site] += 1
            n = sum(per_tid.values())
            if n >= 100:
                top2.append(sum(c for _, c in per_tid.most_common(2)) / n)

    print(
        f"pauses: {stops}, in user code: {100 * user / max(stops, 1):.1f}%"
        f" ({no_maps} without a matching map, {malformed} malformed lines skipped)"
    )
    if top2:
        print(
            f"share of a process's pauses taken by its 2 most-paused threads"
            f" (processes with >= 100 pauses): median {statistics.median(top2):.2f}"
        )
    per_lib = collections.Counter()
    distinct = collections.Counter()
    for (lib, _), n in sites.items():
        per_lib[os.path.basename(lib)] += n
        distinct[os.path.basename(lib)] += 1
    print(f"{'library':32s} {'share':>6s} {'pauses':>9s} {'sites':>7s}")
    for lib, n in per_lib.most_common(args.libraries):
        print(
            f"{lib[:32]:32s} {100 * n / max(user, 1):5.1f}% {n:9d} {distinct[lib]:7d}"
        )
    if args.functions:
        print(f"\ntop functions (share of user-code pauses):")
        for (lib, fn), n in symbolize(sites).most_common(args.functions):
            print(f"  {100 * n / max(user, 1):5.1f}%  {lib[:24]:24s} {fn[:100]}")


if __name__ == "__main__":
    main()
