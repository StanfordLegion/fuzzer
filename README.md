# Legion Fuzzer

This is a fuzzer for the [Legion programming
system](https://legion.stanford.edu/). The purpose of this fuzzer is NOT to
test every possible Legion feature (a job better left to unit and integration
tests). Instead, the goal of this fuzzer is to exercise as thoroughly as
possible certain core Legion analyses that are critical to the correct
execution of every Legion program:

  * Logical analysis
  * Physical analysis
  * Local and remote task execution
  * Inferred copies
  * Partitioning, particularly with nontrivial aliasing and multiple
    partitions per region
  * Region accessors
  * Scalar reductions

## Quickstart

A script is provided that downloads Legion and builds the fuzzer:

```
FUZZER_THREADS=8 ./test.sh
```

Set `FUZZER_THREADS` to some appropriate number of threads for your system.

This will run a very short test, for demonstration purposes only.

Having built the fuzzer, you will likely wish to then run it directly:

```
export REALM_SYNTHETIC_CORE_MAP=
./runner.py --fuzzer build/src/fuzzer -j8 -n1000
```

The parameter `-j` sets the number of threads, while `-n` sets the total
number of tests to run. The setting of `REALM_SYNTHETIC_CORE_MAP=` helps to
ensure that the fuzzer makes full use of your machine.

Every run of the fuzzer is validated to ensure it produces the correct output
(and does not crash). Failing configurations will be minimized and then
reported to the user, along with their failure mode.

## Design

Conceptually, the fuzzer is organized into two pieces:

  * Plan what operations to execute
  * Execute them and compare the result against a serial reference

Each operation in the fuzzer is encapsulated by an `OperationBuilder`
object. The `build()` method uses a (deterministic) random number generator to
decide what parameters to set for the operation. A sample of the current
parameters includes:

  * Launch type: index task launch or (a set of) individual tasks
  * Launch domain
  * Region requirement (currently exactly one per task)
      * Privilege
      * Reduction (if applicable)
      * Fields
      * Partition
      * Projection functor (if applicable)
  * And so on

The `execute()` method constructs the corresponding `Launcher` object and runs
it. In addition, it runs the same task body locally within the top-level task
on a separate, "shadow" copy of the region tree to compute the result of a
serial execution of exactly the same operations. Because Legion's semantics
are sequential, this (by definition) is the correct output of the program, and
deviation from this is a Legion bug.

It is VERY IMPORTANT that the `execute()` method *never* consult the random
number generator *for any purpose*. This is to ensure that the trace, or set
of operations to be executed, remains stable even if some of the earlier
operations in the trace are skipped. (Minimizing the trace is an important
step for debugging. Shorter traces are easier to debug.) When we find a
failure we bisect the trace from both sides to locate the latest point to
start the trace, and the earliest point we can stop the trace, while
continuing to manifest a failure.

Note that Legion supports features that permit nondeterministic execution
(e.g., floating point reductions, relaxed coherence modes). The fuzzer avoids
these features as there is no single, unique sequential execution these
correspond to and thus no source of truth.
