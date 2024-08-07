#!/bin/bash

set -e

if [[ -z ${FUZZER_MACHINE} ]]; then
    echo "Did you remember to source experiments/MY_MACHINE/env.sh? (For an appropriate value of MY_MACHINE)"
    exit 1
fi

root_dir="$(dirname "$(dirname "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")")")"
cd "$root_dir"

export FUZZER_OP_COUNT=1000

function run_fuzzer_config {
    config_name="$1"
    mode="$2"
    extra_flags="$3"

    fuzzer_exe="$PWD/build_${config_name}/src/fuzzer"
    fuzzer_flags="-ll:util 2 -ll:cpu 3 $extra_flags"

    if [[ $mode = single ]]; then
        test_count=100000
        launcher="srun -n 1"
    elif [[ $mode = multi ]]; then
        # We can't do as many tests in multi-node mode because SLURM has a
        # hard upper bound on the number of steps per job.
        test_count=10000
        launcher="srun -n 2 --ntasks-per-node 2 --overlap"
    else
        echo "Don't recognize fuzzer mode $mode"
        exit 1
    fi

    # Generate a random seed so we explore a novel part of the state space.
    seed="$(( 16#$(openssl rand -hex 4) * test_count ))"

    FUZZER_EXE="$fuzzer_exe" FUZZER_MODE=$mode FUZZER_TEST_COUNT=$test_count FUZZER_SEED=$seed FUZZER_LAUNCHER="$launcher" FUZZER_EXTRA_FLAGS="$fuzzer_flags" sbatch --nodes 1 "experiment/$FUZZER_MACHINE/sbatch_fuzzer.sh"
}

run_fuzzer_config debug_single single
run_fuzzer_config release_single single
run_fuzzer_config debug_multi multi
run_fuzzer_config release_multi multi
run_fuzzer_config debug_multi multi "-fuzz:replicate 1"
run_fuzzer_config release_multi multi "-fuzz:replicate 1"
