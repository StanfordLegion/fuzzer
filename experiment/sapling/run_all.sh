#!/bin/bash

set -e

# Currently we give everything equal testing
export FUZZER_TEST_COUNT=10000
export FUZZER_OP_COUNT=1000
export FUZZER_EXTRA_FLAGS="-ll:util 2 -ll:cpu 3"
export FUZZER_LAUNCHER="srun -n 1 --overlap"

function run_fuzzer_config {
    config_name="$1"

    fuzzer_exe="$PWD/build_${config_name}/src/fuzzer"

    FUZZER_EXE="$fuzzer_exe" sbatch --nodes 1 sbatch_fuzzer.sh
}

run_fuzzer_config debug_single
run_fuzzer_config release_single
