#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --partition=all
#SBATCH --time=04:00:00

export REALM_SYNTHETIC_CORE_MAP=
export REALM_BACKTRACE=1

ulimit -S -c 0 # disable core dumps

set -x

launcher=

fuzzer_flags=(
    --fuzzer="$FUZZER_EXE"
    -j${FUZZER_THREADS:-4}
    -n${FUZZER_TEST_COUNT:-1000}
    -o${FUZZER_OP_COUNT:-1000}
    --extra="$FUZZER_EXTRA_FLAGS"
)

if [[ $FUZZER_MODE = single ]]; then
    launcher="$FUZZER_LAUNCHER"
elif [[ $FUZZER_MODE = multi ]]; then
    fuzzer_flags+=(
        --launcher="$FUZZER_LAUNCHER"
    )
else
    echo "Don't recognize fuzzer mode $FUZZER_MODE"
    exit 1
fi

$launcher ./runner.py "${fuzzer_flags[@]}"
