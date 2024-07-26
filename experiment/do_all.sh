#!/bin/bash

machine="$1"

set -e

if [[ -z ${machine} ]]; then
    echo "Please specify a machine to run this experiment for."
    exit 1
fi

root_dir="$(dirname "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")")"
cd "$root_dir"

source experiment/$machine/env.sh
./experiment/$machine/do_build.sh
./experiment/$machine/run_all_tests.sh
