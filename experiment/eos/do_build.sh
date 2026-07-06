#!/bin/bash

branch="$1"

set -e

root_dir="$(dirname "$(dirname "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")")")"
cd "$root_dir"

srun -n 1 -N 1 -c 40 -p all --exclusive --pty ./experiment/common/setup.sh "$branch"
