#!/bin/bash

machine="$1"

set -e

source experiment/$machine/env.sh
./experiment/$machine/do_build.sh
./experiment/$machine/run_all_tests.sh
