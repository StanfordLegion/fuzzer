#!/bin/bash

set -e

srun -n 1 -N 1 -c 40 -p all --exclusive --pty ./experiment/setup.sh
