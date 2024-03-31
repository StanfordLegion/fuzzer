#!/bin/bash

set -e

for f in src/*.h src/*.cc src/*.inl; do
    clang-format -i "$f" &
done
black . &
wait

if [[ $CHECK_FORMAT -eq 1 ]]; then
    set -x
    git status
    git diff
    git diff-index --quiet HEAD
fi
