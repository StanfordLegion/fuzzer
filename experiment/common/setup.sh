#!/bin/bash

set -e

if [[ -z ${FUZZER_MACHINE} ]]; then
    echo "Did you remember to source experiments/MY_MACHINE/env.sh? (For an appropriate value of MY_MACHINE)"
    exit 1
fi

root_dir="$(dirname "$(dirname "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")")")"
cd "$root_dir"

function build_legion_config {
    config_name="$1"
    cmake_build_type="$2"

    build_dir="build_${config_name}"
    install_dir="$PWD/install_${config_name}"

    if [[ -e $build_dir ]]; then
        return
    fi

    mkdir "$build_dir"
    pushd "$build_dir"
    cmake_flags=(
        -DCMAKE_BUILD_TYPE="$cmake_build_type"
        -DCMAKE_INSTALL_PREFIX="$install_dir"
        -DCMAKE_CXX_STANDARD=17
        -DCMAKE_CXX_FLAGS_DEBUG='-g -O2' # improve performance of debug code
    )
    if [[ $cmake_build_type = Debug ]]; then
        cmake_flags+=(
            -DBUILD_SHARED_LIBS=ON # to improve link speed
            -DBUILD_MARCH= # to avoid -march=native for valgrind compatibility
        )
    fi
    cmake "${cmake_flags[@]}" ..
    make install -j${FUZZER_THREADS:-4}
    popd
}

function build_fuzzer_config {
    config_name="$1"
    cmake_build_type="$2"

    build_dir="build_${config_name}"
    legion_install_dir="$PWD/legion/install_${config_name}"

    if [[ -e $build_dir ]]; then
        return
    fi

    mkdir "$build_dir"
    pushd "$build_dir"
    cmake_flags=(
        -DCMAKE_BUILD_TYPE="$cmake_build_type"
        -DCMAKE_PREFIX_PATH="$legion_install_dir"
        -DCMAKE_CXX_FLAGS="-Wall -Werror"
    )
    cmake "${cmake_flags[@]}" ..
    make -j${FUZZER_THREADS:-4}
    popd
}

if [[ ! -e legion ]]; then
    git clone https://gitlab.com/StanfordLegion/legion.git
fi

pushd legion
build_legion_config debug_single Debug
build_legion_config release_single Release
popd

build_fuzzer_config debug_single RelWithDebInfo
build_fuzzer_config release_single Release
