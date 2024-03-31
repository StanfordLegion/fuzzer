#!/bin/bash

set -e
set -x

FUZZER_INSTALL_DEPS=${FUZZER_INSTALL_DEPS:-0}
FUZZER_INSTALL_LEGION=${FUZZER_INSTALL_LEGION:-1}
FUZZER_DEBUG=${FUZZER_DEBUG:-1}
FUZZER_THREADS=${FUZZER_THREADS:-4}

if [[ $FUZZER_INSTALL_DEPS -eq 1 ]]; then
    $SUDO_COMMAND apt-get update -qq
    $SUDO_COMMAND apt-get install -qq mpich libmpich-dev
fi

if [[ $FUZZER_INSTALL_LEGION -eq 1 ]]; then
    if [[ ! -e legion ]]; then
        git clone https://gitlab.com/StanfordLegion/legion.git
    fi

    pushd legion
    if [[ ! -e build ]]; then
        mkdir build
        pushd build
        legion_flags=(
            -DCMAKE_BUILD_TYPE=$([ $FUZZER_DEBUG -eq 1 ] && echo Debug || echo Release)
            -DCMAKE_INSTALL_PREFIX=$PWD/../install
            -DCMAKE_CXX_STANDARD=17
            -DCMAKE_CXX_FLAGS_DEBUG='-g -O2' # improve performance of debug code
            -DBUILD_SHARED_LIBS=ON # to improve link speed
        )
        if [[ $FUZZER_DEBUG -eq 1 ]]; then
            legion_flags+=(
                -DLegion_SPY=ON
                -DBUILD_MARCH= # to avoid -march=native for valgrind compatibility
            )
        fi
        if [[ -n $FUZZER_LEGION_NETWORKS ]]; then
            legion_flags+=(
                -DLegion_NETWORKS=$FUZZER_LEGION_NETWORKS
            )
        fi
        cmake "${legion_flags[@]}" ..
        make install -j${FUZZER_THREADS:-4}
        popd
    fi
    popd
fi

mkdir -p build
pushd build
fuzzer_flags=(
    -DCMAKE_BUILD_TYPE=$([ ${FUZZER_DEBUG:-1} -eq 1 ] && echo RelWithDebInfo || echo Release)
    -DCMAKE_PREFIX_PATH=$PWD/../legion/install
    -DCMAKE_CXX_FLAGS="-Wall -Werror"
)
if [[ -n $FUZZER_LEGION_NETWORKS ]]; then
    fuzzer_flags+=(
        -DFUZZER_TEST_LAUNCHER="mpirun;-n;2"
    )
fi
cmake "${fuzzer_flags[@]}" ..
make -j${FUZZER_THREADS:-4}
popd

export REALM_SYNTHETIC_CORE_MAP=
./runner.py --fuzzer=$PWD/build/src/fuzzer -j${FUZZER_THREADS:-4} -n1000
