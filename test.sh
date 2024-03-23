#!/bin/bash

set -e
set -x

if [[ $INSTALL_DEPS -eq 1 ]]; then
    $SUDO_COMMAND apt-get update -qq
    $SUDO_COMMAND apt-get install -qq mpich libmpich-dev
fi

if [[ ! -e legion ]]; then
    git clone https://gitlab.com/StanfordLegion/legion.git
fi

pushd legion
if [[ ! -e build ]]; then
    mkdir build
    pushd build
    legion_flags=(
        -DCMAKE_BUILD_TYPE=$([ ${FUZZER_DEBUG:-1} -eq 1 ] && echo Debug || echo Release)
        -DCMAKE_INSTALL_PREFIX=$PWD/../install
        -DCMAKE_CXX_STANDARD=17
        -DLegion_SPY=ON
        -DBUILD_SHARED_LIBS=ON # to improve link speed
    )
    if [[ ${FUZZER_DEBUG:-1} -eq 1 ]]; then
        legion_flags+=(
            -DBUILD_MARCH= # to avoid -march=native for valgrind compatibility
        )
    fi
    if [[ -n $FUZZER_LEGION_NETWORKS ]]; then
        legion_flags+=(
            -DLegion_NETWORKS=$FUZZER_LEGION_NETWORKS
        )
    fi
    cmake "${legion_flags[@]}" ..
    make install -j${THREADS:-4}
    popd
fi
popd

mkdir -p build
pushd build
fuzzer_flags=(
    -DCMAKE_BUILD_TYPE=$([ ${FUZZER_DEBUG:-1} -eq 1 ] && echo Debug || echo Release)
    -DCMAKE_PREFIX_PATH=$PWD/../legion/install
    -DCMAKE_CXX_FLAGS="-Wall -Werror"
    # do NOT set NDEBUG, it causes all sorts of issues
    -DCMAKE_CXX_FLAGS_RELEASE="-O2 -march=native"
    -DCMAKE_CXX_FLAGS_RELWITHDEBINFO="-O2 -g -march=native"
)
if [[ -n $FUZZER_LEGION_NETWORKS ]]; then
    fuzzer_flags+=(
        -DFUZZER_TEST_LAUNCHER="mpirun;-n;2"
    )
fi
cmake "${fuzzer_flags[@]}" ..
make -j${THREADS:-4}
export REALM_SYNTHETIC_CORE_MAP=
ctest --output-on-failure -j${THREADS:-4}
popd
