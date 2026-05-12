#!/bin/bash

branch="$1"

set -e

if [[ -z ${FUZZER_MACHINE} ]]; then
    echo "Did you remember to source experiments/MY_MACHINE/env.sh? (For an appropriate value of MY_MACHINE)"
    exit 1
fi

root_dir="$(dirname "$(dirname "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")")")"
cd "$root_dir"

DEP_CACHE="$PWD"/deps
mkdir -p "$DEP_CACHE"

# Borrow the Legion CI configuration here so that we're sure we're building the same way
function build_ucx {
    UCX_VERSION=1.20.1
    UCX_SHA256=545c419a7b5e04643cb8bff5a19b3b5071a8f8f0605f1e8efb36f8f3d7bfb9d3

    mkdir ucx
    pushd ucx
    if ! echo "$UCX_SHA256  $DEP_CACHE/ucx-$UCX_VERSION.tar.gz" | shasum -a 256 -c; then
        curl -sL https://github.com/openucx/ucx/releases/download/v$UCX_VERSION/ucx-$UCX_VERSION.tar.gz -o "$DEP_CACHE"/ucx-$UCX_VERSION.tar.gz
        echo "$UCX_SHA256  $DEP_CACHE/ucx-$UCX_VERSION.tar.gz" | shasum -a 256 -c
    fi
    tar --strip-components=1 -zxf "$DEP_CACHE"/ucx-$UCX_VERSION.tar.gz
    configure_flags=()
    if [[ -n $FUZZER_CUDA ]]; then
        configure_flags+=(
            --with-cuda="$FUZZER_CUDA"
        )
    else
        configure_flags+=(
            --without-cuda
        )
    fi
    contrib/configure-release-mt --prefix="$PWD/install" "${configure_flags[@]}"
    make -j${FUZZER_THREADS:-4} install
    popd
}

function build_ucc {
    UCC_VERSION=1.7.0
    UCC_SHA256=b40df0db75b8505844547574a3a7dad16c9033d7e1ca099ea8508bc57a62b454

    mkdir ucc
    pushd ucc
    if ! echo "$UCC_SHA256  $DEP_CACHE/ucc-$UCC_VERSION.tar.gz" | shasum -a 256 -c; then
        curl -sL https://github.com/openucx/ucc/archive/refs/tags/v$UCC_VERSION.tar.gz -o "$DEP_CACHE"/ucc-$UCC_VERSION.tar.gz
        echo "$UCC_SHA256  $DEP_CACHE/ucc-$UCC_VERSION.tar.gz" | shasum -a 256 -c
    fi
    tar --strip-components=1 -zxf "$DEP_CACHE"/ucc-$UCC_VERSION.tar.gz

    # Remove offload-arch=native.
    sed -i 's/--offload-arch=native//g' ./cuda_lt.sh

    ./autogen.sh
    ./configure --prefix="$PWD/install" --with-ucx="$ucx_ROOT" --without-rocm
    make -j${FUZZER_THREADS:-4} install
    popd
}

function build_legion_config {
    config_name="$1"
    cmake_build_type="$2"
    extra_cmake_flags="$3"

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
        # reduce IBV max medium to trip payload limit
        -DLegion_EMBED_GASNet_CONFIGURE_ARGS="--disable-kind-cuda-uva --with-ibv-max-medium=1024 --disable-pshm"
    )
    if [[ $cmake_build_type = Debug ]]; then
        cmake_flags+=(
            -DBUILD_SHARED_LIBS=ON # to improve link speed
            -DBUILD_MARCH= # to avoid -march=native for valgrind compatibility
        )
    fi
    (set -x; cmake "${cmake_flags[@]}" $extra_cmake_flags ..)
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
    (set -x; cmake "${cmake_flags[@]}" ..)
    make -j${FUZZER_THREADS:-4}
    popd
}

if echo $FUZZER_NETWORKS | grep -q -w ucx; then
    if [[ ! -e ucx ]]; then
        build_ucx
    fi
    export ucx_ROOT="$PWD/ucx/install"

    if [[ ! -e ucc ]]; then
        build_ucc
    fi
    export ucc_ROOT="$PWD/ucc/install"
fi

if [[ ! -e legion ]]; then
    clone_flags=()
    if [[ -n $branch ]]; then
        clone_flags+=(-b "$branch")
    fi
    git clone "${clone_flags[@]}" https://gitlab.com/StanfordLegion/legion.git
fi

pushd legion
build_legion_config debug_single Debug
build_legion_config spy_single Debug -DLegion_SPY=ON
build_legion_config release_single Release
if echo $FUZZER_NETWORKS | grep -q -w gasnetex; then
    build_legion_config debug_multi_gex Debug "-DLegion_NETWORKS=gasnetex -DLegion_EMBED_GASNet=ON -DGASNet_CONDUIT=$FUZZER_GASNET_CONDUIT"
    build_legion_config release_multi_gex Release "-DLegion_NETWORKS=gasnetex -DLegion_EMBED_GASNet=ON -DGASNet_CONDUIT=$FUZZER_GASNET_CONDUIT"
fi
if echo $FUZZER_NETWORKS | grep -q -w ucx; then
    build_legion_config debug_multi_ucx Debug "-DLegion_NETWORKS=ucx -DCMAKE_INSTALL_RPATH=$ucx_ROOT/lib;$ucc_ROOT/lib"
    build_legion_config release_multi_ucx Release "-DLegion_NETWORKS=ucx -DCMAKE_INSTALL_RPATH=$ucx_ROOT/lib;$ucc_ROOT/lib"
fi
popd

build_fuzzer_config debug_single RelWithDebInfo
build_fuzzer_config spy_single RelWithDebInfo
build_fuzzer_config release_single Release
if echo $FUZZER_NETWORKS | grep -q -w gasnetex; then
    build_fuzzer_config debug_multi_gex RelWithDebInfo
    build_fuzzer_config release_multi_gex Release
fi
if echo $FUZZER_NETWORKS | grep -q -w ucx; then
    build_fuzzer_config debug_multi_ucx RelWithDebInfo
    build_fuzzer_config release_multi_ucx Release
fi
