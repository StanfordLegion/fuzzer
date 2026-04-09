export FUZZER_MACHINE=sapling
export FUZZER_THREADS=20
export FUZZER_CONDUIT=ibv
export FUZZER_USE_CUDA=1

module load cuda

export CC=gcc CXX=g++
