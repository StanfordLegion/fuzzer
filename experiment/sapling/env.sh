export FUZZER_MACHINE=sapling
export FUZZER_THREADS=20
export FUZZER_NETWORKS="gasnetex ucx"
export FUZZER_GASNET_CONDUIT=ibv
export FUZZER_USE_CUDA=0

# module load cuda

export CC=gcc CXX=g++
