/* Copyright 2026 Stanford University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cstdint>
#include <vector>

#include "legion.h"
#include "realm/cuda/cuda_module.h"
#include "realm/hip/hip_module.h"
#include "tasks.h"
#include "tasks_gpu.h"

using namespace Legion;

typedef FieldAccessor<LEGION_WRITE_ONLY, uint64_t, 1, coord_t,
                      Realm::AffineAccessor<uint64_t, 1, coord_t>>
    AccessorWO;
typedef FieldAccessor<LEGION_READ_WRITE, uint64_t, 1, coord_t,
                      Realm::AffineAccessor<uint64_t, 1, coord_t>>
    AccessorRW;

__global__ void write_field_kernel(AccessorWO acc, Rect<1> rect, uint64_t value) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < rect.volume()) {
    Point<1> p = rect.lo + tid;
    acc[p] = value;
  }
}

__global__ void modify_field_kernel(AccessorRW acc, Rect<1> rect, uint64_t value) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < rect.volume()) {
    Point<1> p = rect.lo + tid;
    acc[p] = (acc[p] >> 1) + value;
  }
}

template <typename REDOP>
__global__ void reduce_field_kernel(
    ReductionAccessor<REDOP, true /* exclusive */, 1, coord_t,
                      Realm::AffineAccessor<uint64_t, 1, coord_t>>
        acc,
    Rect<1> rect, uint64_t value) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < rect.volume()) {
    Point<1> p = rect.lo + tid;
    acc[p] <<= value;
  }
}

#if defined(LEGION_USE_CUDA)
CUstream_st *get_task_stream() {
  Realm::Runtime runtime = Realm::Runtime::get_runtime();
  Realm::Cuda::CudaModule *module = runtime.get_module<Realm::Cuda::CudaModule>("cuda");
  if (!module) {
    abort();
  }

  CUstream_st *result = module->get_task_cuda_stream();
  module->set_task_ctxsync_required(false);
  return result;
}
#elif defined(LEGION_USE_HIP)
unifiedHipStream_t *get_task_stream() {
  Realm::Runtime runtime = Realm::Runtime::get_runtime();
  Realm::Hip::HipModule *module = runtime.get_module<Realm::Hip::HipModule>("hip");
  if (!module) {
    abort();
  }

  unifiedHipStream_t *result = module->get_task_hip_stream();
  module->set_task_ctxsync_required(false);
  return result;
}
#else
#error "Don't know what GPU backend to build for"
#endif

static constexpr int THREADS_PER_BLOCK = 256;

static int num_blocks(size_t num_elements) {
  return (num_elements + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
}

static void gpu_write_field(const PhysicalRegion &region, Rect<1> rect, FieldID fid,
                            PointTaskArgs args) {
  AccessorWO acc(region, fid);
  if (args.gpu_task_use_stream == 0) {
    write_field_kernel<<<num_blocks(rect.volume()), THREADS_PER_BLOCK>>>(acc, rect,
                                                                         args.value);
  } else {
    write_field_kernel<<<num_blocks(rect.volume()), THREADS_PER_BLOCK, 0,
                         get_task_stream()>>>(acc, rect, args.value);
  }
}

static void gpu_modify_field(const PhysicalRegion &region, Rect<1> rect, FieldID fid,
                             PointTaskArgs args) {
  AccessorRW acc(region, fid);
  if (args.gpu_task_use_stream == 0) {
    modify_field_kernel<<<num_blocks(rect.volume()), THREADS_PER_BLOCK>>>(acc, rect,
                                                                          args.value);
  } else {
    modify_field_kernel<<<num_blocks(rect.volume()), THREADS_PER_BLOCK, 0,
                          get_task_stream()>>>(acc, rect, args.value);
  }
}

template <typename REDOP>
static void gpu_reduce_field(const PhysicalRegion &region, Rect<1> rect, FieldID fid,
                             ReductionOpID redop, PointTaskArgs args) {
  const ReductionAccessor<REDOP, true /* exclusive */, 1, coord_t,
                          Realm::AffineAccessor<uint64_t, 1, coord_t>>
      acc(region, fid, redop);
  if (args.gpu_task_use_stream == 0) {
    reduce_field_kernel<REDOP>
        <<<num_blocks(rect.volume()), THREADS_PER_BLOCK>>>(acc, rect, args.value);
  } else {
    reduce_field_kernel<REDOP>
        <<<num_blocks(rect.volume()), THREADS_PER_BLOCK, 0, get_task_stream()>>>(
            acc, rect, args.value);
  }
}

static void gpu_mutate_region(Runtime *runtime, const IndexSpace &subspace,
                              const PhysicalRegion &region, PrivilegeMode privilege,
                              ReductionOpID redop, const std::vector<FieldID> &fields,
                              PointTaskArgs args) {
  Domain dom = runtime->get_index_space_domain(subspace);
  for (RectInDomainIterator<1> rit(dom); rit(); rit++) {
    Rect<1> rect = *rit;
    if ((privilege & LEGION_WRITE_ONLY) == LEGION_WRITE_ONLY) {
      for (FieldID fid : fields) {
        gpu_write_field(region, rect, fid, args);
      }
    } else if (privilege == LEGION_READ_WRITE) {
      for (FieldID fid : fields) {
        gpu_modify_field(region, rect, fid, args);
      }
    } else if (privilege == LEGION_REDUCE) {
      for (FieldID fid : fields) {
        switch (redop) {
          case LEGION_REDOP_SUM_UINT64: {
            gpu_reduce_field<SumReduction<uint64_t>>(region, rect, fid, redop, args);
          } break;
          case LEGION_REDOP_PROD_UINT64: {
            gpu_reduce_field<ProdReduction<uint64_t>>(region, rect, fid, redop, args);
          } break;
          case LEGION_REDOP_MIN_UINT64: {
            gpu_reduce_field<MinReduction<uint64_t>>(region, rect, fid, redop, args);
          } break;
          case LEGION_REDOP_MAX_UINT64: {
            gpu_reduce_field<MaxReduction<uint64_t>>(region, rect, fid, redop, args);
          } break;
          case LEGION_REDOP_AND_UINT64: {
            gpu_reduce_field<AndReduction<uint64_t>>(region, rect, fid, redop, args);
          } break;
          case LEGION_REDOP_OR_UINT64: {
            gpu_reduce_field<OrReduction<uint64_t>>(region, rect, fid, redop, args);
          } break;
          case LEGION_REDOP_XOR_UINT64: {
            gpu_reduce_field<XorReduction<uint64_t>>(region, rect, fid, redop, args);
          } break;
          default:
            abort();
        }
      }
    } else if ((privilege & LEGION_WRITE_PRIV) != 0) {
      abort();
    }
  }
}

static void gpu_task_body(const Task *task, const std::vector<PhysicalRegion> &regions,
                          Context ctx, Runtime *runtime) {
  const PointTaskArgs args = unpack_args<PointTaskArgs>(task);

  for (size_t idx = 0; idx < task->regions.size(); ++idx) {
    const RegionRequirement &req = task->regions[idx];
    gpu_mutate_region(runtime, req.region.get_index_space(), regions[idx], req.privilege,
                      req.redop, req.instance_fields, args);
  }
}

void void_leaf_gpu(const Task *task, const std::vector<PhysicalRegion> &regions,
                   Context ctx, Runtime *runtime) {
  gpu_task_body(task, regions, ctx, runtime);
}

uint64_t uint64_leaf_gpu(const Task *task, const std::vector<PhysicalRegion> &regions,
                         Context ctx, Runtime *runtime) {
  gpu_task_body(task, regions, ctx, runtime);
  return compute_task_result(task);
}

void void_replicable_leaf_gpu(const Task *task,
                              const std::vector<PhysicalRegion> &regions, Context ctx,
                              Runtime *runtime) {
  gpu_task_body(task, regions, ctx, runtime);
}

uint64_t uint64_replicable_leaf_gpu(const Task *task,
                                    const std::vector<PhysicalRegion> &regions,
                                    Context ctx, Runtime *runtime) {
  gpu_task_body(task, regions, ctx, runtime);
  return compute_task_result(task);
}
