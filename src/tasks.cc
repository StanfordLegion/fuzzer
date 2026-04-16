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

#include "tasks.h"

using namespace Legion;

static void write_field(const PhysicalRegion &region, Domain &dom, FieldID fid,
                        uint64_t value) {
  const FieldAccessor<LEGION_WRITE_ONLY, uint64_t, 1, coord_t,
                      Realm::AffineAccessor<uint64_t, 1, coord_t>>
      acc(region, fid);
  for (Domain::DomainPointIterator it(dom); it; ++it) {
    acc[*it] = value;
  }
}

static void modify_field(const PhysicalRegion &region, Domain &dom, FieldID fid,
                         uint64_t value) {
  const FieldAccessor<LEGION_READ_WRITE, uint64_t, 1, coord_t,
                      Realm::AffineAccessor<uint64_t, 1, coord_t>>
      acc(region, fid);
  for (Domain::DomainPointIterator it(dom); it; ++it) {
    acc[*it] = (acc[*it] >> 1) + value;
  }
}

template <typename REDOP>
static void reduce_field(const PhysicalRegion &region, Domain &dom, FieldID fid,
                         ReductionOpID redop, uint64_t value) {
  const ReductionAccessor<REDOP, true /* exclusive */, 1, coord_t,
                          Realm::AffineAccessor<uint64_t, 1, coord_t>>
      acc(region, fid, redop);
  for (Domain::DomainPointIterator it(dom); it; ++it) {
    acc[*it] <<= value;
  }
}

void mutate_region(Runtime *runtime, const IndexSpace &subspace,
                   const PhysicalRegion &region, PrivilegeMode privilege,
                   ReductionOpID redop, const std::vector<FieldID> &fields,
                   PointTaskArgs args) {
  Domain dom = runtime->get_index_space_domain(subspace);
  if ((privilege & LEGION_WRITE_ONLY) == LEGION_WRITE_ONLY) {
    for (FieldID fid : fields) {
      write_field(region, dom, fid, args.value);
    }
  } else if (privilege == LEGION_READ_WRITE) {
    for (FieldID fid : fields) {
      modify_field(region, dom, fid, args.value);
    }
  } else if (privilege == LEGION_REDUCE) {
    for (FieldID fid : fields) {
      switch (redop) {
        case LEGION_REDOP_SUM_UINT64: {
          reduce_field<SumReduction<uint64_t>>(region, dom, fid, redop, args.value);
        } break;
        case LEGION_REDOP_PROD_UINT64: {
          reduce_field<ProdReduction<uint64_t>>(region, dom, fid, redop, args.value);
        } break;
        case LEGION_REDOP_MIN_UINT64: {
          reduce_field<MinReduction<uint64_t>>(region, dom, fid, redop, args.value);
        } break;
        case LEGION_REDOP_MAX_UINT64: {
          reduce_field<MaxReduction<uint64_t>>(region, dom, fid, redop, args.value);
        } break;
        case LEGION_REDOP_AND_UINT64: {
          reduce_field<AndReduction<uint64_t>>(region, dom, fid, redop, args.value);
        } break;
        case LEGION_REDOP_OR_UINT64: {
          reduce_field<OrReduction<uint64_t>>(region, dom, fid, redop, args.value);
        } break;
        case LEGION_REDOP_XOR_UINT64: {
          reduce_field<XorReduction<uint64_t>>(region, dom, fid, redop, args.value);
        } break;
        default:
          abort();
      }
    }
  } else if ((privilege & LEGION_WRITE_PRIV) != 0) {
    // We'd better not get here with write privileges.
    abort();
  }
}

static void task_body(const Task *task, const std::vector<PhysicalRegion> &regions,
                      Context ctx, Runtime *runtime) {
  const PointTaskArgs args = unpack_args<PointTaskArgs>(task);

  for (size_t idx = 0; idx < task->regions.size(); ++idx) {
    const RegionRequirement &req = task->regions[idx];
    mutate_region(runtime, req.region.get_index_space(), regions[idx], req.privilege,
                  req.redop, req.instance_fields, args);
  }
}

uint64_t compute_task_result(const Task *task) {
  const PointTaskArgs args = unpack_args<PointTaskArgs>(task);
  // Twiddle the bits a bit to make some interesting bit patterns
  uint64_t point = 0;
  if (task->is_index_space) {
    point = task->index_point[0];
  }
  return point ^ args.value;
}

static void inner_task_body(const Task *task, const std::vector<PhysicalRegion> &regions,
                            Context ctx, Runtime *runtime) {
  Future result;
  for (size_t idx = 0; idx < task->regions.size(); ++idx) {
    TaskLauncher launcher(VOID_LEAF_TASK_ID, TaskArgument(task->args, task->arglen));
    launcher.tag = task->tag;
    const RegionRequirement &req = task->regions[idx];
    if (req.privilege == LEGION_REDUCE) {
      launcher.add_region_requirement(RegionRequirement(req.region, req.privilege_fields,
                                                        req.instance_fields, req.redop,
                                                        LEGION_EXCLUSIVE, req.region));
    } else {
      launcher.add_region_requirement(
          RegionRequirement(req.region, req.privilege_fields, req.instance_fields,
                            req.privilege, LEGION_EXCLUSIVE, req.region));
    }
    runtime->execute_task(ctx, launcher);
  }
}

void void_leaf(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx,
               Runtime *runtime) {
  task_body(task, regions, ctx, runtime);
}

uint64_t uint64_leaf(const Task *task, const std::vector<PhysicalRegion> &regions,
                     Context ctx, Runtime *runtime) {
  task_body(task, regions, ctx, runtime);
  return compute_task_result(task);
}

void void_inner(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx,
                Runtime *runtime) {
  inner_task_body(task, regions, ctx, runtime);
}

uint64_t uint64_inner(const Task *task, const std::vector<PhysicalRegion> &regions,
                      Context ctx, Runtime *runtime) {
  inner_task_body(task, regions, ctx, runtime);
  return compute_task_result(task);
}

void void_replicable_leaf(const Task *task, const std::vector<PhysicalRegion> &regions,
                          Context ctx, Runtime *runtime) {
  task_body(task, regions, ctx, runtime);
}

uint64_t uint64_replicable_leaf(const Task *task,
                                const std::vector<PhysicalRegion> &regions, Context ctx,
                                Runtime *runtime) {
  task_body(task, regions, ctx, runtime);
  return compute_task_result(task);
}

void void_replicable_inner(const Task *task, const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime) {
  inner_task_body(task, regions, ctx, runtime);
}

uint64_t uint64_replicable_inner(const Task *task,
                                 const std::vector<PhysicalRegion> &regions, Context ctx,
                                 Runtime *runtime) {
  inner_task_body(task, regions, ctx, runtime);
  return compute_task_result(task);
}
