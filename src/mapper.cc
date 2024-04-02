/* Copyright 2024 Stanford University
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

#include "mapper.h"

namespace FuzzMapper {
using namespace Legion;
using namespace Legion::Mapping;

enum MapperCallIDs {
  SELECT_TASKS_TO_MAP,
  SLICE_TASK,
  MAP_TASK,
  SELECT_TASK_SOURCES,
  MAP_INLINE,
  SELECT_INLINE_SOURCES,
};

static Logger log_map("fuzz_mapper");

FuzzMapper::FuzzMapper(MapperRuntime *rt, Machine machine, Processor local, RngStream st)
    : NullMapper(rt, machine),
      stream(st),
      select_tasks_to_map_channel(st.make_channel(int32_t(SELECT_TASKS_TO_MAP))),
      map_inline_channel(st.make_channel(int32_t(MAP_INLINE))),
      select_inline_sources_channel(st.make_channel(int32_t(SELECT_INLINE_SOURCES))),
      local_proc(local) {
  // TODO: something other than CPU processor
  {
    Machine::ProcessorQuery query(machine);
    query.only_kind(Processor::LOC_PROC);
    query.local_address_space();
    local_procs.insert(local_procs.end(), query.begin(), query.end());
  }
  {
    Machine::ProcessorQuery query(machine);
    query.only_kind(Processor::LOC_PROC);
    global_procs.insert(global_procs.end(), query.begin(), query.end());
  }
}

const char *FuzzMapper::get_mapper_name(void) const { return "fuzz_mapper"; }

Mapper::MapperSyncModel FuzzMapper::get_mapper_sync_model(void) const {
  return SERIALIZED_REENTRANT_MAPPER_MODEL;
}

void FuzzMapper::select_task_options(const MapperContext ctx, const Task &task,
                                     Mapper::TaskOptions &output) {
  // output.initial_proc = local_proc; // Leave the task where it is.
  output.inline_task = false;
  output.stealable = false;
  output.map_locally = false;  // TODO
  output.valid_instances = false;
  output.memoize = true;
  output.replicate = task.get_depth() == 0;  // TODO: replicate other tasks
  // output.parent_priority = ...; // Leave parent at current priority.
  // output.check_collective_regions.insert(...); // TODO
}

void FuzzMapper::premap_task(const MapperContext ctx, const Task &task,
                             const PremapTaskInput &input, PremapTaskOutput &output) {
  // TODO: premap futures
}

void FuzzMapper::slice_task(const MapperContext ctx, const Task &task,
                            const SliceTaskInput &input, SliceTaskOutput &output) {
  RngChannel rng = make_task_channel(SLICE_TASK, task);

  bool local = rng.uniform_range(0, 1) == 0;
  for (Domain::DomainPointIterator it(input.domain); it; ++it) {
    Processor proc = local ? random_local_proc(rng) : random_global_proc(rng);
    output.slices.push_back(TaskSlice(Domain::from_domain_point(*it), proc,
                                      false /* recurse */, false /* stealable */));
  }
}

void FuzzMapper::map_task(const MapperContext ctx, const Task &task,
                          const MapTaskInput &input, MapTaskOutput &output) {
  RngChannel rng = make_task_channel(MAP_TASK, task);

  log_map.debug() << "map_task: Start";

  // TODO: cache this?
  std::vector<VariantID> variants;
  runtime->find_valid_variants(ctx, task.task_id, variants);
  if (variants.size() != 1) {
    log_map.fatal() << "Bad variants in map_task: " << variants.size() << ", expected: 1";
    abort();
  }
  output.chosen_variant = variants.at(0);
  log_map.debug() << "map_task: Selected variant " << output.chosen_variant;

  // TODO: assign to variant's correct processor kind
  if (rng.uniform_range(0, 1) == 0) {
    log_map.debug() << "map_task: Mapping to all local procs";
    output.target_procs.insert(output.target_procs.end(), local_procs.begin(),
                               local_procs.end());
  } else {
    log_map.debug() << "map_task: Mapping to current proc";
    output.target_procs.push_back(local_proc);
  }

  if (runtime->is_inner_variant(ctx, task.task_id, output.chosen_variant)) {
    // For inner variants we'll always select virtual instances
    for (size_t idx = 0; idx < task.regions.size(); ++idx) {
      output.chosen_instances.at(idx).push_back(PhysicalInstance::get_virtual_instance());
    }
  } else {
    for (size_t idx = 0; idx < task.regions.size(); ++idx) {
      const RegionRequirement &req = task.regions.at(idx);
      random_mapping(ctx, rng, req, output.chosen_instances.at(idx));
    }
  }
}

void FuzzMapper::replicate_task(MapperContext ctx, const Task &task,
                                const ReplicateTaskInput &input,
                                ReplicateTaskOutput &output) {
  // TODO: cache this?
  std::vector<VariantID> variants;
  runtime->find_valid_variants(ctx, task.task_id, variants);
  if (variants.size() != 1) {
    log_map.fatal() << "Bad variants in replicate_task: " << variants.size()
                    << ", expected: 1";
    abort();
  }
  output.chosen_variant = variants.at(0);
  // TODO: actually replicate
}

void FuzzMapper::select_task_sources(const MapperContext ctx, const Task &task,
                                     const SelectTaskSrcInput &input,
                                     SelectTaskSrcOutput &output) {
  RngChannel rng = make_task_channel(SELECT_TASK_SOURCES, task);
  random_sources(rng, input.source_instances, output.chosen_ranking);
}

void FuzzMapper::map_inline(const MapperContext ctx, const InlineMapping &inline_op,
                            const MapInlineInput &input, MapInlineOutput &output) {
  RngChannel &rng = map_inline_channel;
  const RegionRequirement &req = inline_op.requirement;
  random_mapping(ctx, rng, req, output.chosen_instances);
}

void FuzzMapper::select_inline_sources(const MapperContext ctx,
                                       const InlineMapping &inline_op,
                                       const SelectInlineSrcInput &input,
                                       SelectInlineSrcOutput &output) {
  RngChannel &rng = select_inline_sources_channel;
  random_sources(rng, input.source_instances, output.chosen_ranking);
}

void FuzzMapper::select_partition_projection(const MapperContext ctx,
                                             const Partition &partition,
                                             const SelectPartitionProjectionInput &input,
                                             SelectPartitionProjectionOutput &output) {
  if (!input.open_complete_partitions.empty())
    output.chosen_partition = input.open_complete_partitions.at(0);
  else
    output.chosen_partition = LogicalPartition::NO_PART;
}

void FuzzMapper::map_partition(const MapperContext ctx, const Partition &partition,
                               const MapPartitionInput &input,
                               MapPartitionOutput &output) {
  output.chosen_instances = input.valid_instances;
  if (!output.chosen_instances.empty())
    runtime->acquire_and_filter_instances(ctx, output.chosen_instances);
}

void FuzzMapper::configure_context(const MapperContext ctx, const Task &task,
                                   ContextConfigOutput &output) {}

void FuzzMapper::select_tasks_to_map(const MapperContext ctx,
                                     const SelectMappingInput &input,
                                     SelectMappingOutput &output) {
  RngChannel &rng = select_tasks_to_map_channel;

  log_map.debug() << "select_tasks_to_map: Start";
  if (input.ready_tasks.empty()) {
    log_map.fatal() << "select_tasks_to_map: Empty ready list";
    abort();
  }

  // Just to really mess with things, we'll pick a random task on every invokation.
  uint64_t target = rng.uniform_range(0, input.ready_tasks.size() - 1);
  log_map.debug() << "select_tasks_to_map: Selected task " << target << " of "
                  << input.ready_tasks.size();
  auto it = input.ready_tasks.begin();
  std::advance(it, target);
  if (it == input.ready_tasks.end()) {
    log_map.fatal() << "select_tasks_to_map: Out of bounds ";
    abort();
  }

  // Choose where to send the task.
  // IMPORTANT: in order for this to be deterministic, we need to use
  // the task channel to decide where it should go.
  RngChannel task_rng = make_task_channel(SELECT_TASKS_TO_MAP, **it);
  switch (task_rng.uniform_range(0, 3)) {
    case 0:
    case 1: {
      output.map_tasks.insert(*it);
    } break;
    case 2: {
      output.relocate_tasks[*it] = random_local_proc(rng);
    } break;
    case 3: {
      output.relocate_tasks[*it] = random_global_proc(rng);
    } break;
    default:
      abort();
  }
}

void FuzzMapper::select_steal_targets(const MapperContext ctx,
                                      const SelectStealingInput &input,
                                      SelectStealingOutput &output) {}

RngChannel FuzzMapper::make_task_channel(int mapper_call,
                                         const Legion::Task &task) const {
  // TODO: index launches, mapper call ID
  static_assert(sizeof(MappingTagID) <= sizeof(uint64_t));
  return stream.make_channel(std::pair(mapper_call, uint64_t(task.tag)));
}

Processor FuzzMapper::random_local_proc(RngChannel &rng) {
  return local_procs.at(rng.uniform_range(0, local_procs.size() - 1));
}

Processor FuzzMapper::random_global_proc(RngChannel &rng) {
  return global_procs.at(rng.uniform_range(0, global_procs.size() - 1));
}

void FuzzMapper::random_mapping(const MapperContext ctx, RngChannel &rng,
                                const RegionRequirement &req,
                                std::vector<PhysicalInstance> &output) {
  if (req.privilege == LEGION_NO_ACCESS || req.privilege_fields.empty()) {
    return;
  }

  // Pick the memory this is going into
  Memory memory;
  {
    Machine::MemoryQuery query(machine);
    query.has_affinity_to(local_proc);
    query.has_capacity(1);
    uint64_t target = rng.uniform_range(0, query.count() - 1);
    auto it = query.begin();
    std::advance(it, target);
    memory = *it;
  }
  log_map.debug() << "random_mapping: Selected memory " << memory << " kind "
                  << memory.kind();

  LayoutConstraintSet constraints;
  if (req.privilege == LEGION_REDUCE) {
    constraints.add_constraint(
        SpecializedConstraint(LEGION_AFFINE_REDUCTION_SPECIALIZE, req.redop));
  } else {
    constraints.add_constraint(SpecializedConstraint());
  }

  constraints.add_constraint(MemoryConstraint(memory.kind()));

  {
    std::vector<FieldID> fields;
    if (rng.uniform_range(0, 1) == 0) {
      FieldSpace handle = req.region.get_field_space();
      runtime->get_field_space_fields(ctx, handle, fields);
    } else {
      fields.insert(fields.end(), req.instance_fields.begin(), req.instance_fields.end());
    }
    bool contiguous = rng.uniform_range(0, 1) == 0;
    bool inorder = rng.uniform_range(0, 1) == 0;

    constraints.add_constraint(FieldConstraint(fields, contiguous, inorder));
  }

  {
    IndexSpace is = req.region.get_index_space();
    Domain domain = runtime->get_index_space_domain(ctx, is);
    int dim = domain.get_dim();
    std::vector<DimensionKind> dimension_ordering(dim + 1);
    for (int i = 0; i < dim; ++i)
      dimension_ordering.at(i) =
          static_cast<DimensionKind>(static_cast<int>(LEGION_DIM_X) + i);
    dimension_ordering[dim] = LEGION_DIM_F;
    rng.shuffle(dimension_ordering);
    bool contiguous = rng.uniform_range(0, 1) == 0;
    constraints.add_constraint(OrderingConstraint(dimension_ordering, contiguous));
  }

  // Coarsen the region by a random amount by walking up the region tree
  LogicalRegion region = req.region;
  while (runtime->has_parent_logical_partition(ctx, region) &&
         rng.uniform_range(0, 1) == 0) {
    LogicalPartition parent = runtime->get_parent_logical_partition(ctx, region);
    region = runtime->get_parent_logical_region(ctx, parent);
  }
  std::vector<LogicalRegion> regions = {region};

  // Either force the runtime to create a fresh instance, or allow one to be reused
  PhysicalInstance instance;
  if (rng.uniform_range(0, 1) == 0) {
    if (!runtime->create_physical_instance(ctx, memory, constraints, regions, instance,
                                           true /* acquire */, LEGION_GC_MAX_PRIORITY)) {
      log_map.fatal() << "random_mapping: Failed to create instance";
      abort();
    }
  } else {
    bool created;
    if (!runtime->find_or_create_physical_instance(ctx, memory, constraints, regions,
                                                   instance, created, true /* acquire */,
                                                   LEGION_GC_NEVER_PRIORITY)) {
      log_map.fatal() << "random_mapping: Failed to create instance";
      abort();
    }
  }
  output.push_back(instance);
}

void FuzzMapper::random_sources(RngChannel &rng,
                                const std::vector<PhysicalInstance> &source_instances,
                                std::deque<PhysicalInstance> &chosen_ranking) {
  std::vector<PhysicalInstance> sources(source_instances.begin(), source_instances.end());
  rng.shuffle(sources);
  chosen_ranking.insert(chosen_ranking.end(), sources.begin(), sources.end());
}

}  // namespace FuzzMapper
