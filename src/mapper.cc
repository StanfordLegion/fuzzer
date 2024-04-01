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

using namespace Legion;
using namespace Legion::Mapping;

enum MapperCallIDs {
  SELECT_TASKS_TO_MAP,
  MAP_TASK,
};

static Logger log_map("fuzz_mapper");

FuzzMapper::FuzzMapper(MapperRuntime* rt, Machine machine, RngStream st)
    : NullMapper(rt, machine),
      stream(st),
      select_tasks_to_map_channel(st.make_channel(int(SELECT_TASKS_TO_MAP))) {}

const char* FuzzMapper::get_mapper_name(void) const { return "fuzz_mapper"; }

Mapper::MapperSyncModel FuzzMapper::get_mapper_sync_model(void) const {
  return SERIALIZED_REENTRANT_MAPPER_MODEL;
}

void FuzzMapper::select_task_options(const MapperContext ctx, const Task& task,
                                     Mapper::TaskOptions& output) {
  // output.initial_proc = local_proc; // Leave the task where it is.
  output.inline_task = false;
  output.stealable = false;
  output.map_locally = false;  // TODO
  output.valid_instances = false;
  output.memoize = true;
  output.replicate = true;
  // output.parent_priority = ...; // Leave parent at current priority.
  // output.check_collective_regions.insert(...); // TODO
}

void FuzzMapper::replicate_task(MapperContext ctx, const Task& task,
                                const ReplicateTaskInput& input,
                                ReplicateTaskOutput& output) {
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

void FuzzMapper::map_task(const MapperContext ctx, const Task& task,
                          const MapTaskInput& input, MapTaskOutput& output) {
  // RngChannel rng = make_task_channel(MAP_TASK, task);

  // TODO: cache this?
  std::vector<VariantID> variants;
  runtime->find_valid_variants(ctx, task.task_id, variants);
  if (variants.size() != 1) {
    log_map.fatal() << "Bad variants in map_task: " << variants.size() << ", expected: 1";
    abort();
  }
  output.chosen_variant = variants.at(0);

  // TODO: default mapper does a query for each kind here; there is no way to
  // look up a variant's kind
  Machine::ProcessorQuery query(machine);
  query.only_kind(Processor::LOC_PROC);
  query.local_address_space();
  // TODO: should we randomize this?
  output.target_procs.insert(output.target_procs.end(), query.begin(), query.end());
}

void FuzzMapper::select_partition_projection(const MapperContext ctx,
                                             const Partition& partition,
                                             const SelectPartitionProjectionInput& input,
                                             SelectPartitionProjectionOutput& output) {
  if (!input.open_complete_partitions.empty())
    output.chosen_partition = input.open_complete_partitions.at(0);
  else
    output.chosen_partition = LogicalPartition::NO_PART;
}

void FuzzMapper::configure_context(const MapperContext ctx, const Task& task,
                                   ContextConfigOutput& output) {}

void FuzzMapper::select_tasks_to_map(const MapperContext ctx,
                                     const SelectMappingInput& input,
                                     SelectMappingOutput& output) {
  // Just to really mess with things, we'll pick a random task on every invokation.
  uint64_t target =
      select_tasks_to_map_channel.uniform_range(0, input.ready_tasks.size());
  auto it = input.ready_tasks.begin();
  for (uint64_t idx = 0; idx < target; ++idx) {
    if (it == input.ready_tasks.end()) {
      log_map.fatal() << "Out of bounds in select_tasks_to_map";
      abort();
    }
    ++it;
  }
  if (it == input.ready_tasks.end()) {
    log_map.fatal() << "Out of bounds in select_tasks_to_map";
    abort();
  }
  output.map_tasks.insert(*it);
}

void FuzzMapper::select_steal_targets(const MapperContext ctx,
                                      const SelectStealingInput& input,
                                      SelectStealingOutput& output) {}

RngChannel FuzzMapper::make_task_channel(int mapper_call,
                                         const Legion::Task& task) const {
  // TODO: index launches, mapper call ID
  static_assert(sizeof(MappingTagID) <= sizeof(uint64_t));
  return stream.make_channel(std::pair(mapper_call, uint64_t(task.tag)));
}
