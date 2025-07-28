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

#ifndef MAPPER_H_
#define MAPPER_H_

#include "deterministic_random.h"
#include "legion.h"
#include "null_mapper.h"

namespace FuzzMapper {
// Because C++ won't let us put this in the class, we have to create an
// entire namespace to scope these
using namespace Legion;
using namespace Legion::Mapping;

class FuzzMapper : public NullMapper {
public:
  FuzzMapper(MapperRuntime *runtime, Machine machine, Processor local, RngStream stream,
             uint64_t replicate);

public:
  const char *get_mapper_name(void) const override;
  MapperSyncModel get_mapper_sync_model(void) const override;

public:  // Task mapping calls
  void select_task_options(const MapperContext ctx, const Task &task,
                           TaskOptions &output) override;
  void premap_task(const MapperContext ctx, const Task &task,
                   const PremapTaskInput &input, PremapTaskOutput &output) override;
  void slice_task(const MapperContext ctx, const Task &task, const SliceTaskInput &input,
                  SliceTaskOutput &output) override;
  void map_task(const MapperContext ctx, const Task &task, const MapTaskInput &input,
                MapTaskOutput &output) override;
  void replicate_task(MapperContext ctx, const Task &task,
                      const ReplicateTaskInput &input,
                      ReplicateTaskOutput &output) override;
  void select_task_sources(const MapperContext ctx, const Task &task,
                           const SelectTaskSrcInput &input,
                           SelectTaskSrcOutput &output) override;
  void select_sharding_functor(const MapperContext ctx, const Task &task,
                               const SelectShardingFunctorInput &input,
                               SelectShardingFunctorOutput &output) override;

public:  // Inline mapping calls
  void map_inline(const MapperContext ctx, const InlineMapping &inline_op,
                  const MapInlineInput &input, MapInlineOutput &output) override;
  void select_inline_sources(const MapperContext ctx, const InlineMapping &inline_op,
                             const SelectInlineSrcInput &input,
                             SelectInlineSrcOutput &output) override;

public:  // Partition mapping calls
  void select_partition_projection(const MapperContext ctx, const Partition &partition,
                                   const SelectPartitionProjectionInput &input,
                                   SelectPartitionProjectionOutput &output) override;
  void map_partition(const MapperContext ctx, const Partition &partition,
                     const MapPartitionInput &input, MapPartitionOutput &output) override;

public:  // Task execution mapping calls
  void configure_context(const MapperContext ctx, const Task &task,
                         ContextConfigOutput &output) override;

public:  // Mapping control and stealing
  void select_tasks_to_map(const MapperContext ctx, const SelectMappingInput &input,
                           SelectMappingOutput &output) override;
  void select_steal_targets(const MapperContext ctx, const SelectStealingInput &input,
                            SelectStealingOutput &output) override;

private:
  RngChannel make_task_channel(int32_t mapper_call, const Task &task,
                               uint64_t salt = 0) const;

  Processor random_local_proc(RngChannel &rng);
  Processor random_global_proc(RngChannel &rng);

  void random_mapping(const MapperContext ctx, RngChannel &rng,
                      const RegionRequirement &req,
                      std::vector<PhysicalInstance> &output);

  void random_sources(RngChannel &rng,
                      const std::vector<PhysicalInstance> &source_instances,
                      std::deque<PhysicalInstance> &chosen_ranking);

private:
  RngStream stream;
  RngChannel select_tasks_to_map_channel;
  RngChannel map_inline_channel;
  RngChannel select_inline_sources_channel;

  Processor local_proc;
  std::vector<Processor> local_procs;
  std::vector<Processor> global_procs;

  uint64_t replicate_levels;
};

}  // namespace FuzzMapper

#endif  // MAPPER_H_
