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

class FuzzMapper : public Legion::Mapping::NullMapper {
public:
  FuzzMapper(Legion::Mapping::MapperRuntime* runtime, Legion::Machine machine,
             RngStream stream);

public:
  const char* get_mapper_name(void) const override;
  MapperSyncModel get_mapper_sync_model(void) const override;

public:  // Task mapping calls
  void select_task_options(const Legion::Mapping::MapperContext ctx,
                           const Legion::Task& task, TaskOptions& output) override;
  void replicate_task(Legion::Mapping::MapperContext ctx, const Legion::Task& task,
                      const ReplicateTaskInput& input,
                      ReplicateTaskOutput& output) override;
  void map_task(const Legion::Mapping::MapperContext ctx, const Legion::Task& task,
                const MapTaskInput& input, MapTaskOutput& output) override;

public:  // Partition mapping calls
  void select_partition_projection(const Legion::Mapping::MapperContext ctx,
                                   const Legion::Partition& partition,
                                   const SelectPartitionProjectionInput& input,
                                   SelectPartitionProjectionOutput& output) override;

public:  // Task execution mapping calls
  void configure_context(const Legion::Mapping::MapperContext ctx,
                         const Legion::Task& task, ContextConfigOutput& output) override;

public:  // Mapping control and stealing
  void select_tasks_to_map(const Legion::Mapping::MapperContext ctx,
                           const SelectMappingInput& input,
                           SelectMappingOutput& output) override;
  void select_steal_targets(const Legion::Mapping::MapperContext ctx,
                            const SelectStealingInput& input,
                            SelectStealingOutput& output) override;

private:
  RngChannel make_task_channel(int mapper_call, const Legion::Task& task) const;

private:
  RngStream stream;
  RngChannel select_tasks_to_map_channel;
};

#endif  // MAPPER_H_
