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

#ifndef TASKS_H_
#define TASKS_H_

#include <cstdint>
#include <type_traits>
#include <vector>

#include "legion.h"

enum TaskIDs {
  VOID_LEAF_TASK_ID,
  UINT64_LEAF_TASK_ID,
  VOID_INNER_TASK_ID,
  UINT64_INNER_TASK_ID,
  VOID_REPLICABLE_LEAF_TASK_ID,
  UINT64_REPLICABLE_LEAF_TASK_ID,
  VOID_REPLICABLE_INNER_TASK_ID,
  UINT64_REPLICABLE_INNER_TASK_ID,
  COLOR_POINTS_TASK_ID,
  TOP_LEVEL_TASK_ID,
};

struct PointTaskArgs {
  PointTaskArgs() = delete;
  explicit PointTaskArgs(uint64_t _value, uint64_t _gpu_task_use_stream)
      : value(_value), gpu_task_use_stream(_gpu_task_use_stream) {}
  uint64_t value;
  // Hack: we're packing this as a uint64_t to avoid padding bytes in the
  // struct representation
  uint64_t gpu_task_use_stream;
};
static_assert(std::is_trivially_copyable_v<PointTaskArgs>);
static_assert(std::has_unique_object_representations_v<PointTaskArgs>);

template <typename T>
const T unpack_args(const Legion::Task *task) {
  if (task->arglen != sizeof(T)) {
    abort();
  }
  const T result = *reinterpret_cast<const T *>(task->args);
  return result;
}

uint64_t compute_task_result(const Legion::Task *task);

void mutate_region(Legion::Runtime *runtime, const Legion::IndexSpace &subspace,
                   const Legion::PhysicalRegion &region, Legion::PrivilegeMode privilege,
                   Legion::ReductionOpID redop,
                   const std::vector<Legion::FieldID> &fields, PointTaskArgs args);

void void_leaf(const Legion::Task *task,
               const std::vector<Legion::PhysicalRegion> &regions, Legion::Context ctx,
               Legion::Runtime *runtime);
uint64_t uint64_leaf(const Legion::Task *task,
                     const std::vector<Legion::PhysicalRegion> &regions,
                     Legion::Context ctx, Legion::Runtime *runtime);
void void_inner(const Legion::Task *task,
                const std::vector<Legion::PhysicalRegion> &regions, Legion::Context ctx,
                Legion::Runtime *runtime);
uint64_t uint64_inner(const Legion::Task *task,
                      const std::vector<Legion::PhysicalRegion> &regions,
                      Legion::Context ctx, Legion::Runtime *runtime);
void void_replicable_leaf(const Legion::Task *task,
                          const std::vector<Legion::PhysicalRegion> &regions,
                          Legion::Context ctx, Legion::Runtime *runtime);
uint64_t uint64_replicable_leaf(const Legion::Task *task,
                                const std::vector<Legion::PhysicalRegion> &regions,
                                Legion::Context ctx, Legion::Runtime *runtime);
void void_replicable_inner(const Legion::Task *task,
                           const std::vector<Legion::PhysicalRegion> &regions,
                           Legion::Context ctx, Legion::Runtime *runtime);
uint64_t uint64_replicable_inner(const Legion::Task *task,
                                 const std::vector<Legion::PhysicalRegion> &regions,
                                 Legion::Context ctx, Legion::Runtime *runtime);

#endif  // TASKS_H_
