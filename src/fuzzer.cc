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

#include <cstdint>
#include <iostream>
#include <set>
#include <string>

#include "legion.h"
#include "siphash.h"

using namespace Legion;

enum TaskIDs {
  VOID_LEAF_TASK_ID,
  INT64_LEAF_TASK_ID,
  VOID_INNER_TASK_ID,
  INT64_INNER_TASK_ID,
  VOID_REPLICABLE_LEAF_TASK_ID,
  INT64_REPLICABLE_LEAF_TASK_ID,
  VOID_REPLICABLE_INNER_TASK_ID,
  INT64_REPLICABLE_INNER_TASK_ID,
  TOP_LEVEL_TASK_ID,
};

#define LOG_ONCE(x) runtime->log_once(ctx, (x))

static Logger log_fuzz("fuzz");

static long long parse_long_long(const std::string &flag, const std::string &arg) {
  long long result;
  size_t consumed;
  result = std::stoll(arg, &consumed);
  if (consumed != arg.size()) {
    log_fuzz.error() << "error in parsing flag: " << flag << " " << arg;
    abort();
  }
  return result;
}

static uint64_t parse_uint64_t(const std::string &flag, const std::string &arg) {
  long long result = parse_long_long(flag, arg);
  if (result < 0) {
    log_fuzz.error() << "error in parsing flag: " << flag << " " << arg
                     << " (value is negative)";
    abort();
  }
  return result;
}

struct FuzzerConfig {
  uint64_t initial_seed;
  uint64_t region_tree_depth;
  uint64_t region_tree_width;
  uint64_t region_tree_branch_factor;
  uint64_t region_tree_num_fields;
  uint64_t num_ops;
  FuzzerConfig()
      : initial_seed(0),
        region_tree_depth(1),
        region_tree_width(4),
        region_tree_branch_factor(1),
        region_tree_num_fields(4),
        num_ops(1) {}

  static FuzzerConfig parse_args(int argc, char **argv) {
    FuzzerConfig config;
    for (int i = 1; i < argc; i++) {
      std::string flag(argv[i]);
      if (flag == "-fuzz:seed") {
        std::string arg(argv[++i]);
        config.initial_seed = parse_uint64_t(flag, arg);
      } else if (flag == "-fuzz:depth") {
        std::string arg(argv[++i]);
        config.region_tree_depth = parse_uint64_t(flag, arg);
      } else if (flag == "-fuzz:width") {
        std::string arg(argv[++i]);
        config.region_tree_width = parse_uint64_t(flag, arg);
      } else if (flag == "-fuzz:branch") {
        std::string arg(argv[++i]);
        config.region_tree_branch_factor = parse_uint64_t(flag, arg);
      } else if (flag == "-fuzz:fields") {
        std::string arg(argv[++i]);
        config.region_tree_num_fields = parse_uint64_t(flag, arg);
      } else if (flag == "-fuzz:ops") {
        std::string arg(argv[++i]);
        config.num_ops = parse_uint64_t(flag, arg);
      }
    }
    return config;
  }

  void log_config(Runtime *runtime, Context ctx) {
    LOG_ONCE(log_fuzz.print() << "Fuzzer Configuration:");
    LOG_ONCE(log_fuzz.print() << "  config.initial_seed = " << initial_seed);
    LOG_ONCE(log_fuzz.print() << "  config.region_tree_depth = " << region_tree_depth);
    LOG_ONCE(log_fuzz.print() << "  config.region_tree_width = " << region_tree_width);
    LOG_ONCE(log_fuzz.print() << "  config.region_tree_branch_factor = "
                              << region_tree_branch_factor);
    LOG_ONCE(log_fuzz.print() << "  config.region_tree_num_fields = "
                              << region_tree_num_fields);
    LOG_ONCE(log_fuzz.print() << "  config.num_ops = " << num_ops);
  }
};

static void gen_bits(const uint8_t *input, size_t input_bytes, uint8_t *output,
                     size_t output_bytes) {
  // To generate deterministic uniformly distributed bits, run a hash
  // function on the seed and use the hash value as the output.
  const uint8_t k[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  siphash(input, input_bytes, k, output, output_bytes);
}

static uint64_t uniform_uint64_t(uint64_t seed, uint64_t sequence_number) {
  const uint64_t input[2] = {seed, sequence_number};
  uint64_t result;
  gen_bits(reinterpret_cast<const uint8_t *>(&input), sizeof(input),
           reinterpret_cast<uint8_t *>(&result), sizeof(result));
  return result;
}

static bool is_power_of_2(uint64_t value) { return (value & (value - 1)) == 0; }

static uint64_t uniform_range(uint64_t seed, uint64_t sequence_number, uint64_t range_lo,
                              uint64_t range_hi /* exclusive */) {
  if (range_hi <= range_lo) {
    return range_lo;
  }

  uint64_t range_size = range_hi - range_lo;
  if (!is_power_of_2(range_size)) {
    // TODO: make this work on non-power-of-2 range sizes.
    // (The naive implementation is biased.)
    abort();
  }

  uint64_t random = uniform_uint64_t(seed, sequence_number);
  return range_lo + (random % range_size);
}

void void_leaf(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx,
               Runtime *runtime) {}

int64_t int64_leaf(const Task *task, const std::vector<PhysicalRegion> &regions,
                   Context ctx, Runtime *runtime) {
  return 1;
}

void void_inner(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx,
                Runtime *runtime) {}

int64_t int64_inner(const Task *task, const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime) {
  return 2;
}

void void_replicable_leaf(const Task *task, const std::vector<PhysicalRegion> &regions,
                          Context ctx, Runtime *runtime) {}

int64_t int64_replicable_leaf(const Task *task,
                              const std::vector<PhysicalRegion> &regions, Context ctx,
                              Runtime *runtime) {
  return 3;
}

void void_replicable_inner(const Task *task, const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime) {}

int64_t int64_replicable_inner(const Task *task,
                               const std::vector<PhysicalRegion> &regions, Context ctx,
                               Runtime *runtime) {
  return 4;
}

template <int N>
class RegionForest {
public:
  RegionForest(Runtime *_runtime, Context _ctx, const FuzzerConfig &config)
      : runtime(_runtime), ctx(_ctx) {
    ispace = runtime->create_index_space<1>(
        ctx, Rect<1>(Point<1>(0), Point<1>(config.region_tree_width)));
    fspace = runtime->create_field_space(ctx);
    {
      FieldAllocator falloc = runtime->create_field_allocator(ctx, fspace);
      for (uint64_t field = 0; field < config.region_tree_num_fields; ++field) {
        falloc.allocate_field(sizeof(uint64_t), field);
      }
    }
    root = runtime->create_logical_region(ctx, ispace, fspace);
    ipart = runtime->create_equal_partition(ctx, ispace, ispace);
    lpart = runtime->get_logical_partition(root, ipart);

    // Initialize everything so we don't get unitialized read warnings
    for (uint64_t field = 0; field < config.region_tree_num_fields; ++field) {
      runtime->fill_field<uint64_t>(ctx, root, root, field, field);
    }
  }

  ~RegionForest() {
    runtime->destroy_logical_region(ctx, root);
    runtime->destroy_field_space(ctx, fspace);
    runtime->destroy_index_space(ctx, ispace);
  }

  LogicalRegion get_root() const { return root; }

  LogicalPartition get_disjoint_partition() const { return lpart; }

private:
  Runtime *runtime;
  Context ctx;
  IndexSpaceT<N> ispace;
  FieldSpace fspace;
  LogicalRegion root;
  IndexPartition ipart;
  LogicalPartition lpart;
};

class Operation {
public:
  Operation(Runtime *_runtime, Context _ctx, const FuzzerConfig &_config,
            const RegionForest<1> &_forest)
      : runtime(_runtime),
        ctx(_ctx),
        config(_config),
        forest(_forest),
        task_id(LEGION_MAX_APPLICATION_TASK_ID),
        task_produces_value(false),
        launch_complete(false),
        range_min(0),
        range_max(0),
        range_size(0),
        launch_domain(Rect<1>::make_empty()),
        scalar_redop(LEGION_REDOP_LAST),
        scalar_reduction_ordered(false),
        elide_future_return(false),
        privilege(LEGION_NO_ACCESS),
        redop(LEGION_REDOP_LAST),
        launch_type(0) {}

  void select_task_id(const uint64_t seed, uint64_t &seq) {
    switch (uniform_range(seed, seq++, 0, 4) & 3) {
      case 0: {
        task_id = VOID_LEAF_TASK_ID;
        task_produces_value = false;
      } break;
      case 1: {
        task_id = VOID_INNER_TASK_ID;
        task_produces_value = false;
      } break;
      case 2: {
        task_id = INT64_LEAF_TASK_ID;
        task_produces_value = true;
      } break;
      case 3: {
        task_id = INT64_INNER_TASK_ID;
        task_produces_value = true;
      } break;
      default:
        abort();
    }
    LOG_ONCE(log_fuzz.info() << "  Task ID: " << task_id);
  }

  void select_launch_domain(const uint64_t seed, uint64_t &seq) {
    // A lot of Legion algorithms hinge on whether a launch is
    // complete or not, so we'll make that a special case here.

    launch_complete = uniform_range(seed, seq++, 0, 2) == 0;

    if (launch_complete) {
      range_min = 0;
      range_max = config.region_tree_width;
    } else {
      range_min = uniform_range(seed, seq++, 0, config.region_tree_width);
      range_max = uniform_range(seed, seq++, 0, config.region_tree_width);
      // Make sure the range is always non-empty.
      if (range_max < range_min) {
        std::swap(range_min, range_max);
      }
    }

    range_size = range_max - range_min + 1;
    launch_domain = Rect<1>(Point<1>(range_min), Point<1>(range_max));
    LOG_ONCE(log_fuzz.info() << "  Launch domain: " << launch_domain);
  }

  void select_scalar_reduction(const uint64_t seed, uint64_t &seq) {
    scalar_redop = LEGION_REDOP_LAST;
    if (task_produces_value) {
      switch (uniform_range(seed, seq++, 0, 4)) {
        case 0: {
          scalar_redop = LEGION_REDOP_SUM_INT64;
        } break;
        case 1: {
          scalar_redop = LEGION_REDOP_PROD_INT64;
        } break;
        case 2: {
          scalar_redop = LEGION_REDOP_MIN_INT64;
        } break;
        case 3: {
          scalar_redop = LEGION_REDOP_MAX_INT64;
        } break;
        default:
          abort();
      }
      LOG_ONCE(log_fuzz.info() << "  Scalar redop: " << scalar_redop);

      scalar_reduction_ordered = uniform_range(seed, seq++, 0, 2) == 0;
      LOG_ONCE(log_fuzz.info() << "    Ordered: " << scalar_reduction_ordered);
    }
  }

  void select_elide_future_return(const uint64_t seed, uint64_t &seq) {
    elide_future_return = uniform_range(seed, seq++, 0, 2) == 0;
    LOG_ONCE(log_fuzz.info() << "  Elide future return: " << elide_future_return);
  }

  void select_fields(const uint64_t seed, uint64_t &seq) {
    fields.clear();

    uint64_t field_set =
        uniform_range(seed, seq++, 0, 1 << config.region_tree_num_fields);
    LOG_ONCE(log_fuzz.info() << "  Field set: 0x" << std::hex << field_set);
    for (uint64_t field = 0; field < config.region_tree_num_fields; ++field) {
      if ((field_set & (1 << field)) != 0) {
        fields.push_back(field);
      }
    }
  }

  void select_privilege(const uint64_t seed, uint64_t &seq) {
    switch (uniform_range(seed, seq++, 0, 4)) {
      case 0: {
        privilege = LEGION_READ_ONLY;
      } break;
      case 1: {
        privilege = LEGION_READ_WRITE;
      } break;
      case 2: {
        privilege = LEGION_WRITE_DISCARD;
      } break;
      case 3: {
        privilege = LEGION_REDUCE;
      } break;
      default:
        abort();
    }
    LOG_ONCE(log_fuzz.info() << "  Privilege: 0x" << std::hex << privilege);
  }

  void select_reduction(const uint64_t seed, uint64_t &seq) {
    redop = LEGION_REDOP_LAST;
    if (privilege == LEGION_REDUCE) {
      switch (uniform_range(seed, seq++, 0, 4)) {
        case 0: {
          redop = LEGION_REDOP_SUM_INT64;
        } break;
        case 1: {
          redop = LEGION_REDOP_PROD_INT64;
        } break;
        case 2: {
          redop = LEGION_REDOP_MIN_INT64;
        } break;
        case 3: {
          redop = LEGION_REDOP_MAX_INT64;
        } break;
        default:
          abort();
      }
      LOG_ONCE(log_fuzz.info() << "  Region redop: " << redop);
    }
  }

  void select_launch_type(const uint64_t seed, uint64_t &seq) {
    launch_type = uniform_range(seed, seq++, 0, 2);
  }

  void select_projection(const uint64_t seed, uint64_t &seq) {
    projection_offset = 0;
    if (launch_type == 1) {
      projection_offset = uniform_range(seed, seq++, 0, config.region_tree_width);
      LOG_ONCE(log_fuzz.info() << "  Shifting shard points by: " << projection_offset);
    }
  }

  void execute(std::vector<Future> &futures) {
    if (launch_type == 0) {
      LOG_ONCE(log_fuzz.info() << "  Launch type: index space");
      IndexTaskLauncher launcher(task_id, launch_domain, TaskArgument(), ArgumentMap());
      if (!fields.empty()) {
        if (privilege == LEGION_REDUCE) {
          launcher.add_region_requirement(
              RegionRequirement(forest.get_disjoint_partition(), 0,
                                std::set<FieldID>(fields.begin(), fields.end()), fields,
                                redop, EXCLUSIVE, forest.get_root()));
        } else {
          launcher.add_region_requirement(
              RegionRequirement(forest.get_disjoint_partition(), 0,
                                std::set<FieldID>(fields.begin(), fields.end()), fields,
                                privilege, EXCLUSIVE, forest.get_root()));
        }
      }
      if (elide_future_return) {
        launcher.elide_future_return = true;
      }
      if (task_produces_value) {
        Future result = runtime->execute_index_space(ctx, launcher, scalar_redop,
                                                     scalar_reduction_ordered);
        if (!elide_future_return) {
          futures.push_back(result);
        }
      } else {
        runtime->execute_index_space(ctx, launcher);
      }
    } else {
      LOG_ONCE(log_fuzz.info() << "  Launch type: individual tasks");

      for (uint64_t point = range_min; point <= range_max; ++point) {
        TaskLauncher launcher(task_id, TaskArgument());
        launcher.point =
            Point<1>((point - range_min + projection_offset) % range_size + range_min);
        LOG_ONCE(log_fuzz.info() << "  Task: " << point);
        LOG_ONCE(log_fuzz.info() << "    Shard point: " << launcher.point);
        IndexSpaceT<1> launch_space = runtime->create_index_space<1>(ctx, launch_domain);
        launcher.sharding_space = launch_space;
        if (!fields.empty()) {
          LogicalRegion subregion = runtime->get_logical_subregion_by_color(
              forest.get_disjoint_partition(), point);
          if (privilege == LEGION_REDUCE) {
            launcher.add_region_requirement(RegionRequirement(
                subregion, std::set<FieldID>(fields.begin(), fields.end()), fields, redop,
                EXCLUSIVE, forest.get_root()));
          } else {
            launcher.add_region_requirement(RegionRequirement(
                subregion, std::set<FieldID>(fields.begin(), fields.end()), fields,
                privilege, EXCLUSIVE, forest.get_root()));
          }
        }
        if (elide_future_return) {
          launcher.elide_future_return = true;
        }
        if (task_produces_value) {
          Future result = runtime->execute_task(ctx, launcher);
          if (!elide_future_return) {
            futures.push_back(result);
          }
        } else {
          runtime->execute_task(ctx, launcher);
        }
        runtime->destroy_index_space(ctx, launch_space);
      }
    }
  }

private:
  Runtime *runtime;
  Context ctx;
  const FuzzerConfig &config;
  const RegionForest<1> &forest;
  TaskID task_id;
  bool task_produces_value;
  bool launch_complete;
  uint64_t range_min;
  uint64_t range_max;
  uint64_t range_size;
  Rect<1> launch_domain;
  ReductionOpID scalar_redop;
  bool scalar_reduction_ordered;
  bool elide_future_return;
  std::vector<FieldID> fields;
  PrivilegeMode privilege;
  ReductionOpID redop = LEGION_REDOP_LAST;
  uint64_t launch_type;
  uint64_t projection_offset;
};

void top_level(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx,
               Runtime *runtime) {
  InputArgs args = Runtime::get_input_args();
  FuzzerConfig config = FuzzerConfig::parse_args(args.argc, args.argv);
  config.log_config(runtime, ctx);

  RegionForest<1> forest(runtime, ctx, config);

  std::vector<Future> futures;

  const uint64_t seed = config.initial_seed;
  uint64_t seq = 0;
  for (uint64_t op_idx = 0; op_idx < config.num_ops; ++op_idx) {
    LOG_ONCE(log_fuzz.info() << "Operation: " << op_idx);

    Operation op(runtime, ctx, config, forest);

    // Step 1. Choose a random task to run.
    op.select_task_id(seed, seq);

    // Step 2. Choose launch domain.
    op.select_launch_domain(seed, seq);

    // Step 3. Choose scalar reduction.
    op.select_scalar_reduction(seed, seq);

    // Step 5. Choose whether to elide future return.
    op.select_elide_future_return(seed, seq);

    // Step 6. Choose fields.
    op.select_fields(seed, seq);

    // Step 7. Choose privilege.
    op.select_privilege(seed, seq);

    // Step 8. Choose reduction.
    op.select_reduction(seed, seq);

    // Step 9. Choose the launch type.
    op.select_launch_type(seed, seq);
    op.select_projection(seed, seq);
    op.execute(futures);
  }

  for (Future &future : futures) {
    int64_t result = future.get_result<int64_t>();
    LOG_ONCE(log_fuzz.info() << "Future result: " << result);
  }
}

int main(int argc, char **argv) {
  Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);
  {
    TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_inner();
    registrar.set_replicable();
    Runtime::preregister_task_variant<top_level>(registrar, "top_level");
  }

  {
    TaskVariantRegistrar registrar(VOID_LEAF_TASK_ID, "void_leaf");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<void_leaf>(registrar, "void_leaf");
  }

  {
    TaskVariantRegistrar registrar(INT64_LEAF_TASK_ID, "int64_leaf");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<int64_t, int64_leaf>(registrar, "int64_leaf");
  }

  {
    TaskVariantRegistrar registrar(VOID_INNER_TASK_ID, "void_inner");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_inner();
    Runtime::preregister_task_variant<void_inner>(registrar, "void_inner");
  }

  {
    TaskVariantRegistrar registrar(INT64_INNER_TASK_ID, "int64_inner");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_inner();
    Runtime::preregister_task_variant<int64_t, int64_inner>(registrar, "int64_inner");
  }

  {
    TaskVariantRegistrar registrar(VOID_REPLICABLE_LEAF_TASK_ID, "void_replicable_leaf");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    registrar.set_replicable();
    Runtime::preregister_task_variant<void_replicable_leaf>(registrar,
                                                            "void_replicable_leaf");
  }

  {
    TaskVariantRegistrar registrar(INT64_REPLICABLE_LEAF_TASK_ID,
                                   "int64_replicable_leaf");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    registrar.set_replicable();
    Runtime::preregister_task_variant<int64_t, int64_replicable_leaf>(
        registrar, "int64_replicable_leaf");
  }

  {
    TaskVariantRegistrar registrar(VOID_REPLICABLE_INNER_TASK_ID,
                                   "void_replicable_inner");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_inner();
    registrar.set_replicable();
    Runtime::preregister_task_variant<void_replicable_inner>(registrar,
                                                             "void_replicable_inner");
  }

  {
    TaskVariantRegistrar registrar(INT64_REPLICABLE_INNER_TASK_ID,
                                   "int64_replicable_inner");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_inner();
    registrar.set_replicable();
    Runtime::preregister_task_variant<int64_t, int64_replicable_inner>(
        registrar, "int64_replicable_inner");
  }

  return Runtime::start(argc, argv);
}
