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
#include <string>

#include "legion.h"
#include "siphash.h"

using namespace Legion;

enum TaskIDs {
  VOID_LEAF_TASK_ID,
  INT_LEAF_TASK_ID,
  VOID_INNER_TASK_ID,
  INT_INNER_TASK_ID,
  VOID_REPLICABLE_LEAF_TASK_ID,
  INT_REPLICABLE_LEAF_TASK_ID,
  VOID_REPLICABLE_INNER_TASK_ID,
  INT_REPLICABLE_INNER_TASK_ID,
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

int int_leaf(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx,
             Runtime *runtime) {
  return 1;
}

void void_inner(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx,
                Runtime *runtime) {}

int int_inner(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx,
              Runtime *runtime) {
  return 2;
}

void void_replicable_leaf(const Task *task, const std::vector<PhysicalRegion> &regions,
                          Context ctx, Runtime *runtime) {}

int int_replicable_leaf(const Task *task, const std::vector<PhysicalRegion> &regions,
                        Context ctx, Runtime *runtime) {
  return 3;
}

void void_replicable_inner(const Task *task, const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime) {}

int int_replicable_inner(const Task *task, const std::vector<PhysicalRegion> &regions,
                         Context ctx, Runtime *runtime) {
  return 4;
}

void top_level(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx,
               Runtime *runtime) {
  InputArgs args = Runtime::get_input_args();
  FuzzerConfig config = FuzzerConfig::parse_args(args.argc, args.argv);
  config.log_config(runtime, ctx);

  IndexSpaceT<1> ispace = runtime->create_index_space<1>(
      ctx, Rect<1>(Point<1>(0), Point<1>(config.region_tree_width)));
  FieldSpace fspace = runtime->create_field_space(ctx);
  {
    FieldAllocator falloc = runtime->create_field_allocator(ctx, fspace);
    for (uint64_t field = 0; field < config.region_tree_num_fields; ++field) {
      falloc.allocate_field(sizeof(uint64_t), field);
    }
  }
  LogicalRegion tree = runtime->create_logical_region(ctx, ispace, fspace);
  IndexPartition ipart = runtime->create_equal_partition(ctx, ispace, ispace);
  LogicalPartition lpart = runtime->get_logical_partition(tree, ipart);

  for (uint64_t field = 0; field < config.region_tree_num_fields; ++field) {
    runtime->fill_field<uint64_t>(ctx, tree, tree, field, field);
  }

  const uint64_t seed = config.initial_seed;
  uint64_t seq = 0;
  for (uint64_t op = 0; op < config.num_ops; ++op) {
    LOG_ONCE(log_fuzz.info() << "Operation: " << op);

    // Step 1. Choose a random task to run.
    TaskID task_id;
    switch (uniform_range(seed, seq++, 0, 4) & 3) {
      case 0: {
        task_id = VOID_LEAF_TASK_ID;
      } break;
      case 1: {
        task_id = VOID_INNER_TASK_ID;
      } break;
      case 2: {
        task_id = INT_LEAF_TASK_ID;
      } break;
      case 3: {
        task_id = INT_INNER_TASK_ID;
      } break;
    }
    LOG_ONCE(log_fuzz.info() << "  Task ID: " << task_id);

    // Step 2. Choose launch domain.
    uint64_t range_min = uniform_range(seed, seq++, 0, config.region_tree_width);
    uint64_t range_max = uniform_range(seed, seq++, 0, config.region_tree_width);
    // Make sure the range is always non-empty.
    if (range_max < range_min) {
      std::swap(range_min, range_max);
    }
    Rect<1> domain((Point<1>(range_min)), (Point<1>(range_max)));
    LOG_ONCE(log_fuzz.info() << "  Launch domain: " << domain);

    // Step 3. Choose fields.
    IndexTaskLauncher launcher(task_id, domain, TaskArgument(), ArgumentMap());
    uint64_t field_set =
        uniform_range(seed, seq++, 0, 1 << config.region_tree_num_fields);
    LOG_ONCE(log_fuzz.info() << "  Field set: " << field_set);
    if (field_set != 0) {
      RegionRequirement &req = launcher.add_region_requirement(
          RegionRequirement(lpart, 0, READ_WRITE, EXCLUSIVE, tree));
      for (uint64_t field = 0; field < config.region_tree_num_fields; ++field) {
        if ((field_set & (1 << field)) != 0) {
          req.add_field(field);
        }
      }
    }

    // Step 4. Launch.
    runtime->execute_index_space(ctx, launcher);
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
    TaskVariantRegistrar registrar(INT_LEAF_TASK_ID, "int_leaf");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<int, int_leaf>(registrar, "int_leaf");
  }

  {
    TaskVariantRegistrar registrar(VOID_INNER_TASK_ID, "void_inner");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_inner();
    Runtime::preregister_task_variant<void_inner>(registrar, "void_inner");
  }

  {
    TaskVariantRegistrar registrar(INT_INNER_TASK_ID, "int_inner");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_inner();
    Runtime::preregister_task_variant<int, int_inner>(registrar, "int_inner");
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
    TaskVariantRegistrar registrar(INT_REPLICABLE_LEAF_TASK_ID, "int_replicable_leaf");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    registrar.set_replicable();
    Runtime::preregister_task_variant<int, int_replicable_leaf>(registrar,
                                                                "int_replicable_leaf");
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
    TaskVariantRegistrar registrar(INT_REPLICABLE_INNER_TASK_ID, "int_replicable_inner");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_inner();
    registrar.set_replicable();
    Runtime::preregister_task_variant<int, int_replicable_inner>(registrar,
                                                                 "int_replicable_inner");
  }

  return Runtime::start(argc, argv);
}
