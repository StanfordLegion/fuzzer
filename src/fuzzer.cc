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

#include <iostream>
#include <string>

#include "legion.h"

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
  uint64_t num_ops;
  FuzzerConfig() : initial_seed(0), region_tree_depth(1), num_ops(1) {}
};

FuzzerConfig parse_args(int argc, char **argv) {
  FuzzerConfig config;
  for (int i = 1; i < argc; i++) {
    std::string flag(argv[i]);
    if (flag == "-fuzz:seed") {
      std::string arg(argv[++i]);
      config.initial_seed = parse_uint64_t(flag, arg);
    } else if (flag == "-fuzz:depth") {
      std::string arg(argv[++i]);
      config.region_tree_depth = parse_uint64_t(flag, arg);
    } else if (flag == "-fuzz:ops") {
      std::string arg(argv[++i]);
      config.num_ops = parse_uint64_t(flag, arg);
    }
  }
  return config;
}

void void_leaf(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx,
               Runtime *runtime) {}

int int_leaf(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx,
             Runtime *runtime) {
  return 123;
}

void void_inner(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx,
                Runtime *runtime) {}

int int_inner(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx,
              Runtime *runtime) {
  return 123;
}

void void_replicable_leaf(const Task *task, const std::vector<PhysicalRegion> &regions,
                          Context ctx, Runtime *runtime) {}

int int_replicable_leaf(const Task *task, const std::vector<PhysicalRegion> &regions,
                        Context ctx, Runtime *runtime) {
  return 123;
}

void void_replicable_inner(const Task *task, const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime) {}

int int_replicable_inner(const Task *task, const std::vector<PhysicalRegion> &regions,
                         Context ctx, Runtime *runtime) {
  return 123;
}

void top_level(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx,
               Runtime *runtime) {
  InputArgs args = Runtime::get_input_args();
  FuzzerConfig config = parse_args(args.argc, args.argv);

  for (uint64_t op = 0; op < config.num_ops; ++op) {
    TaskLauncher fx_launcher(INT_LEAF_TASK_ID, TaskArgument());
    Future fx = runtime->execute_task(ctx, fx_launcher);
    int rx = fx.get_result<int>();
    std::cout << "rx : " << rx << std::endl;
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
