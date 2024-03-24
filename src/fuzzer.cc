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

#include "deterministic_random.h"
#include "legion.h"

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
  COLOR_POINTS_TASK_ID,
  TOP_LEVEL_TASK_ID,
};

enum ProjectionIDs {
  PROJECTION_OFFSET_1_ID = 1,
  PROJECTION_OFFSET_2_ID = 2,
  PROJECTION_LAST_DISJOINT_ID = PROJECTION_OFFSET_2_ID,
  PROJECTION_RANDOM_DEPTH_1_ID = 3,
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
  uint64_t initial_seed = 0;
  uint64_t region_tree_depth = 1;
  uint64_t region_tree_width = 4;
  uint64_t region_tree_branch_factor = 4;
  uint64_t region_tree_size_factor = 4;
  uint64_t region_tree_num_fields = 4;
  uint64_t num_ops = 1;
  uint64_t skip_ops = 0;

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
      } else if (flag == "-fuzz:size") {
        std::string arg(argv[++i]);
        config.region_tree_size_factor = parse_uint64_t(flag, arg);
      } else if (flag == "-fuzz:fields") {
        std::string arg(argv[++i]);
        config.region_tree_num_fields = parse_uint64_t(flag, arg);
      } else if (flag == "-fuzz:ops") {
        std::string arg(argv[++i]);
        config.num_ops = parse_uint64_t(flag, arg);
      } else if (flag == "-fuzz:skip") {
        std::string arg(argv[++i]);
        config.skip_ops = parse_uint64_t(flag, arg);
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
    LOG_ONCE(log_fuzz.print() << "  config.region_tree_size_factor = "
                              << region_tree_size_factor);
    LOG_ONCE(log_fuzz.print() << "  config.region_tree_num_fields = "
                              << region_tree_num_fields);
    LOG_ONCE(log_fuzz.print() << "  config.num_ops = " << num_ops);
    LOG_ONCE(log_fuzz.print() << "  config.skip_ops = " << skip_ops);
  }
};

class OffsetProjection : public ProjectionFunctor {
public:
  OffsetProjection(uint64_t _offset) : offset(_offset) {}
  bool is_functional(void) const override { return true; }
  bool is_invertible(void) const override { return false; }
  unsigned get_depth(void) const override { return 0; }
  LogicalRegion project(LogicalPartition upper_bound, const DomainPoint &point,
                        const Domain &launch_domain) override {
    Domain color_space =
        runtime->get_index_partition_color_space(upper_bound.get_index_partition());
    Rect<1> rect = color_space;
    uint64_t index = Point<1>(point)[0];
    index = (index - rect.lo[0] + offset) % (rect.hi[0] - rect.lo[0] + 1) + rect.lo[0];

    return runtime->get_logical_subregion_by_color(upper_bound, index);
  }

protected:
  uint64_t offset;
};

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

struct ColorPointsArgs {
  uint64_t seed;
  uint64_t stream;
  uint64_t region_tree_width;
};

void color_points_task(const Task *task, const std::vector<PhysicalRegion> &regions,
                       Context ctx, Runtime *runtime) {
  assert(task->arglen == sizeof(ColorPointsArgs));
  const ColorPointsArgs args = *reinterpret_cast<ColorPointsArgs *>(task->args);
  const uint64_t seed = args.seed;
  const uint64_t stream = args.stream;
  const uint64_t region_tree_width = args.region_tree_width;
  uint64_t seq = 0;

  PhysicalRegion pr = regions[0];
  std::vector<FieldID> fields;
  pr.get_fields(fields);

  DomainT<1> domain = runtime->get_index_space_domain(
      ctx, IndexSpaceT<1>(pr.get_logical_region().get_index_space()));

  for (FieldID field : fields) {
    const FieldAccessor<READ_WRITE, Point<1>, 1> acc(pr, field);
    for (PointInDomainIterator<1> pir(domain); pir(); pir++) {
      uint64_t color = uniform_range(seed, stream, seq, 0, region_tree_width - 1);
      acc[*pir] = color;
    }
  }
}

class RegionForest {
public:
  RegionForest(Runtime *_runtime, Context _ctx, const FuzzerConfig &config,
               const uint64_t seed, uint64_t &stream)
      : runtime(_runtime), ctx(_ctx) {
    ispace = runtime->create_index_space<1>(
        ctx,
        Rect<1>(Point<1>(0),
                Point<1>(config.region_tree_width * config.region_tree_size_factor - 1)));

    fspace = runtime->create_field_space(ctx);
    FieldAllocator falloc = runtime->create_field_allocator(ctx, fspace);
    uint64_t num_fields = config.region_tree_num_fields;
    uint64_t branch_factor = config.region_tree_branch_factor;
    {
      FieldID fid = 0;
      for (uint64_t field = 0; field < num_fields; ++field) {
        falloc.allocate_field(sizeof(uint64_t), fid++);
      }
      for (uint64_t num_colors = 1; num_colors < branch_factor; ++num_colors) {
        for (uint64_t color = 0; color < num_colors; ++color) {
          falloc.allocate_field(sizeof(Point<1>), fid++);
        }
      }
    }
    root = runtime->create_logical_region(ctx, ispace, fspace);

    color_space = runtime->create_index_space<1>(
        ctx, Rect<1>(Point<1>(0), Point<1>(config.region_tree_width - 1)));

    // Always make an equal partition first.
    disjoint_partitions.push_back(
        runtime->create_equal_partition(ctx, ispace, color_space));

    // After this we make progressive colored regions.
    {
      FieldID fid = num_fields;
      for (uint64_t num_colors = 1; num_colors < branch_factor; ++num_colors) {
        std::vector<FieldID> colors;
        for (uint64_t color = 0; color < num_colors; ++color) {
          colors.push_back(fid++);
        }

        color_points(colors, config, seed, stream);

        std::vector<IndexPartition> color_parts;
        for (FieldID color : colors) {
          color_parts.push_back(
              runtime->create_partition_by_field(ctx, root, root, color, color_space));
        }

        if (color_parts.size() == 1) {
          disjoint_partitions.push_back(color_parts[0]);
        } else {
          IndexPartition part =
              runtime->create_pending_partition(ctx, ispace, color_space);
          for (uint64_t point = 0; point < config.region_tree_width; ++point) {
            std::vector<IndexSpace> subspaces;
            for (IndexPartition color_part : color_parts) {
              subspaces.push_back(runtime->get_index_subspace(color_part, point));
            }
            runtime->create_index_space_union(ctx, part, Point<1>(point), subspaces);
          }
          aliased_partitions.push_back(part);
        }

        for (FieldID color : colors) {
          falloc.free_field(color);
        }
      }
    }

    // Initialize everything so we don't get unitialized read warnings
    for (uint64_t field = 0; field < config.region_tree_num_fields; ++field) {
      runtime->fill_field<uint64_t>(ctx, root, root, field, field);
    }
  }

  ~RegionForest() {
    runtime->destroy_logical_region(ctx, root);
    runtime->destroy_field_space(ctx, fspace);
    runtime->destroy_index_space(ctx, ispace);
    runtime->destroy_index_space(ctx, color_space);
  }

  LogicalRegion get_root() const { return root; }

  LogicalPartition select_disjoint_partition(const uint64_t seed, const uint64_t stream,
                                             uint64_t &seq) {
    uint64_t num_disjoint = disjoint_partitions.size();
    uint64_t part_idx = uniform_range(seed, stream, seq, 0, num_disjoint - 1);
    return runtime->get_logical_partition(root, disjoint_partitions[part_idx]);
  }

  LogicalPartition select_any_partition(const uint64_t seed, const uint64_t stream,
                                        uint64_t &seq) {
    uint64_t num_disjoint = disjoint_partitions.size();
    uint64_t num_aliased = aliased_partitions.size();
    uint64_t num_total = num_disjoint + num_aliased;
    uint64_t part_idx = uniform_range(seed, stream, seq, 0, num_total - 1);
    if (part_idx < num_disjoint) {
      return runtime->get_logical_partition(root, disjoint_partitions[part_idx]);
    } else {
      part_idx -= num_disjoint;
      assert(part_idx < num_aliased);
      return runtime->get_logical_partition(root, aliased_partitions[part_idx]);
    }
  }

  void set_last_redop(ReductionOpID redop) { last_redop = redop; }
  ReductionOpID get_last_redop() const { return last_redop; }

private:
  void color_points(const std::vector<FieldID> &colors, const FuzzerConfig &config,
                    const uint64_t seed, uint64_t &stream) {
    ColorPointsArgs args;
    args.seed = seed;
    args.stream = stream;
    args.region_tree_width = config.region_tree_width;
    TaskLauncher launcher(COLOR_POINTS_TASK_ID, TaskArgument(&args, sizeof(args)));
    launcher.add_region_requirement(
        RegionRequirement(root, std::set<FieldID>(colors.begin(), colors.end()), colors,
                          WRITE_DISCARD, EXCLUSIVE, root));
    launcher.elide_future_return = true;
    runtime->execute_task(ctx, launcher);
  }

private:
  Runtime *runtime;
  Context ctx;
  IndexSpaceT<1> ispace;
  FieldSpace fspace;
  LogicalRegion root;
  IndexSpaceT<1> color_space;
  std::vector<IndexPartition> disjoint_partitions;
  std::vector<IndexPartition> aliased_partitions;
  ReductionOpID last_redop = LEGION_REDOP_LAST;
};

static ReductionOpID select_redop(const uint64_t seed, const uint64_t stream,
                                  uint64_t &seq) {
  switch (uniform_range(seed, stream, seq, 0, 8)) {
    case 0:
      return LEGION_REDOP_SUM_INT64;
    case 1:
      return LEGION_REDOP_DIFF_INT64;
    case 2:
      return LEGION_REDOP_PROD_INT64;
    case 3:
      return LEGION_REDOP_DIV_INT64;
    case 4:
      return LEGION_REDOP_MIN_INT64;
    case 5:
      return LEGION_REDOP_MAX_INT64;
    case 6:
      return LEGION_REDOP_AND_INT64;
    case 7:
      return LEGION_REDOP_OR_INT64;
    case 8:
      return LEGION_REDOP_XOR_INT64;
    default:
      abort();
  }
}

class RequirementBuilder {
public:
  RequirementBuilder(const FuzzerConfig &_config, RegionForest &_forest)
      : config(_config), forest(_forest) {}

  void build(const uint64_t seed, const uint64_t stream, uint64_t &seq,
             bool launch_complete, bool requires_projection) {
    select_fields(seed, stream, seq);
    select_privilege(seed, stream, seq);
    select_reduction(seed, stream, seq, launch_complete);
    select_projection(seed, stream, seq, requires_projection);
    select_partition(seed, stream, seq, requires_projection);
  }

private:
  void select_fields(const uint64_t seed, const uint64_t stream, uint64_t &seq) {
    fields.clear();

    uint64_t field_set =
        uniform_range(seed, stream, seq, 0, (1 << config.region_tree_num_fields) - 1);
    for (uint64_t field = 0; field < config.region_tree_num_fields; ++field) {
      if ((field_set & (1 << field)) != 0) {
        fields.push_back(field);
      }
    }
  }

  void select_privilege(const uint64_t seed, const uint64_t stream, uint64_t &seq) {
    switch (uniform_range(seed, stream, seq, 0, 4)) {
      case 0: {
        privilege = LEGION_NO_ACCESS;
      } break;
      case 1: {
        privilege = LEGION_READ_ONLY;
      } break;
      case 2: {
        privilege = LEGION_READ_WRITE;
      } break;
      case 3: {
        privilege = LEGION_WRITE_DISCARD;
      } break;
      case 4: {
        privilege = LEGION_REDUCE;
      } break;
      default:
        abort();
    }
  }

  void select_reduction(const uint64_t seed, const uint64_t stream, uint64_t &seq,
                        bool launch_complete) {
    redop = LEGION_REDOP_LAST;
    if (privilege == LEGION_REDUCE) {
      if (forest.get_last_redop() != LEGION_REDOP_LAST) {
        // When we run two or more reductions back to back, we must use the
        // same reduction operator again.
        redop = forest.get_last_redop();
      } else {
        redop = select_redop(seed, stream, seq);
        forest.set_last_redop(redop);
      }
    } else if (privilege != LEGION_NO_ACCESS && launch_complete) {
      // Any non-reduction (non-no-access) privilege will clear the last
      // reduction, making it safe to reduce again, assuming the launch
      // actually covers the entire region.
      forest.set_last_redop(LEGION_REDOP_LAST);
    }
  }

  bool need_disjoint() const {
    // If our privilege involves writing (read-write, write-discard, etc.),
    // then we require a disjoint region requirement (i.e. disjoint partition
    // and disjoint projection functor).
    return (privilege & LEGION_WRITE_PRIV) != 0;
  }

  void select_projection(const uint64_t seed, const uint64_t stream, uint64_t &seq,
                         bool requires_projection) {
    projection = LEGION_MAX_APPLICATION_PROJECTION_ID;
    if (requires_projection) {
      switch (uniform_range(seed, stream, seq, 0, 2)) {
        case 0: {
          projection = 0;  // identity projection functor
        } break;
        case 1: {
          projection = PROJECTION_OFFSET_1_ID;
        } break;
        case 2: {
          projection = PROJECTION_OFFSET_2_ID;
        } break;
        default:
          abort();
      }
    }
  }

  void select_partition(const uint64_t seed, const uint64_t stream, uint64_t &seq,
                        bool requires_projection) {
    if (requires_projection && need_disjoint()) {
      partition = forest.select_disjoint_partition(seed, stream, seq);
    } else {
      partition = forest.select_any_partition(seed, stream, seq);
    }
  }

public:
  void add_to_index_task(IndexTaskLauncher &launcher) {
    if (!fields.empty()) {
      assert(projection != LEGION_MAX_APPLICATION_PROJECTION_ID);
      if (privilege == LEGION_REDUCE) {
        launcher.add_region_requirement(RegionRequirement(
            partition, projection, std::set<FieldID>(fields.begin(), fields.end()),
            fields, redop, EXCLUSIVE, forest.get_root()));
      } else {
        launcher.add_region_requirement(RegionRequirement(
            partition, projection, std::set<FieldID>(fields.begin(), fields.end()),
            fields, privilege, EXCLUSIVE, forest.get_root()));
      }
    }
  }

  void add_to_single_task(Runtime *runtime, TaskLauncher &launcher, Point<1> point) {
    if (!fields.empty()) {
      LogicalRegion subregion =
          runtime->get_logical_subregion_by_color(partition, point[0]);
      if (privilege == LEGION_REDUCE) {
        launcher.add_region_requirement(
            RegionRequirement(subregion, std::set<FieldID>(fields.begin(), fields.end()),
                              fields, redop, EXCLUSIVE, forest.get_root()));
      } else {
        launcher.add_region_requirement(
            RegionRequirement(subregion, std::set<FieldID>(fields.begin(), fields.end()),
                              fields, privilege, EXCLUSIVE, forest.get_root()));
      }
    }
  }

  void display(Runtime *runtime, Context ctx) const {
    {
      Realm::LoggerMessage msg = log_fuzz.info();
      msg = msg << "  Fields:";
      for (FieldID field : fields) {
        msg = msg << " " << field;
      }
      LOG_ONCE(msg);
    }
    LOG_ONCE(log_fuzz.info() << "  Privilege: 0x" << std::hex << privilege);
    if (redop != LEGION_REDOP_LAST) {
      LOG_ONCE(log_fuzz.info() << "  Region redop: " << redop);
    }
    if (projection == LEGION_MAX_APPLICATION_PROJECTION_ID) {
      LOG_ONCE(log_fuzz.info() << "  Projection: " << projection);
    }
    LOG_ONCE(log_fuzz.info() << "  Partition: " << partition);
  }

private:
  const FuzzerConfig &config;
  RegionForest &forest;
  std::vector<FieldID> fields;
  PrivilegeMode privilege = LEGION_NO_ACCESS;
  ReductionOpID redop = LEGION_REDOP_LAST;
  ProjectionID projection = LEGION_MAX_APPLICATION_PROJECTION_ID;
  LogicalPartition partition = LogicalPartition::NO_PART;
};

enum class LaunchType {
  SINGLE_TASK,
  INDEX_TASK,
  INVALID,
};

class OperationBuilder {
public:
  OperationBuilder(const FuzzerConfig &_config, RegionForest &_forest)
      : config(_config), launch_domain(Rect<1>::make_empty()), req(_config, _forest) {}

  void build(const uint64_t seed, const uint64_t stream, uint64_t &seq) {
    select_launch_type(seed, stream, seq);
    select_task_id(seed, stream, seq);
    select_launch_domain(seed, stream, seq);
    select_scalar_reduction(seed, stream, seq);
    select_elide_future_return(seed, stream, seq);
    select_region_requirement(seed, stream, seq);
    select_shard_offset(seed, stream, seq);
  }

private:
  void select_launch_type(const uint64_t seed, const uint64_t stream, uint64_t &seq) {
    switch (uniform_range(seed, stream, seq, 0, 1)) {
      case 0: {
        launch_type = LaunchType::SINGLE_TASK;
      } break;
      case 1: {
        launch_type = LaunchType::INDEX_TASK;
      } break;
      default:
        abort();
    }
  }

  void select_task_id(const uint64_t seed, const uint64_t stream, uint64_t &seq) {
    switch (uniform_range(seed, stream, seq, 0, 3)) {
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
  }

  void select_launch_domain(const uint64_t seed, const uint64_t stream, uint64_t &seq) {
    // A lot of Legion algorithms hinge on whether a launch is
    // complete or not, so we'll make that a special case here.

    launch_complete = uniform_range(seed, stream, seq, 0, 1) == 0;

    if (launch_complete) {
      range_min = 0;
      range_max = config.region_tree_width - 1;
    } else {
      range_min = uniform_range(seed, stream, seq, 0, config.region_tree_width - 1);
      range_max = uniform_range(seed, stream, seq, 0, config.region_tree_width - 1);
      // Make sure the range is always non-empty.
      if (range_max < range_min) {
        std::swap(range_min, range_max);
      }
    }

    range_size = range_max - range_min + 1;
    launch_domain = Rect<1>(Point<1>(range_min), Point<1>(range_max));
  }

  void select_scalar_reduction(const uint64_t seed, const uint64_t stream,
                               uint64_t &seq) {
    scalar_redop = LEGION_REDOP_LAST;
    if (launch_type == LaunchType::INDEX_TASK && task_produces_value) {
      scalar_redop = select_redop(seed, stream, seq);
      scalar_reduction_ordered = uniform_range(seed, stream, seq, 0, 1) == 0;
    }
  }

  void select_elide_future_return(const uint64_t seed, const uint64_t stream,
                                  uint64_t &seq) {
    elide_future_return = uniform_range(seed, stream, seq, 0, 1) == 0;
  }

  void select_region_requirement(const uint64_t seed, const uint64_t stream,
                                 uint64_t &seq) {
    req.build(seed, stream, seq, launch_complete, launch_type == LaunchType::INDEX_TASK);
  }

  void select_shard_offset(const uint64_t seed, const uint64_t stream, uint64_t &seq) {
    shard_offset = 0;
    if (launch_type == LaunchType::SINGLE_TASK) {
      shard_offset = uniform_range(seed, stream, seq, 0, config.region_tree_width - 1);
    }
  }

public:
  void execute(Runtime *runtime, Context ctx, std::vector<Future> &futures) {
    display(runtime, ctx);
    switch (launch_type) {
      case LaunchType::SINGLE_TASK: {
        execute_single_task(runtime, ctx, futures);
      } break;
      case LaunchType::INDEX_TASK: {
        execute_index_task(runtime, ctx, futures);
      } break;
      default:
        abort();
    }
  }

private:
  void execute_index_task(Runtime *runtime, Context ctx, std::vector<Future> &futures) {
    IndexTaskLauncher launcher(task_id, launch_domain, TaskArgument(), ArgumentMap());
    req.add_to_index_task(launcher);
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
  }

  void execute_single_task(Runtime *runtime, Context ctx, std::vector<Future> &futures) {
    for (uint64_t point = range_min; point <= range_max; ++point) {
      TaskLauncher launcher(task_id, TaskArgument());
      launcher.point =
          Point<1>((point - range_min + shard_offset) % range_size + range_min);
      LOG_ONCE(log_fuzz.info() << "  Task: " << point);
      LOG_ONCE(log_fuzz.info() << "    Shard point: " << launcher.point);
      IndexSpaceT<1> launch_space = runtime->create_index_space<1>(ctx, launch_domain);
      launcher.sharding_space = launch_space;
      req.add_to_single_task(runtime, launcher, point);
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

  void display(Runtime *runtime, Context ctx) const {
    switch (launch_type) {
      case LaunchType::SINGLE_TASK: {
        LOG_ONCE(log_fuzz.info() << "  Launch type: single task");
      } break;
      case LaunchType::INDEX_TASK: {
        LOG_ONCE(log_fuzz.info() << "  Launch type: index space");
      } break;
      default:
        abort();
    }

    LOG_ONCE(log_fuzz.info() << "  Task ID: " << task_id);
    LOG_ONCE(log_fuzz.info() << "  Launch domain: " << launch_domain);
    if (scalar_redop != LEGION_REDOP_LAST) {
      LOG_ONCE(log_fuzz.info() << "  Scalar redop: " << scalar_redop);
      LOG_ONCE(log_fuzz.info() << "    Ordered: " << scalar_reduction_ordered);
    }
    LOG_ONCE(log_fuzz.info() << "  Elide future return: " << elide_future_return);
    req.display(runtime, ctx);
    if (launch_type == LaunchType::SINGLE_TASK) {
      LOG_ONCE(log_fuzz.info() << "  Shifting shard points by: " << shard_offset);
    }
  }

private:
  const FuzzerConfig &config;
  LaunchType launch_type = LaunchType::INVALID;
  TaskID task_id = LEGION_MAX_APPLICATION_TASK_ID;
  bool task_produces_value = false;
  bool launch_complete = false;
  uint64_t range_min = 0;
  uint64_t range_max = 0;
  uint64_t range_size = 0;
  Rect<1> launch_domain;
  ReductionOpID scalar_redop = LEGION_REDOP_LAST;
  bool scalar_reduction_ordered = false;
  bool elide_future_return = false;
  RequirementBuilder req;
  uint64_t shard_offset = 0;
};

void top_level(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx,
               Runtime *runtime) {
  InputArgs args = Runtime::get_input_args();
  FuzzerConfig config = FuzzerConfig::parse_args(args.argc, args.argv);
  config.log_config(runtime, ctx);

  const uint64_t seed = config.initial_seed;

  uint64_t next_stream = 0;
  const uint64_t stream = next_stream++;

  RegionForest forest(runtime, ctx, config, seed, next_stream);

  std::vector<Future> futures;
  uint64_t seq = 0;
  for (uint64_t op_idx = 0; op_idx < config.num_ops; ++op_idx) {
    OperationBuilder op(config, forest);
    op.build(seed, stream, seq);

    // It is VERY IMPORTANT that the random number generator is NOT USED
    // inside this if statement. Otherwise, we lose the ability to replay
    // while skipping operations.
    if (op_idx >= config.skip_ops) {
      LOG_ONCE(log_fuzz.info() << "Operation: " << op_idx);
      op.execute(runtime, ctx, futures);
    }
  }

  for (Future &future : futures) {
    int64_t result = future.get_result<int64_t>();
    LOG_ONCE(log_fuzz.info() << "Future result: " << result);
  }
}

int main(int argc, char **argv) {
  Runtime::preregister_projection_functor(PROJECTION_OFFSET_1_ID,
                                          new OffsetProjection(1));
  Runtime::preregister_projection_functor(PROJECTION_OFFSET_2_ID,
                                          new OffsetProjection(2));

  Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);
  {
    TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_inner();
    registrar.set_replicable();
    Runtime::preregister_task_variant<top_level>(registrar, "top_level");
  }

  {
    TaskVariantRegistrar registrar(COLOR_POINTS_TASK_ID, "color_points");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<color_points_task>(registrar, "color_points");
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
