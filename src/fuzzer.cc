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

template <typename T>
const T unpack_args(const Task *task) {
  if (task->arglen != sizeof(T)) {
    log_fuzz.fatal() << "Wrong size in unpack_args: " << task->arglen
                     << ", expected: " << sizeof(T);
    abort();
  }
  const T result = *reinterpret_cast<T *>(task->args);
  return result;
}

struct PointTaskArgs {
  PointTaskArgs(uint64_t _value) : value(_value) {}
  uint64_t value;
};

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

static void mutate_region(Runtime *runtime, const IndexSpace &subspace,
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

struct ColorPointsArgs {
  ColorPointsArgs(RngStream _stream, uint64_t _width)
      : stream(_stream), region_tree_width(_width) {}

  RngStream stream;
  uint64_t region_tree_width;
};

void color_points_task(const Task *task, const std::vector<PhysicalRegion> &regions,
                       Context ctx, Runtime *runtime) {
  const ColorPointsArgs args = unpack_args<ColorPointsArgs>(task);
  RngStream rng = args.stream;
  const uint64_t region_tree_width = args.region_tree_width;

  PhysicalRegion pr = regions[0];
  std::vector<FieldID> fields;
  pr.get_fields(fields);

  DomainT<1> domain = runtime->get_index_space_domain(
      ctx, IndexSpaceT<1>(pr.get_logical_region().get_index_space()));

  for (FieldID field : fields) {
    const FieldAccessor<READ_WRITE, Point<1>, 1> acc(pr, field);
    for (PointInDomainIterator<1> pir(domain); pir(); pir++) {
      uint64_t color = rng.uniform_range(0, region_tree_width - 1);
      acc[*pir] = color;
    }
  }
}

class RegionForest {
public:
  RegionForest(Runtime *_runtime, Context _ctx, const FuzzerConfig &config, RngSeed &seed)
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
        falloc.allocate_field(sizeof(int64_t), fid++);
      }
      for (uint64_t num_colors = 1; num_colors < branch_factor; ++num_colors) {
        for (uint64_t color = 0; color < num_colors; ++color) {
          falloc.allocate_field(sizeof(Point<1>), fid++);
        }
      }
    }
    root = runtime->create_logical_region(ctx, ispace, fspace);
    shadow_root = runtime->create_logical_region(ctx, ispace, fspace);

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

        color_points(colors, config, seed);

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
      runtime->fill_field<uint64_t>(ctx, shadow_root, shadow_root, field, field);
    }

    // Inline map the shadow region for direct access.
    InlineLauncher launcher(
        RegionRequirement(shadow_root, LEGION_READ_WRITE, LEGION_EXCLUSIVE, shadow_root));
    for (uint64_t field = 0; field < config.region_tree_num_fields; ++field) {
      launcher.add_field(field);
    }
    shadow_inst = runtime->map_region(ctx, launcher);
    shadow_inst.wait_until_valid();
  }

  ~RegionForest() {
    runtime->unmap_region(ctx, shadow_inst);
    shadow_inst = PhysicalRegion();  // clear reference to instance
    runtime->destroy_logical_region(ctx, shadow_root);
    runtime->destroy_logical_region(ctx, root);
    runtime->destroy_field_space(ctx, fspace);
    runtime->destroy_index_space(ctx, ispace);
    runtime->destroy_index_space(ctx, color_space);
  }

  LogicalRegion get_root() const { return root; }
  LogicalRegion get_shadow_root() const { return shadow_root; }
  PhysicalRegion &get_shadow_inst() { return shadow_inst; }

  LogicalPartition select_disjoint_partition(RngStream &rng) {
    uint64_t num_disjoint = disjoint_partitions.size();
    uint64_t part_idx = rng.uniform_range(0, num_disjoint - 1);
    return runtime->get_logical_partition(root, disjoint_partitions[part_idx]);
  }

  LogicalPartition select_any_partition(RngStream &rng) {
    uint64_t num_disjoint = disjoint_partitions.size();
    uint64_t num_aliased = aliased_partitions.size();
    uint64_t num_total = num_disjoint + num_aliased;
    uint64_t part_idx = rng.uniform_range(0, num_total - 1);
    if (part_idx < num_disjoint) {
      return runtime->get_logical_partition(root, disjoint_partitions[part_idx]);
    } else {
      part_idx -= num_disjoint;
      assert(part_idx < num_aliased);
      return runtime->get_logical_partition(root, aliased_partitions[part_idx]);
    }
  }

  void set_last_redop(const std::vector<FieldID> &fields, ReductionOpID redop) {
    for (FieldID field : fields) {
      last_field_redop[field] = redop;
    }
  }

  bool get_last_redop(const std::vector<FieldID> &fields,
                      ReductionOpID &last_redop) const {
    last_redop = LEGION_REDOP_LAST;
    for (FieldID field : fields) {
      auto it = last_field_redop.find(field);
      if (it != last_field_redop.end()) {
        if (last_redop == LEGION_REDOP_LAST || last_redop == it->second) {
          last_redop = it->second;
        } else {
          last_redop = LEGION_REDOP_LAST;
          return false;
        }
      }
    }
    return true;
  }

  void verify_contents() {
    std::vector<FieldID> fields;
    shadow_inst.get_fields(fields);

    InlineLauncher launcher(
        RegionRequirement(root, LEGION_READ_WRITE, LEGION_EXCLUSIVE, root));
    for (FieldID field : fields) {
      launcher.add_field(field);
    }
    PhysicalRegion inst = runtime->map_region(ctx, launcher);
    inst.wait_until_valid();

    Domain dom = runtime->get_index_space_domain(root.get_index_space());

    for (FieldID field : fields) {
      log_fuzz.info() << "Verifying field: " << field;
      const FieldAccessor<LEGION_READ_ONLY, uint64_t, 1, coord_t,
                          Realm::AffineAccessor<uint64_t, 1, coord_t>>
          acc(inst, field);
      const FieldAccessor<LEGION_READ_ONLY, uint64_t, 1, coord_t,
                          Realm::AffineAccessor<uint64_t, 1, coord_t>>
          shadow_acc(shadow_inst, field);
      for (Domain::DomainPointIterator it(dom); it; ++it) {
        if (acc[*it] != shadow_acc[*it]) {
          log_fuzz.fatal() << "Bad region value: " << acc[*it]
                           << ", expected: " << shadow_acc[*it];
          abort();
        }
      }
    }

    runtime->unmap_region(ctx, inst);
  }

private:
  void color_points(const std::vector<FieldID> &colors, const FuzzerConfig &config,
                    RngSeed &seed) {
    ColorPointsArgs args(seed.make_stream(), config.region_tree_width);
    TaskLauncher launcher(COLOR_POINTS_TASK_ID, TaskArgument(&args, sizeof(args)));
    launcher.add_region_requirement(
        RegionRequirement(root, std::set<FieldID>(colors.begin(), colors.end()), colors,
                          LEGION_WRITE_DISCARD, LEGION_EXCLUSIVE, root));
    launcher.elide_future_return = true;
    runtime->execute_task(ctx, launcher);
  }

private:
  Runtime *runtime;
  Context ctx;
  IndexSpaceT<1> ispace;
  FieldSpace fspace;
  LogicalRegion root;
  LogicalRegion shadow_root;
  PhysicalRegion shadow_inst;
  IndexSpaceT<1> color_space;
  std::vector<IndexPartition> disjoint_partitions;
  std::vector<IndexPartition> aliased_partitions;
  std::map<FieldID, ReductionOpID> last_field_redop;
};

const char *task_name(TaskID task_id) {
  switch (task_id) {
    case VOID_LEAF_TASK_ID:
      return "VOID_LEAF_TASK_ID";
    case UINT64_LEAF_TASK_ID:
      return "UINT64_LEAF_TASK_ID";
    case VOID_INNER_TASK_ID:
      return "VOID_INNER_TASK_ID";
    case UINT64_INNER_TASK_ID:
      return "UINT64_INNER_TASK_ID";
    case VOID_REPLICABLE_LEAF_TASK_ID:
      return "VOID_REPLICABLE_LEAF_TASK_ID";
    case UINT64_REPLICABLE_LEAF_TASK_ID:
      return "UINT64_REPLICABLE_LEAF_TASK_ID";
    case VOID_REPLICABLE_INNER_TASK_ID:
      return "VOID_REPLICABLE_INNER_TASK_ID";
    case UINT64_REPLICABLE_INNER_TASK_ID:
      return "UINT64_REPLICABLE_INNER_TASK_ID";
    case COLOR_POINTS_TASK_ID:
      return "COLOR_POINTS_TASK_ID";
    case TOP_LEVEL_TASK_ID:
      return "TOP_LEVEL_TASK_ID";
    default:
      abort();
  }
}

const char *privilege_name(PrivilegeMode privilege) {
  switch (privilege) {
    case LEGION_NO_ACCESS:
      return "LEGION_NO_ACCESS";
    case LEGION_READ_ONLY:
      return "LEGION_READ_ONLY";
    case LEGION_REDUCE:
      return "LEGION_REDUCE";
    case LEGION_READ_WRITE:
      return "LEGION_READ_WRITE";
    case LEGION_WRITE_ONLY:
      return "LEGION_WRITE_ONLY";
    case LEGION_WRITE_DISCARD:
      return "LEGION_WRITE_DISCARD";
    default:
      abort();
  }
}

static ReductionOpID select_redop(RngStream &rng) {
  switch (rng.uniform_range(0, 5)) {
    case 0:
      return LEGION_REDOP_SUM_UINT64;
    // FIXME: shut off prod reduction because it seems to have bad behavior with inner
    // case 1:
    //   return LEGION_REDOP_PROD_UINT64;
    case 1:
      return LEGION_REDOP_MIN_UINT64;
    case 2:
      return LEGION_REDOP_MAX_UINT64;
    case 3:
      return LEGION_REDOP_AND_UINT64;
    case 4:
      return LEGION_REDOP_OR_UINT64;
    case 5:
      return LEGION_REDOP_XOR_UINT64;
    default:
      abort();
  }
}

const char *redop_name(ReductionOpID redop) {
  switch (redop) {
    case LEGION_REDOP_SUM_UINT64:
      return "SumReduction<uint64_t>";
    case LEGION_REDOP_PROD_UINT64:
      return "ProdReduction<uint64_t>";
    case LEGION_REDOP_MIN_UINT64:
      return "MinReduction<uint64_t>";
    case LEGION_REDOP_MAX_UINT64:
      return "MaxReduction<uint64_t>";
    case LEGION_REDOP_AND_UINT64:
      return "AndReduction<uint64_t>";
    case LEGION_REDOP_OR_UINT64:
      return "OrReduction<uint64_t>";
    case LEGION_REDOP_XOR_UINT64:
      return "XorReduction<uint64_t>";
    default:
      abort();
  }
}

class RequirementBuilder {
public:
  RequirementBuilder(const FuzzerConfig &_config, RegionForest &_forest)
      : config(_config), forest(_forest) {}

  void build(RngStream &rng, bool launch_complete, bool requires_projection,
             bool task_is_inner) {
    select_fields(rng);
    select_privilege(rng, task_is_inner);
    select_reduction(rng, launch_complete, task_is_inner);
    select_projection(rng, requires_projection);
    select_partition(rng, requires_projection);
  }

private:
  void select_fields(RngStream &rng) {
    fields.clear();

    uint64_t field_set = rng.uniform_range(0, (1 << config.region_tree_num_fields) - 1);
    for (uint64_t field = 0; field < config.region_tree_num_fields; ++field) {
      if ((field_set & (1 << field)) != 0) {
        fields.push_back(field);
      }
    }
  }

  void select_privilege(RngStream &rng, bool task_is_inner) {
    switch (rng.uniform_range(0, 4)) {
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

    if (task_is_inner && ((privilege & LEGION_WRITE_PRIV) != 0)) {
      // FIXME: https://github.com/StanfordLegion/legion/issues/1659
      privilege = LEGION_REDUCE;
    }
  }

  void select_reduction(RngStream &rng, bool launch_complete, bool task_is_inner) {
    redop = LEGION_REDOP_LAST;
    if (privilege == LEGION_REDUCE) {
      ReductionOpID last_redop;
      bool ok = forest.get_last_redop(fields, last_redop);
      if (ok && last_redop != LEGION_REDOP_LAST) {
        // When we run two or more reductions back to back, we must use the
        // same reduction operator again.
        redop = last_redop;
      } else if (!ok) {
        // Two or more fields had conflicting redops, so there's no way to
        // pick a single one across the entire set. Fall back to read-write.
        if (!task_is_inner) {
          privilege = LEGION_READ_WRITE;
        } else {
          // Unless the task is inner, in which case we're impacted by
          // https://github.com/StanfordLegion/legion/issues/1659
          privilege = LEGION_READ_ONLY;
        }
      } else {
        // No previous reduction, we're ok to go ahead and pick.
        redop = select_redop(rng);
      }
    }

    // We have to be conservative here: always set the redop (if we're doing a
    // reduction) but clear it only if the launch is complete.
    if (redop != LEGION_REDOP_LAST) {
      // Note: even if we got the redop through the cache, we still have to
      // make sure all fields are covered.
      forest.set_last_redop(fields, redop);
    } else if (privilege != LEGION_NO_ACCESS && launch_complete) {
      forest.set_last_redop(fields, LEGION_REDOP_LAST);
    }
  }

  bool need_disjoint() const {
    // If our privilege involves writing (read-write, write-discard, etc.),
    // then we require a disjoint region requirement (i.e. disjoint partition
    // and disjoint projection functor).
    return (privilege & LEGION_WRITE_PRIV) != 0;
  }

  void select_projection(RngStream &rng, bool requires_projection) {
    projection = LEGION_MAX_APPLICATION_PROJECTION_ID;
    if (requires_projection) {
      switch (rng.uniform_range(0, 2)) {
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

  void select_partition(RngStream &rng, bool requires_projection) {
    if (requires_projection && need_disjoint()) {
      partition = forest.select_disjoint_partition(rng);
    } else {
      partition = forest.select_any_partition(rng);
    }
  }

public:
  void add_to_index_task(IndexTaskLauncher &launcher) const {
    if (!fields.empty()) {
      assert(projection != LEGION_MAX_APPLICATION_PROJECTION_ID);
      if (privilege == LEGION_REDUCE) {
        launcher.add_region_requirement(RegionRequirement(
            partition, projection, std::set<FieldID>(fields.begin(), fields.end()),
            fields, redop, LEGION_EXCLUSIVE, forest.get_root()));
      } else {
        launcher.add_region_requirement(RegionRequirement(
            partition, projection, std::set<FieldID>(fields.begin(), fields.end()),
            fields, privilege, LEGION_EXCLUSIVE, forest.get_root()));
      }
    }
  }

  void add_to_single_task(Runtime *runtime, TaskLauncher &launcher,
                          Point<1> point) const {
    if (!fields.empty()) {
      LogicalRegion subregion =
          runtime->get_logical_subregion_by_color(partition, point[0]);
      if (privilege == LEGION_REDUCE) {
        launcher.add_region_requirement(
            RegionRequirement(subregion, std::set<FieldID>(fields.begin(), fields.end()),
                              fields, redop, LEGION_EXCLUSIVE, forest.get_root()));
      } else {
        launcher.add_region_requirement(
            RegionRequirement(subregion, std::set<FieldID>(fields.begin(), fields.end()),
                              fields, privilege, LEGION_EXCLUSIVE, forest.get_root()));
      }
    }
  }

  LogicalRegion project(Point<1> point, Rect<1> launch_domain) const {
    if (projection == LEGION_MAX_APPLICATION_PROJECTION_ID) {
      return LogicalRegion::NO_REGION;
    }

    ProjectionFunctor *functor = Runtime::get_projection_functor(projection);
    return functor->project(partition, point, launch_domain);
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
    LOG_ONCE(log_fuzz.info() << "  Privilege: " << privilege_name(privilege));
    if (redop != LEGION_REDOP_LAST) {
      LOG_ONCE(log_fuzz.info() << "  Region redop: " << redop_name(redop));
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

using FutureCheck = std::pair<Future, uint64_t>;

class OperationBuilder {
public:
  OperationBuilder(const FuzzerConfig &_config, RegionForest &_forest)
      : config(_config),
        forest(_forest),
        launch_domain(Rect<1>::make_empty()),
        req(_config, _forest) {}

  void build(RngStream &rng) {
    select_launch_type(rng);
    select_task_id(rng);
    select_launch_domain(rng);
    select_scalar_reduction(rng);
    select_elide_future_return(rng);
    select_region_requirement(rng);
    select_shard_offset(rng);
    select_task_arg(rng);
  }

private:
  void select_launch_type(RngStream &rng) {
    switch (rng.uniform_range(0, 1)) {
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

  void select_task_id(RngStream &rng) {
    task_produces_value = false;
    task_is_inner = false;
    switch (rng.uniform_range(0, 3)) {
      case 0: {
        task_id = VOID_LEAF_TASK_ID;
      } break;
      case 1: {
        task_id = UINT64_LEAF_TASK_ID;
        task_produces_value = true;
      } break;
      case 2: {
        task_id = VOID_INNER_TASK_ID;
        task_is_inner = true;
      } break;
      case 3: {
        task_id = UINT64_INNER_TASK_ID;
        task_produces_value = true;
        task_is_inner = true;
      } break;
      default:
        abort();
    }
  }

  void select_launch_domain(RngStream &rng) {
    // A lot of Legion algorithms hinge on whether a launch is
    // complete or not, so we'll make that a special case here.

    launch_complete = rng.uniform_range(0, 1) == 0;

    if (launch_complete) {
      range_min = 0;
      range_max = config.region_tree_width - 1;
    } else {
      range_min = rng.uniform_range(0, config.region_tree_width - 1);
      range_max = rng.uniform_range(0, config.region_tree_width - 1);
      // Make sure the range is always non-empty.
      if (range_max < range_min) {
        std::swap(range_min, range_max);
      }
    }

    range_size = range_max - range_min + 1;
    launch_domain = Rect<1>(Point<1>(range_min), Point<1>(range_max));
  }

  void select_scalar_reduction(RngStream &rng) {
    scalar_redop = LEGION_REDOP_LAST;
    if (launch_type == LaunchType::INDEX_TASK && task_produces_value) {
      scalar_redop = select_redop(rng);
      scalar_reduction_ordered = rng.uniform_range(0, 1) == 0;
    }
  }

  void select_elide_future_return(RngStream &rng) {
    elide_future_return = rng.uniform_range(0, 1) == 0;
  }

  void select_region_requirement(RngStream &rng) {
    req.build(rng, launch_complete, launch_type == LaunchType::INDEX_TASK, task_is_inner);
  }

  void select_shard_offset(RngStream &rng) {
    shard_offset = 0;
    if (launch_type == LaunchType::SINGLE_TASK) {
      shard_offset = rng.uniform_range(0, config.region_tree_width - 1);
    }
  }

  void select_task_arg(RngStream &rng) {
    // Choose something small enough to avoid overflow if we combine a lot of
    // values.
    task_arg_value = rng.uniform_range(0, 0xFFFFFFFFULL);
  }

public:
  void execute(Runtime *runtime, Context ctx, std::vector<FutureCheck> &futures) {
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
  template <typename REDOP>
  uint64_t compute_scalar_reduction() const {
    uint64_t result = REDOP::identity;
    for (uint64_t point = range_min; point <= range_max; ++point) {
      REDOP::template apply<true>(result, point ^ task_arg_value);
    }
    return result;
  }

  uint64_t compute_scalar_reduction_result() const {
    switch (scalar_redop) {
      case LEGION_REDOP_SUM_UINT64:
        return compute_scalar_reduction<SumReduction<uint64_t>>();
      case LEGION_REDOP_PROD_UINT64:
        return compute_scalar_reduction<ProdReduction<uint64_t>>();
      case LEGION_REDOP_MIN_UINT64:
        return compute_scalar_reduction<MinReduction<uint64_t>>();
      case LEGION_REDOP_MAX_UINT64:
        return compute_scalar_reduction<MaxReduction<uint64_t>>();
      case LEGION_REDOP_AND_UINT64:
        return compute_scalar_reduction<AndReduction<uint64_t>>();
      case LEGION_REDOP_OR_UINT64:
        return compute_scalar_reduction<OrReduction<uint64_t>>();
      case LEGION_REDOP_XOR_UINT64:
        return compute_scalar_reduction<XorReduction<uint64_t>>();
      default:
        abort();
    }
  }

  void execute_index_task(Runtime *runtime, Context ctx,
                          std::vector<FutureCheck> &futures) {
    PointTaskArgs args(task_arg_value);
    IndexTaskLauncher launcher(task_id, launch_domain, TaskArgument(&args, sizeof(args)),
                               ArgumentMap());
    req.add_to_index_task(launcher);
    if (elide_future_return) {
      launcher.elide_future_return = true;
    }
    if (task_produces_value) {
      Future result = runtime->execute_index_space(ctx, launcher, scalar_redop,
                                                   scalar_reduction_ordered);
      if (!elide_future_return) {
        uint64_t expected = compute_scalar_reduction_result();
        futures.push_back(std::pair(result, expected));
      }
    } else {
      runtime->execute_index_space(ctx, launcher);
    }

    // Now we run the task body on the shadow copy for verification.
    if (!launcher.region_requirements.empty()) {
      RegionRequirement &creq = launcher.region_requirements[0];
      for (uint64_t point = range_min; point <= range_max; ++point) {
        IndexSpace subspace =
            req.project(Point<1>(point), launch_domain).get_index_space();
        PhysicalRegion &shadow_inst = forest.get_shadow_inst();

        mutate_region(runtime, subspace, shadow_inst, creq.privilege, creq.redop,
                      creq.instance_fields, args);
      }
    }
  }

  void execute_single_task(Runtime *runtime, Context ctx,
                           std::vector<FutureCheck> &futures) {
    for (uint64_t point = range_min; point <= range_max; ++point) {
      PointTaskArgs args(task_arg_value);
      TaskLauncher launcher(task_id, TaskArgument(&args, sizeof(args)));
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
          futures.push_back(std::pair(result, task_arg_value));
        }
      } else {
        runtime->execute_task(ctx, launcher);
      }
      runtime->destroy_index_space(ctx, launch_space);

      // Now we run the task body on the shadow copy for verification.
      if (!launcher.region_requirements.empty()) {
        RegionRequirement &creq = launcher.region_requirements[0];
        IndexSpace subspace = creq.region.get_index_space();
        PhysicalRegion &shadow_inst = forest.get_shadow_inst();

        mutate_region(runtime, subspace, shadow_inst, creq.privilege, creq.redop,
                      creq.instance_fields, args);
      }
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

    LOG_ONCE(log_fuzz.info() << "  Task ID: " << task_name(task_id));
    LOG_ONCE(log_fuzz.info() << "  Launch domain: " << launch_domain);
    if (scalar_redop != LEGION_REDOP_LAST) {
      LOG_ONCE(log_fuzz.info() << "  Scalar redop: " << redop_name(scalar_redop));
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
  RegionForest &forest;
  LaunchType launch_type = LaunchType::INVALID;
  TaskID task_id = LEGION_MAX_APPLICATION_TASK_ID;
  bool task_produces_value = false;
  bool task_is_inner = false;
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
  uint64_t task_arg_value = 0;
};

void top_level(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx,
               Runtime *runtime) {
  InputArgs args = Runtime::get_input_args();
  FuzzerConfig config = FuzzerConfig::parse_args(args.argc, args.argv);
  config.log_config(runtime, ctx);

  RngSeed seed(config.initial_seed);
  RngStream rng = seed.make_stream();

  RegionForest forest(runtime, ctx, config, seed);

  std::vector<FutureCheck> futures;
  for (uint64_t op_idx = 0; op_idx < config.num_ops; ++op_idx) {
    OperationBuilder op(config, forest);
    op.build(rng);

    // It is VERY IMPORTANT that the random number generator is NOT USED
    // inside this if statement. Otherwise, we lose the ability to replay
    // while skipping operations.
    if (op_idx >= config.skip_ops) {
      LOG_ONCE(log_fuzz.info() << "Operation: " << op_idx);
      op.execute(runtime, ctx, futures);
    }
  }

  forest.verify_contents();

  for (FutureCheck &check : futures) {
    uint64_t result = check.first.get_result<uint64_t>();
    uint64_t expected = check.second;
    if (result != expected) {
      LOG_ONCE(log_fuzz.fatal()
               << "Bad future: " << result << ", expected: " << expected);
      abort();
    }
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
    TaskVariantRegistrar registrar(UINT64_LEAF_TASK_ID, "uint64_leaf");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<uint64_t, uint64_leaf>(registrar, "uint64_leaf");
  }

  {
    TaskVariantRegistrar registrar(VOID_INNER_TASK_ID, "void_inner");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_inner();
    Runtime::preregister_task_variant<void_inner>(registrar, "void_inner");
  }

  {
    TaskVariantRegistrar registrar(UINT64_INNER_TASK_ID, "uint64_inner");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_inner();
    Runtime::preregister_task_variant<uint64_t, uint64_inner>(registrar, "uint64_inner");
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
    TaskVariantRegistrar registrar(UINT64_REPLICABLE_LEAF_TASK_ID,
                                   "uint64_replicable_leaf");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    registrar.set_replicable();
    Runtime::preregister_task_variant<uint64_t, uint64_replicable_leaf>(
        registrar, "uint64_replicable_leaf");
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
    TaskVariantRegistrar registrar(UINT64_REPLICABLE_INNER_TASK_ID,
                                   "uint64_replicable_inner");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_inner();
    registrar.set_replicable();
    Runtime::preregister_task_variant<uint64_t, uint64_replicable_inner>(
        registrar, "uint64_replicable_inner");
  }

  return Runtime::start(argc, argv);
}
