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

enum HashTypeIDs {
  INT32_T_TYPE_ID,
  UINT32_T_TYPE_ID,
  INT64_T_TYPE_ID,
  UINT64_T_TYPE_ID,
  STD_PAIR_TYPE_ID,
  LEGION_DOMAIN_POINT_TYPE_ID,
  LEGION_DOMAIN_TYPE_ID,
};

class Hasher {
public:
  template <typename T>
  void hash(const T &value) {
    hash_type_tag(value);
    hash_value(value);
  }

  uint64_t result();

private:
  void hash_type_tag(int32_t value) { hash_raw(INT32_T_TYPE_ID); }
  void hash_type_tag(uint32_t value) { hash_raw(UINT32_T_TYPE_ID); }
  void hash_type_tag(int64_t value) { hash_raw(INT64_T_TYPE_ID); }
  void hash_type_tag(uint64_t value) { hash_raw(UINT64_T_TYPE_ID); }
  template <typename T, typename U>
  void hash_type_tag(const std::pair<T, U> &value) {
    hash_raw(STD_PAIR_TYPE_ID);
    hash_type_tag(value.first);
    hash_type_tag(value.second);
  }
  void hash_type_tag(const Legion::DomainPoint &value) {
    hash_raw(LEGION_DOMAIN_POINT_TYPE_ID);
  }
  void hash_type_tag(const Legion::Domain &value) { hash_raw(LEGION_DOMAIN_TYPE_ID); }

  void hash_value(int32_t value) { hash_raw(value); }
  void hash_value(uint32_t value) { hash_raw(value); }
  void hash_value(int64_t value) { hash_raw(value); }
  void hash_value(uint64_t value) { hash_raw(value); }
  template <typename T, typename U>
  void hash_value(const std::pair<T, U> &value) {
    hash_value(value.first);
    hash_value(value.second);
  }
  void hash_value(const Legion::DomainPoint &value) {
    int32_t dim = value.get_dim();
    hash_value(dim);
    for (int32_t idx = 0; idx < dim; ++idx) {
      hash_value(int64_t(value[idx]));
    }
  }
  void hash_value(const Legion::Domain &value) {
    if (!value.dense()) {
      abort();
    }
    int32_t dim = value.get_dim();
    hash_value(dim);
    for (int32_t idx = 0; idx < dim; ++idx) {
      hash_value(int64_t(value.lo()[idx]));
      hash_value(int64_t(value.hi()[idx]));
    }
  }

  template <typename T>
  void hash_raw(const T &value) {
    // Ensure we use this only on POD types with no padding.
    static_assert(std::is_trivially_copyable_v<T>);
    static_assert(std::has_unique_object_representations_v<T>);
    buffer.write(reinterpret_cast<const char *>(&value), sizeof(value));
  }

private:
  std::stringstream buffer;
};

template <typename T>
uint64_t hash(const T &value) {
  Hasher hasher;
  hasher.hash(value);
  return hasher.result();
}
