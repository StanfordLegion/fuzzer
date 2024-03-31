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

enum HashTypeTag {
  LONG_LONG_TYPE_ID,
  INT32_T_TYPE_ID,
  UINT32_T_TYPE_ID,
  // Clang thinks this is the same as long long, so we don't get to have nice things
  // INT64_T_TYPE_ID,
  UINT64_T_TYPE_ID,
  STD_PAIR_TYPE_ID,
  STD_VECTOR_TYPE_ID,
  LEGION_DOMAIN_POINT_TYPE_ID,
  LEGION_DOMAIN_TYPE_ID,
};

template <typename T>
class HashTypeTagAdapter {};

template <typename T>
class HashValueAdapter {};

class Hasher {
public:
  Hasher();

  template <typename T>
  void hash(const T &value);

  uint64_t result();

private:
  void hash_type_tag(HashTypeTag type_tag);

  template <typename T>
  void hash_value(const T &value);

  template <typename T, typename U>
  void hash_value(const std::pair<T, U> &value);
  template <typename T>
  void hash_value(const std::vector<T> &value);

  void hash_value(const Legion::DomainPoint &value);
  void hash_value(const Legion::Domain &value);

private:
  std::stringstream buffer;

  template <typename V>
  friend class HashTypeTagAdapter;

  template <typename V>
  friend class HashValueAdapter;
};

// The only way in C++ to avoid implicit type conversion is to use an explicit
// constructor. The adapters below jump through the necessary hoops to make
// sure we don't perform implicit conversions by accident.

#define DECLARE_TYPE_ADAPTER(W, W_TYPE_ID)                                           \
  template <>                                                                        \
  class HashTypeTagAdapter<W> {                                                      \
  private:                                                                           \
    explicit HashTypeTagAdapter(Hasher &hasher) { hasher.hash_type_tag(W_TYPE_ID); } \
    friend class Hasher;                                                             \
    template <typename V>                                                            \
    friend class HashTypeTagAdapter;                                                 \
  };

DECLARE_TYPE_ADAPTER(long long, LONG_LONG_TYPE_ID)
DECLARE_TYPE_ADAPTER(int32_t, INT32_T_TYPE_ID)
DECLARE_TYPE_ADAPTER(uint32_t, UINT32_T_TYPE_ID)
// DECLARE_TYPE_ADAPTER(int64_t, INT64_T_TYPE_ID)
DECLARE_TYPE_ADAPTER(uint64_t, UINT64_T_TYPE_ID)
DECLARE_TYPE_ADAPTER(Legion::DomainPoint, LEGION_DOMAIN_POINT_TYPE_ID)
DECLARE_TYPE_ADAPTER(Legion::Domain, LEGION_DOMAIN_TYPE_ID)
#undef DECLARE_TYPE_ADAPTER

template <typename T, typename U>
class HashTypeTagAdapter<std::pair<T, U>> {
private:
  explicit HashTypeTagAdapter(Hasher &hasher) {
    hasher.hash_type_tag(STD_PAIR_TYPE_ID);
    ::HashTypeTagAdapter<T>{hasher};
    ::HashTypeTagAdapter<U>{hasher};
  }
  friend class Hasher;
  template <typename V>
  friend class HashTypeTagAdapter;
};

template <typename T>
class HashTypeTagAdapter<std::vector<T>> {
private:
  explicit HashTypeTagAdapter(Hasher &hasher) {
    hasher.hash_type_tag(STD_VECTOR_TYPE_ID);
    ::HashTypeTagAdapter<T>{hasher};
  }
  friend class Hasher;
  template <typename V>
  friend class HashTypeTagAdapter;
};

#define DECLARE_SIMPLE_VALUE_ADAPTER(W)                         \
  template <>                                                   \
  class HashValueAdapter<W> {                                   \
  private:                                                      \
    explicit HashValueAdapter(Hasher &hasher, const W &value) { \
      hasher.hash_value(value);                                 \
    }                                                           \
    friend class Hasher;                                        \
    template <typename V>                                       \
    friend class HashValueAdapter;                              \
  };

DECLARE_SIMPLE_VALUE_ADAPTER(long long)
DECLARE_SIMPLE_VALUE_ADAPTER(int32_t)
DECLARE_SIMPLE_VALUE_ADAPTER(uint32_t)
// DECLARE_SIMPLE_VALUE_ADAPTER(int64_t)
DECLARE_SIMPLE_VALUE_ADAPTER(uint64_t)
#undef DECLARE_SIMPLE_VALUE_ADAPTER

template <typename T, typename U>
class HashValueAdapter<std::pair<T, U>> {
private:
  explicit HashValueAdapter(Hasher &hasher, const std::pair<T, U> &value) {
    ::HashValueAdapter<T>{hasher, value.first};
    ::HashValueAdapter<U>{hasher, value.second};
  }
  friend class Hasher;
  template <typename V>
  friend class HashValueAdapter;
};

template <typename T>
class HashValueAdapter<std::vector<T>> {
private:
  explicit HashValueAdapter(Hasher &hasher, const std::vector<T> &value) {
    ::HashValueAdapter{hasher, value.size()};
    for (const T &elem : value) {
      ::HashValueAdapter<T>{hasher, elem};
    }
  }
  friend class Hasher;
  template <typename V>
  friend class HashValueAdapter;
};

template <>
class HashValueAdapter<Legion::DomainPoint> {
private:
  explicit HashValueAdapter(Hasher &hasher, const Legion::DomainPoint &value) {
    int dim = value.get_dim();
    ::HashValueAdapter<int>{hasher, dim};
    for (int idx = 0; idx < dim; ++idx) {
      ::HashValueAdapter<Legion::coord_t>{hasher, value[idx]};
    }
  }
  friend class Hasher;
  template <typename V>
  friend class HashValueAdapter;
};

template <>
class HashValueAdapter<Legion::Domain> {
private:
  explicit HashValueAdapter(Hasher &hasher, const Legion::Domain &value) {
    if (!value.dense()) {
      abort();
    }
    int dim = value.get_dim();
    ::HashValueAdapter<int>{hasher, dim};
    for (int idx = 0; idx < dim; ++idx) {
      ::HashValueAdapter<Legion::coord_t>{hasher, value.lo()[idx]};
      ::HashValueAdapter<Legion::coord_t>{hasher, value.hi()[idx]};
    }
  }
  friend class Hasher;
  template <typename V>
  friend class HashValueAdapter;
};

template <typename T>
void Hasher::hash(const T &value) {
  HashTypeTagAdapter<T>{*this};
  HashValueAdapter<T>{*this, value};
}

template <typename T>
void Hasher::hash_value(const T &value) {
  // Ensure we use this only on POD types with no padding.
  static_assert(std::is_trivially_copyable_v<T>);
  static_assert(std::has_unique_object_representations_v<T>);
  buffer.write(reinterpret_cast<const char *>(&value), sizeof(value));
}

template <typename T>
uint64_t hash(const T &value) {
  Hasher hasher;
  hasher.hash(value);
  return hasher.result();
}
