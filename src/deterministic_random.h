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

// A deterministic "random" number generator, built on SipHash.

// We generate "random" numbers by hashing three values:
//
//  * A "seed" (some number fixed at the beginning of a run).
//
//  * A "stream" (assigned to each task, so we can query random numbers in
//    parallel in a deterministic fashion).
//
//  * A "sequence number" (gets incremented in every task so that we never
//    generate the same number twice).
//
// SipHash is not a cryptographically strong hash function, but it is strong
// enough to ensure the results are not statistically biased.

#ifndef DETERMINISTIC_RANDOM_H_
#define DETERMINISTIC_RANDOM_H_

#include <cstdint>
#include <type_traits>

#include "hasher.h"

class RngStream;
class RngChannel;

class RngSeed {
public:
  RngSeed();
  RngSeed(uint64_t seed);

  RngSeed(const RngSeed &rng) = delete;
  RngSeed(RngSeed &&rng);
  RngSeed &operator=(const RngSeed &rng) = delete;
  RngSeed &operator=(RngSeed &&rng);

  RngStream make_stream();

private:
  uint64_t seed;
  uint64_t stream;
};

class RngStream {
private:
  RngStream(uint64_t seed, uint64_t stream);

public:
  uint64_t uniform_uint64_t();
  uint64_t uniform_range(uint64_t range_lo, uint64_t range_hi /* inclusive */);
  template <typename T>
  void shuffle(std::vector<T> &vec);

  template <typename T>
  RngChannel make_channel(const T &hashable) const;

private:
  const uint64_t seed;
  const uint64_t stream;
  uint64_t seq;

  friend class RngSeed;
};
static_assert(std::is_trivially_copyable_v<RngStream>);

class RngChannel {
private:
  template <typename T>
  RngChannel(uint64_t seed, uint64_t stream, const T &hashable);

public:
  uint64_t uniform_uint64_t();
  uint64_t uniform_range(uint64_t range_lo, uint64_t range_hi /* inclusive */);
  template <typename T>
  void shuffle(std::vector<T> &vec);

private:
  const uint64_t seed;
  const uint64_t stream;
  const uint64_t channel;
  uint64_t seq;

  friend class RngStream;
};
static_assert(std::is_trivially_copyable_v<RngChannel>);

#include "deterministic_random.inl"

#endif  // DETERMINISTIC_RANDOM_H_
