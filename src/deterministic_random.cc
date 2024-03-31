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

#include "deterministic_random.h"

#include <cstdlib>

#include "siphash.h"

RngSeed::RngSeed() : seed(UINT64_MAX), stream(0) {}

RngSeed::RngSeed(uint64_t _seed) : seed(_seed), stream(0) {}

RngStream RngSeed::make_stream() {
  if (seed == UINT64_MAX) {
    abort();  // use of invalid RngSeed
  }
  RngStream result(seed, stream++);
  return result;
}

RngSeed::RngSeed(RngSeed &&rng) : seed(rng.seed), stream(rng.stream) {
  rng.seed = UINT64_MAX;
  rng.stream = 0;
}

RngSeed &RngSeed::operator=(RngSeed &&rng) {
  seed = rng.seed;
  stream = rng.stream;
  rng.seed = UINT64_MAX;
  rng.stream = 0;
  return *this;
}

RngStream::RngStream(uint64_t _seed, uint64_t _stream)
    : seed(_seed), stream(_stream), seq(0) {}

static void gen_bits(const uint8_t *input, size_t input_bytes, uint8_t *output,
                     size_t output_bytes) {
  // To generate deterministic uniformly distributed bits, run a hash
  // function on the seed and use the hash value as the output.
  const uint8_t k[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  siphash(input, input_bytes, k, output, output_bytes);
}

template <typename T>
uint64_t uniform_range(T &rng, uint64_t range_lo, uint64_t range_hi /* inclusive */) {
  if (range_hi <= range_lo) {
    return range_lo;
  }

  uint64_t range_size = range_hi - range_lo + 1;
  uint64_t remainder = UINT64_MAX % range_size;

  // To avoid bias, we reject results that use the final
  // `UINT64_MAX - remainder` values. In practice this should almost never
  // loop (for small ranges), so the expected trip count is 1.
  uint64_t bits;
  do {
    bits = rng.uniform_uint64_t();
  } while (bits >= UINT64_MAX - remainder);
  return range_lo + (bits % range_size);
}

uint64_t RngStream::uniform_uint64_t() {
  const uint64_t input[3] = {seed, stream, seq++};
  uint64_t result;
  gen_bits(reinterpret_cast<const uint8_t *>(&input), sizeof(input),
           reinterpret_cast<uint8_t *>(&result), sizeof(result));
  return result;
}

uint64_t RngStream::uniform_range(uint64_t range_lo, uint64_t range_hi /* inclusive */) {
  return ::uniform_range(*this, range_lo, range_hi);
}

uint64_t RngChannel::uniform_uint64_t() {
  const uint64_t input[4] = {seed, stream, channel, seq++};
  uint64_t result;
  gen_bits(reinterpret_cast<const uint8_t *>(&input), sizeof(input),
           reinterpret_cast<uint8_t *>(&result), sizeof(result));
  return result;
}

uint64_t RngChannel::uniform_range(uint64_t range_lo, uint64_t range_hi /* inclusive */) {
  return ::uniform_range(*this, range_lo, range_hi);
}
