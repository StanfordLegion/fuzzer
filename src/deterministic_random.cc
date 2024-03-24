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

static void gen_bits(const uint8_t *input, size_t input_bytes, uint8_t *output,
                     size_t output_bytes) {
  // To generate deterministic uniformly distributed bits, run a hash
  // function on the seed and use the hash value as the output.
  const uint8_t k[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  siphash(input, input_bytes, k, output, output_bytes);
}

uint64_t uniform_uint64_t(const uint64_t seed, const uint64_t stream,
                          uint64_t &sequence_number) {
  const uint64_t input[3] = {seed, stream, sequence_number++};
  uint64_t result;
  gen_bits(reinterpret_cast<const uint8_t *>(&input), sizeof(input),
           reinterpret_cast<uint8_t *>(&result), sizeof(result));
  return result;
}

uint64_t uniform_range(const uint64_t seed, const uint64_t stream,
                       uint64_t &sequence_number, uint64_t range_lo,
                       uint64_t range_hi /* inclusive */) {
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
    bits = uniform_uint64_t(seed, stream, sequence_number);
  } while (bits >= UINT64_MAX - remainder);
  return range_lo + (bits % range_size);
}
