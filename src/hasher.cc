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

#include "hasher.h"

#include <sstream>

#include "siphash.h"

static void hash_bytes(const uint8_t *input, size_t input_bytes, uint8_t *output,
                       size_t output_bytes) {
  // Choose different magic numbers to avoid colliding with gen_bits.
  const uint8_t k[16] = {16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31};
  siphash(input, input_bytes, k, output, output_bytes);
}

Hasher::Hasher() {}

void Hasher::hash_type_tag(HashTypeTag type_tag) {
  buffer.write(reinterpret_cast<const char *>(&type_tag), sizeof(type_tag));
}

uint64_t Hasher::result() {
  uint64_t result;
  std::string content = std::move(buffer).str();
  static_assert(sizeof(char) == sizeof(uint8_t));
  hash_bytes(reinterpret_cast<const uint8_t *>(content.data()), content.size(),
             reinterpret_cast<uint8_t *>(&result), sizeof(result));
  return result;
}
