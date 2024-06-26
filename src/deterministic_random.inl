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

template <typename T>
RngChannel RngStream::make_channel(const T &hashable) const {
  return RngChannel(seed, stream, hashable);
}

template <typename T>
RngChannel::RngChannel(uint64_t _seed, uint64_t _stream, const T &hashable)
    : seed(_seed), stream(_stream), channel(hash(hashable)), seq(0) {}

template <typename T, typename S>
void shuffle(T &rng, std::vector<S> &vec) {
  if (vec.empty()) {
    return;
  }

  for (uint64_t idx = vec.size() - 1; idx > 0; --idx) {
    uint64_t target = rng.uniform_range(0, idx);
    std::swap(vec[idx], vec[target]);
  }
}

template <typename T>
void RngStream::shuffle(std::vector<T> &vec) {
  ::shuffle(*this, vec);
}

template <typename T>
void RngChannel::shuffle(std::vector<T> &vec) {
  ::shuffle(*this, vec);
}
