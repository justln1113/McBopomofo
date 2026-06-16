// Copyright (c) 2025 and onwards The McBopomofo Authors.
//
// Permission is hereby granted, free of charge, to any person
// obtaining a copy of this software and associated documentation
// files (the "Software"), to deal in the Software without
// restriction, including without limitation the rights to use,
// copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following
// conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
// OTHER DEALINGS IN THE SOFTWARE.

#include "NeuralRescorer.h"

#include <limits>
#include <utility>

namespace McBopomofo {

namespace {
// Separator between values when forming a prefix cache key. Uses a control
// character unlikely to appear in candidate values.
constexpr char kPrefixKeySeparator = '\x1f';
}  // namespace

double NeuralRescorer::modelScoreForPath(
    const NBestPath& path, const HomophoneProvider& homophones) {
  // Walk the value sequence left-to-right. After consuming each value we have a
  // prefix; cache the (cumulative log prob, model state) keyed by that prefix so
  // that other candidates sharing the same prefix reuse the work instead of
  // re-running the model from the start. This is what keeps the amortized cost
  // close to "one step per genuinely new character" rather than O(N * L).
  std::string prefixKey;
  RescorerModelState state = model_->initialState();
  double cumulative = 0.0;

  for (size_t i = 0; i < path.values.size(); ++i) {
    const std::string& value = path.values[i];
    prefixKey += value;
    prefixKey += kPrefixKeySeparator;

    auto cached = prefixCache_.find(prefixKey);
    if (cached != prefixCache_.end()) {
      cumulative = cached->second.cumulativeLogProb;
      state = cached->second.state;
      ++lastStepsCached_;
      continue;
    }

    std::vector<std::string> homophoneSet;
    if (homophones) {
      homophoneSet = homophones(i);
    }
    auto [logProb, nextState] = model_->step(state, value, homophoneSet);
    cumulative += logProb;
    state = std::move(nextState);
    ++lastStepsComputed_;

    prefixCache_.emplace(prefixKey,
                         CachedPrefix{cumulative, state});
  }

  return cumulative;
}

size_t NeuralRescorer::rerankBestIndex(
    const std::vector<NBestPath>& candidates,
    const HomophoneProvider& homophones, std::vector<double>* outScores) {
  lastStepsComputed_ = 0;
  lastStepsCached_ = 0;

  if (outScores != nullptr) {
    outScores->assign(candidates.size(), 0.0);
  }
  if (candidates.empty()) {
    return 0;
  }

  size_t bestIndex = 0;
  double bestScore = -std::numeric_limits<double>::infinity();

  for (size_t i = 0; i < candidates.size(); ++i) {
    const NBestPath& path = candidates[i];
    double modelScore = modelScoreForPath(path, homophones);
    double combined = path.score + lambda_ * modelScore;
    if (outScores != nullptr) {
      (*outScores)[i] = combined;
    }
    // Strictly greater so ties keep the earlier (higher unigram) candidate,
    // preserving the original walk ordering when the model is indifferent.
    if (combined > bestScore) {
      bestScore = combined;
      bestIndex = i;
    }
  }

  return bestIndex;
}

const NeuralRescorer::NBestPath& NeuralRescorer::rerank(
    const std::vector<NBestPath>& candidates,
    const HomophoneProvider& homophones) {
  size_t index = rerankBestIndex(candidates, homophones);
  return candidates[index];
}

}  // namespace McBopomofo
