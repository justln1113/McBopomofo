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

// Readings that begin with '_' are non-linguistic (punctuation, half-width
// symbols, macros). Their candidates share a tied score and the dictionary's
// listing order is the intended default; a fluency model has no meaningful
// opinion about 『 vs 《, so it must not be allowed to reorder them.
bool IsSpecialReading(const std::string& reading) {
  return !reading.empty() && reading.front() == '_';
}

// The (reading, value) sequence of a path's special-reading nodes, in order.
// Special nodes always span exactly one reading and never merge into phrases,
// so this sequence is directly comparable across differently segmented paths.
std::vector<std::pair<std::string, std::string>> SpecialNodes(
    const Formosa::Gramambular2::ReadingGrid::NBestPath& path) {
  std::vector<std::pair<std::string, std::string>> out;
  for (size_t i = 0; i < path.readings.size() && i < path.values.size(); ++i) {
    if (IsSpecialReading(path.readings[i])) {
      out.emplace_back(path.readings[i], path.values[i]);
    }
  }
  return out;
}
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

  // Explicit user intent must not be overridden by a statistical model. The
  // top candidate (index 0) already reflects any user override / user-phrase
  // boost via its unigram score, so if it carries an override we keep it as-is
  // and skip re-ranking entirely. (We check candidates[0] specifically: that is
  // the walk's chosen path, the one the user's override/boost was designed to
  // make win.) See NBestPath::overridden for the known gap re: user-phrase
  // provenance that this flag does not yet cover.
  if (candidates[0].hasUserOverride()) {
    if (outScores != nullptr) {
      (*outScores)[0] = candidates[0].score;
    }
    return 0;
  }

  size_t bestIndex = 0;
  double bestScore = -std::numeric_limits<double>::infinity();
  const auto topSpecialNodes = SpecialNodes(candidates[0]);

  for (size_t i = 0; i < candidates.size(); ++i) {
    const NBestPath& path = candidates[i];
    // A candidate may only differ from the walk's choice in linguistic
    // (Han) positions; one that swaps a punctuation/symbol value is out.
    if (i > 0 && SpecialNodes(path) != topSpecialNodes) {
      if (outScores != nullptr) {
        (*outScores)[i] = -std::numeric_limits<double>::infinity();
      }
      continue;
    }
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
