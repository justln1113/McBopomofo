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

#ifndef SRC_ENGINE_NEURALRESCORER_H_
#define SRC_ENGINE_NEURALRESCORER_H_

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "gramambular2/reading_grid.h"

namespace McBopomofo {

// An opaque per-step state carried through the model as it consumes a sentence
// left-to-right. For a real recurrent model `hidden` holds the hidden vector
// (LSTM) or accumulated cache (Transformer KV). `context` is a small free-form
// tag a model may use to carry short discrete context (e.g. the previous value
// for a bigram-style model or mock) without abusing a member variable -- which
// would break correctness under prefix caching, since cache hits skip step()
// and would leave any such member stale.
//
// The state MUST be self-contained: step() must be a pure function of
// (prevState, value), so that a cached state can be resumed at any time.
struct RescorerModelState {
  std::vector<float> hidden;
  std::string context;
};

// Abstract second-pass language model used to re-rank the n-best candidate
// sentences produced by ReadingGrid::walkNBest.
//
// The interface is deliberately *incremental* (one character/value at a time)
// rather than whole-sentence. This single shape expresses everything the
// rescoring loop needs to stay real-time:
//
//   - Incremental extension: step() advances one position from a prior state,
//     so appending a character costs one step, not a full re-encode.
//   - Prefix caching: because step() is a pure function of (state, value), the
//     NeuralRescorer can cache the state after each prefix and share it across
//     candidate sentences that agree on that prefix.
//   - Selective softmax: step() receives the homophone set at the position, so
//     a real model can normalize over only those few characters instead of the
//     whole vocabulary -- the key to keeping per-keystroke latency low.
//
// Implementations must be deterministic and side-effect free.
class RescorerModel {
 public:
  virtual ~RescorerModel() = default;

  // The state representing the empty prefix (beginning of sentence).
  virtual RescorerModelState initialState() = 0;

  // Scores the next value given the prior state, and returns the advanced
  // state. The returned score is a log probability: log P(nextValue | prefix).
  // homophones is the set of values sharing nextValue's reading at this
  // position, i.e. the denominator set for a selective softmax. It may be empty
  // for mock models that do not need it.
  virtual std::pair<double, RescorerModelState> step(
      const RescorerModelState& prevState, const std::string& nextValue,
      const std::vector<std::string>& homophones) = 0;
};

// Re-ranks n-best candidate sentences by combining the grid's unigram score
// with a second-pass model score. The model can apply longer-range context than
// the unigram walk, which is what fixes homophone choices whose disambiguating
// evidence lies several words away (e.g. choosing 依你 over 一你).
class NeuralRescorer {
 public:
  using NBestPath = Formosa::Gramambular2::ReadingGrid::NBestPath;

  // Provides the homophone set (selective-softmax denominator) for the value at
  // a given position in a candidate's value sequence. The default returns an
  // empty set; the real wiring supplies grid candidates here.
  using HomophoneProvider =
      std::function<std::vector<std::string>(size_t position)>;

  explicit NeuralRescorer(std::shared_ptr<RescorerModel> model,
                          double lambda = 1.0)
      : model_(std::move(model)), lambda_(lambda) {}

  // The interpolation weight applied to the model score:
  //   finalScore = unigramScore + lambda * modelScore
  void setLambda(double lambda) { lambda_ = lambda; }
  [[nodiscard]] double lambda() const { return lambda_; }

  // Re-ranks the candidates and returns the index of the best one. Returns 0 if
  // candidates is empty-safe (caller should check emptiness). The combined
  // score for each candidate is also written back into a parallel vector if
  // outScores is non-null.
  size_t rerankBestIndex(const std::vector<NBestPath>& candidates,
                         const HomophoneProvider& homophones = nullptr,
                         std::vector<double>* outScores = nullptr);

  // Convenience: returns the highest-scoring candidate (by value). Caller must
  // ensure candidates is non-empty.
  const NBestPath& rerank(const std::vector<NBestPath>& candidates,
                          const HomophoneProvider& homophones = nullptr);

  // Clears the prefix cache. Call when the sentence prefix changes in a way that
  // invalidates cached states (e.g. the user edits an earlier character).
  void clearCache() { prefixCache_.clear(); }

  // Diagnostics: number of model steps actually executed vs served from cache
  // during the last rerank call. Useful for verifying that prefix sharing works.
  [[nodiscard]] size_t lastStepsComputed() const { return lastStepsComputed_; }
  [[nodiscard]] size_t lastStepsCached() const { return lastStepsCached_; }

 private:
  // Cached cumulative score and state for a consumed value prefix.
  struct CachedPrefix {
    double cumulativeLogProb = 0.0;
    RescorerModelState state;
  };

  double modelScoreForPath(const NBestPath& path,
                           const HomophoneProvider& homophones);

  std::shared_ptr<RescorerModel> model_;
  double lambda_;
  std::unordered_map<std::string, CachedPrefix> prefixCache_;
  size_t lastStepsComputed_ = 0;
  size_t lastStepsCached_ = 0;
};

}  // namespace McBopomofo

#endif  // SRC_ENGINE_NEURALRESCORER_H_
