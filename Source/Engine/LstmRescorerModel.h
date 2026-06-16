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

#ifndef SRC_ENGINE_LSTMRESCORERMODEL_H_
#define SRC_ENGINE_LSTMRESCORERMODEL_H_

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "NeuralRescorer.h"

namespace McBopomofo {

// A concrete RescorerModel: a hand-written forward pass of the char-level LSTM
// trained by Source/Data/rescorer-training. It loads the flat weight file
// produced by export_weights.py ("MCBLSTM1") and the parallel vocab.txt.
//
// Why hand-written rather than CoreML: the model is tiny (~1-3M params) and the
// rescoring loop needs fine control CoreML's static graph does not give us --
// per-step hidden-state handoff for prefix caching, and a *selective* softmax
// over just the few homophones at each position instead of the full vocabulary.
// Those two are what keep per-keystroke latency low; a full-vocab softmax (the
// proj is [vocab, hidden]) would dominate the cost.
//
// Math (single-layer LSTM, PyTorch formulation, gate order i, f, g, o):
//   i = sigmoid(W_ii x + b_ii + W_hi h + b_hi)
//   f = sigmoid(W_if x + b_if + W_hf h + b_hf)
//   g = tanh   (W_ig x + b_ig + W_hg h + b_hg)
//   o = sigmoid(W_io x + b_io + W_ho h + b_ho)
//   c' = f * c + i * g
//   h' = o * tanh(c')
//   logits = proj.weight * h' + proj.bias
// PyTorch keeps both bias vectors (b_ih and b_hh); we add them, matching it.
class LstmRescorerModel : public RescorerModel {
 public:
  // Loads the model from the weights file (export_weights.py output) and the
  // id-ordered vocab.txt. Throws std::runtime_error on any malformed input
  // (bad magic, truncated tensors, missing <bos>/<unk>) so the caller can fall
  // back to running without a rescorer. int8 weights are dequantized to float
  // at load time; inference is always float.
  static std::shared_ptr<LstmRescorerModel> Load(const std::string& weightsPath,
                                                 const std::string& vocabPath);

  RescorerModelState initialState() override;

  // log P(nextValue | prefix) under a selective softmax over `homophones` (the
  // values sharing nextValue's reading at this position), plus the advanced
  // state. A position with <=1 distinct candidate cannot discriminate between
  // sentences, so it contributes 0 to the score and only advances the state --
  // this is both correct (the term is identical across candidates) and the
  // reason most positions cost just one LSTM step with no softmax.
  std::pair<double, RescorerModelState> step(
      const RescorerModelState& prevState, const std::string& nextValue,
      const std::vector<std::string>& homophones) override;

  [[nodiscard]] uint32_t vocabSize() const { return vocab_; }
  [[nodiscard]] uint32_t embedDim() const { return embed_; }
  [[nodiscard]] uint32_t hiddenDim() const { return hidden_; }

 private:
  LstmRescorerModel() = default;

  // Maps a value (possibly multi-character) to its char ids, unknown chars to
  // <unk>. The vocab is char-level, so we split on Unicode code points.
  [[nodiscard]] std::vector<int> tokenize(const std::string& value) const;

  // One LSTM time step: consumes the embedding of token `tokenId` from
  // (hPrev, cPrev) and writes (hOut, cOut). Pointers are length hidden_.
  void lstmStep(int tokenId, const float* hPrev, const float* cPrev,
                float* hOut, float* cOut) const;

  // Unnormalized score (sum of per-char proj logits, autoregressive) of `ids`
  // starting from (hIn, cIn). Also returns the resulting state in (hOut, cOut)
  // when they are non-null. This is the value's contribution to the selective
  // softmax numerator/denominator; for single-character values it is exactly
  // the proj logit of that character, so the softmax over the homophone set is
  // exact. (For multi-character homophone values of differing length it is an
  // approximation; such sets are rare in practice.)
  double scoreValue(const std::vector<int>& ids, const float* hIn,
                    const float* cIn, std::vector<float>* hOut,
                    std::vector<float>* cOut) const;

  uint32_t vocab_ = 0;
  uint32_t embed_ = 0;
  uint32_t hidden_ = 0;

  // Weights, all dequantized to float. Row-major, matching export order.
  std::vector<float> embedWeight_;  // [vocab, embed]
  std::vector<float> wIh_;          // [4*hidden, embed]
  std::vector<float> wHh_;          // [4*hidden, hidden]
  std::vector<float> bIh_;          // [4*hidden]
  std::vector<float> bHh_;          // [4*hidden]
  std::vector<float> projWeight_;   // [vocab, hidden]
  std::vector<float> projBias_;     // [vocab]

  std::vector<std::string> idToToken_;
  std::unordered_map<std::string, int> tokenToId_;
  int bosId_ = -1;
  int unkId_ = -1;

  RescorerModelState initialState_;  // cached: state after consuming <bos>
};

}  // namespace McBopomofo

#endif  // SRC_ENGINE_LSTMRESCORERMODEL_H_
