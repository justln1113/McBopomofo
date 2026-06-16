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

#include "LstmRescorerModel.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "NeuralRescorer.h"

namespace McBopomofo {

namespace {

// A deliberately tiny model (embed=1, hidden=1) chosen so every value below is
// hand-computable, which catches gate-order / indexing transcription bugs that
// invariant-only tests would miss. Weights are written in the exact MCBLSTM1
// layout export_weights.py produces.
//
// vocab (id-ordered): <pad> <bos> <eos> <unk> 甲 乙
// embed:    bos=1, 甲=1, 乙=1, rest 0
// W_ih:     gate order i,f,g,o -> [0, 0, 1, 0]   (only the cell gate g is active)
// W_hh:     all 0
// biases:   all 0
// proj.W:   all 0   (so logits depend only on bias -> independent of hidden)
// proj.b:   甲=+1, 乙=-1, rest 0
//
// Hand-computed (see the matching python trace):
//   initial state after <bos>:  h = 0.1816997422,  c = 0.3807970780
//   after stepping 甲:           h = 0.2581184019,  c = 0.5711956170
//   selective softmax over {甲,乙} from initial: logP(甲) = -0.1269280110,
//                                                logP(乙) = -2.1269280110
constexpr double kInitH = 0.1816997422;
constexpr double kInitC = 0.3807970780;
constexpr double kAfterJiaH = 0.2581184019;
constexpr double kAfterJiaC = 0.5711956170;
constexpr double kLogPJia = -0.1269280110;
constexpr double kLogPYi = -2.1269280110;

void PutU32(std::ofstream& f, uint32_t v) {
  f.write(reinterpret_cast<const char*>(&v), sizeof(v));  // little-endian host
}

void PutF32(std::ofstream& f, float v) {
  f.write(reinterpret_cast<const char*>(&v), sizeof(v));
}

void PutFloats(std::ofstream& f, const std::vector<float>& xs) {
  for (float x : xs) {
    PutF32(f, x);
  }
}

class LstmRescorerModelTest : public ::testing::Test {
 protected:
  void SetUp() override {
    namespace fs = std::filesystem;
    auto dir = fs::temp_directory_path();
    weightsPath_ = (dir / "mcb_lstm_test_weights.bin").string();
    vocabPath_ = (dir / "mcb_lstm_test_vocab.txt").string();

    {
      std::ofstream wf(weightsPath_, std::ios::binary | std::ios::trunc);
      const char magic[8] = {'M', 'C', 'B', 'L', 'S', 'T', 'M', '1'};
      wf.write(magic, sizeof(magic));
      PutU32(wf, 0);  // dtype = float32
      PutU32(wf, 6);  // vocab
      PutU32(wf, 1);  // embed
      PutU32(wf, 1);  // hidden
      PutU32(wf, 0);  // reserved
      PutFloats(wf, {0, 1, 0, 0, 1, 1});  // embed.weight [6,1]
      PutFloats(wf, {0, 0, 1, 0});        // lstm W_ih [4,1] (i,f,g,o)
      PutFloats(wf, {0, 0, 0, 0});        // lstm W_hh [4,1]
      PutFloats(wf, {0, 0, 0, 0});        // lstm b_ih [4]
      PutFloats(wf, {0, 0, 0, 0});        // lstm b_hh [4]
      PutFloats(wf, {0, 0, 0, 0, 0, 0});  // proj.weight [6,1]
      PutFloats(wf, {0, 0, 0, 0, 1, -1}); // proj.bias [6]
    }
    {
      std::ofstream vf(vocabPath_, std::ios::trunc);
      vf << "<pad>\n<bos>\n<eos>\n<unk>\n\xe7\x94\xb2\n\xe4\xb9\x99\n";  // 甲 乙
    }
  }

  void TearDown() override {
    std::error_code ec;
    std::filesystem::remove(weightsPath_, ec);
    std::filesystem::remove(vocabPath_, ec);
  }

  std::shared_ptr<LstmRescorerModel> Load() {
    return LstmRescorerModel::Load(weightsPath_, vocabPath_);
  }

  std::string weightsPath_;
  std::string vocabPath_;
};

const std::string kJia = "\xe7\x94\xb2";  // 甲
const std::string kYi = "\xe4\xb9\x99";   // 乙

}  // namespace

TEST_F(LstmRescorerModelTest, LoadsHeaderAndDims) {
  auto model = Load();
  EXPECT_EQ(model->vocabSize(), 6u);
  EXPECT_EQ(model->embedDim(), 1u);
  EXPECT_EQ(model->hiddenDim(), 1u);
}

TEST_F(LstmRescorerModelTest, BadMagicThrows) {
  std::ofstream(weightsPath_, std::ios::binary | std::ios::trunc) << "NOTMAGIC";
  EXPECT_THROW(Load(), std::runtime_error);
}

TEST_F(LstmRescorerModelTest, InitialStateMatchesHandComputed) {
  auto model = Load();
  RescorerModelState s = model->initialState();
  ASSERT_EQ(s.hidden.size(), 2u);  // [h, c] for hidden=1
  EXPECT_NEAR(s.hidden[0], kInitH, 1e-5);
  EXPECT_NEAR(s.hidden[1], kInitC, 1e-5);
}

TEST_F(LstmRescorerModelTest, StepScoresAndAdvancesState) {
  auto model = Load();
  auto [logProb, next] = model->step(model->initialState(), kJia, {kJia, kYi});
  EXPECT_NEAR(logProb, kLogPJia, 1e-5);
  ASSERT_EQ(next.hidden.size(), 2u);
  EXPECT_NEAR(next.hidden[0], kAfterJiaH, 1e-5);
  EXPECT_NEAR(next.hidden[1], kAfterJiaC, 1e-5);
}

TEST_F(LstmRescorerModelTest, SelectiveSoftmaxNormalizesOverHomophones) {
  auto model = Load();
  auto [lpJia, s1] = model->step(model->initialState(), kJia, {kJia, kYi});
  auto [lpYi, s2] = model->step(model->initialState(), kYi, {kJia, kYi});
  EXPECT_NEAR(lpJia, kLogPJia, 1e-5);
  EXPECT_NEAR(lpYi, kLogPYi, 1e-5);
  // A softmax over the homophone set: the probabilities must sum to 1.
  EXPECT_NEAR(std::exp(lpJia) + std::exp(lpYi), 1.0, 1e-6);
  // And the model prefers 甲 (proj.bias favors it).
  EXPECT_GT(lpJia, lpYi);
}

TEST_F(LstmRescorerModelTest, SingleCandidatePositionContributesZero) {
  auto model = Load();
  // No homophones, or just the value itself: nothing to discriminate -> 0.
  auto [lpEmpty, s1] = model->step(model->initialState(), kJia, {});
  auto [lpSelf, s2] = model->step(model->initialState(), kJia, {kJia});
  EXPECT_DOUBLE_EQ(lpEmpty, 0.0);
  EXPECT_DOUBLE_EQ(lpSelf, 0.0);
  // The advanced state must still be correct regardless of scoring.
  EXPECT_NEAR(s1.hidden[0], kAfterJiaH, 1e-5);
}

TEST_F(LstmRescorerModelTest, StateIsSelfContained) {
  auto model = Load();
  RescorerModelState original = model->initialState();
  RescorerModelState copy = original;  // independent copy
  auto [lp1, n1] = model->step(original, kJia, {kJia, kYi});
  auto [lp2, n2] = model->step(copy, kJia, {kJia, kYi});
  // Stepping from a copied state must give identical results, proving step() is
  // a pure function of (state, value) with no hidden member dependence.
  EXPECT_DOUBLE_EQ(lp1, lp2);
  ASSERT_EQ(n1.hidden.size(), n2.hidden.size());
  EXPECT_FLOAT_EQ(n1.hidden[0], n2.hidden[0]);
  EXPECT_FLOAT_EQ(n1.hidden[1], n2.hidden[1]);
}

TEST_F(LstmRescorerModelTest, UnknownCharMapsToUnk) {
  auto model = Load();
  // A char not in the vocab should not crash; it maps to <unk> and advances.
  auto [logProb, next] = model->step(model->initialState(), "\xe9\xbe\x9c",
                                     {});  // 龜, not in vocab
  EXPECT_DOUBLE_EQ(logProb, 0.0);
  EXPECT_EQ(next.hidden.size(), 2u);
}

// End-to-end through NeuralRescorer: the model should flip the walk's top choice
// when its sentence-level judgement disagrees -- the whole point of rescoring.
// Mirrors 依你 (preferred) winning over a higher-unigram 一你.
TEST_F(LstmRescorerModelTest, RerankFlipsToModelPreferredCandidate) {
  auto model = Load();
  NeuralRescorer rescorer(model, /*lambda=*/1.0);

  using NBestPath = NeuralRescorer::NBestPath;
  // candidates[0] = 乙 is the walk's top (higher unigram score), candidates[1]
  // = 甲 is slightly lower. The model strongly prefers 甲, enough to overcome
  // the 0.5 unigram gap.
  NBestPath yiPath;
  yiPath.values = {kYi};
  yiPath.readings = {"r"};
  yiPath.overridden = {false};
  yiPath.fromUserPhrase = {false};
  yiPath.score = 0.0;

  NBestPath jiaPath;
  jiaPath.values = {kJia};
  jiaPath.readings = {"r"};
  jiaPath.overridden = {false};
  jiaPath.fromUserPhrase = {false};
  jiaPath.score = -0.5;

  std::vector<NBestPath> candidates = {yiPath, jiaPath};
  auto homophones = [](size_t) -> std::vector<std::string> {
    return {kJia, kYi};
  };

  // combined(乙) = 0.0  + (-2.1269) = -2.1269
  // combined(甲) = -0.5 + (-0.1269) = -0.6269  -> wins
  size_t best = rescorer.rerankBestIndex(candidates, homophones);
  EXPECT_EQ(best, 1u);

  // With lambda = 0 the model is ignored and the walk's top (乙) is kept.
  rescorer.setLambda(0.0);
  rescorer.clearCache();
  EXPECT_EQ(rescorer.rerankBestIndex(candidates, homophones), 0u);
}

// A path carrying explicit user intent must never be reranked away.
TEST_F(LstmRescorerModelTest, UserOverrideIsNotReranked) {
  auto model = Load();
  NeuralRescorer rescorer(model, /*lambda=*/1.0);

  using NBestPath = NeuralRescorer::NBestPath;
  NBestPath overridden;
  overridden.values = {kYi};  // model dislikes 乙, but the user chose it
  overridden.readings = {"r"};
  overridden.overridden = {true};
  overridden.fromUserPhrase = {false};
  overridden.score = 0.0;

  NBestPath alt;
  alt.values = {kJia};
  alt.readings = {"r"};
  alt.overridden = {false};
  alt.fromUserPhrase = {false};
  alt.score = -0.5;

  std::vector<NBestPath> candidates = {overridden, alt};
  auto homophones = [](size_t) -> std::vector<std::string> {
    return {kJia, kYi};
  };
  // Despite the model preferring 甲, index 0 is kept because it is overridden.
  EXPECT_EQ(rescorer.rerankBestIndex(candidates, homophones), 0u);
}

}  // namespace McBopomofo
