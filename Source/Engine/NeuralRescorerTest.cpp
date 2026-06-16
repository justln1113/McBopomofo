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

#include <chrono>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "gramambular2/reading_grid.h"
#include "gtest/gtest.h"

namespace McBopomofo {

using Formosa::Gramambular2::LanguageModel;
using Formosa::Gramambular2::ReadingGrid;

namespace {

// A trivial LM reading "reading value score" lines (same format as the
// gramambular2 sample data), used to build a real grid for the rescorer tests.
class SimpleLM : public LanguageModel {
 public:
  explicit SimpleLM(const char* input) {
    std::stringstream sstream(input);
    while (sstream.good()) {
      std::string line;
      getline(sstream, line);
      if (line.empty() || line[0] == '#') {
        continue;
      }
      std::stringstream linestream(line);
      std::string reading;
      std::string value;
      std::string score;
      linestream >> reading >> value >> score;
      db_[reading].emplace_back(value, std::stod(score));
    }
  }

  std::vector<Unigram> getUnigrams(const std::string& key) override {
    auto f = db_.find(key);
    return f == db_.end() ? std::vector<Unigram>() : f->second;
  }
  bool hasUnigrams(const std::string& key) override {
    return db_.find(key) != db_.end();
  }

 private:
  std::unordered_map<std::string, std::vector<Unigram>> db_;
};

// Same homophone scenario as the gramambular2 walkNBest tests: plain walk picks
// 一你, but 依你 should win once context is considered.
constexpr char kHomophoneData[] = R"(
ㄧ 一 -2.08214539
ㄧ 醫 -3.44703033
ㄧ 依 -3.64082381
ㄧ 衣 -4.00449553
ㄋㄧˇ 你 -3.20000000
ㄋㄧˇ 妳 -5.50000000
)";

// Returns a constant log prob per step. Used to measure pure orchestration
// overhead (cache lookups, key building) without any model math.
class ConstantMockModel : public RescorerModel {
 public:
  RescorerModelState initialState() override { return {}; }
  std::pair<double, RescorerModelState> step(
      const RescorerModelState&, const std::string&,
      const std::vector<std::string>&) override {
    return {-1.0, RescorerModelState{}};
  }
};

// Encodes a tiny hand-written bigram-ish rule: it strongly rewards specific
// value sequences (依你) and penalizes others (一你). This stands in for what a
// trained model would learn, so we can verify the rerank actually flips the
// choice end-to-end.
class RuleBasedMockModel : public RescorerModel {
 public:
  RescorerModelState initialState() override { return {}; }

  std::pair<double, RescorerModelState> step(
      const RescorerModelState& prev, const std::string& value,
      const std::vector<std::string>&) override {
    // Previous value comes from the state, not a member -- this is what keeps
    // step() pure and correct under prefix caching.
    const std::string& prevValue = prev.context;

    double logProb = -5.0;  // default: unlikely
    if (value == "依") {
      logProb = -0.1;  // 依 is a great sentence start here
    } else if (value == "一") {
      logProb = -4.0;  // 一 as a start is penalized
    } else if (value == "你" && prevValue == "依") {
      logProb = -0.05;  // 依 你 is a strong bigram
    } else if (value == "你") {
      logProb = -2.0;
    }

    RescorerModelState next;
    next.context = value;  // self-contained context for the next step
    return {logProb, next};
  }
};

ReadingGrid BuildYiNiGrid() {
  ReadingGrid grid(std::make_shared<SimpleLM>(kHomophoneData));
  grid.setReadingSeparator("");
  grid.insertReading("ㄧ");
  grid.insertReading("ㄋㄧˇ");
  return grid;
}

// Builds a HomophoneProvider backed by a real grid. For an all-single-syllable
// sentence the candidate value index equals the grid reading position, so we
// can map position -> candidatesAt(position) directly. (Multi-char nodes would
// need spanning-length accumulation; not needed for these single-char cases.)
NeuralRescorer::HomophoneProvider GridHomophoneProvider(ReadingGrid* grid) {
  return [grid](size_t position) {
    std::vector<std::string> values;
    for (const auto& c : grid->candidatesAt(position)) {
      values.push_back(c.value);
    }
    return values;
  };
}

// A model that records the homophone set it receives at each step, so a test
// can assert the selective-softmax denominator is actually wired through from
// the grid. Scores like the rule model so reranking still flips to 依你.
class HomophoneRecordingModel : public RescorerModel {
 public:
  RescorerModelState initialState() override { return {}; }

  std::pair<double, RescorerModelState> step(
      const RescorerModelState& prev, const std::string& value,
      const std::vector<std::string>& homophones) override {
    receivedHomophones.push_back(homophones);

    const std::string& prevValue = prev.context;
    double logProb = -5.0;
    if (value == "依") {
      logProb = -0.1;
    } else if (value == "一") {
      logProb = -4.0;
    } else if (value == "你" && prevValue == "依") {
      logProb = -0.05;
    } else if (value == "你") {
      logProb = -2.0;
    }

    RescorerModelState next;
    next.context = value;
    return {logProb, next};
  }

  // One entry per executed step (cache hits don't call step()).
  std::vector<std::vector<std::string>> receivedHomophones;
};

}  // namespace

TEST(NeuralRescorerTest, ConstantModelPreservesUnigramOrder) {
  ReadingGrid grid = BuildYiNiGrid();
  auto candidates = grid.walkNBest(5);
  ASSERT_GE(candidates.size(), 2u);

  // A constant per-step model adds the same offset to every path (paths have
  // equal length here), so the unigram order must be preserved: 一你 stays top.
  auto model = std::make_shared<ConstantMockModel>();
  NeuralRescorer rescorer(model, /*lambda=*/1.0);
  const auto& best = rescorer.rerank(candidates);
  ASSERT_EQ(best.valuesAsStrings(),
            (std::vector<std::string>{"一", "你"}));
}

TEST(NeuralRescorerTest, RuleModelFlipsToYiNi) {
  ReadingGrid grid = BuildYiNiGrid();
  auto candidates = grid.walkNBest(5);
  ASSERT_GE(candidates.size(), 2u);

  // The plain walk's top candidate is 一你.
  ASSERT_EQ(candidates[0].valuesAsStrings(),
            (std::vector<std::string>{"一", "你"}));

  // With the rule model and a large enough lambda, the rescorer must flip to
  // 依你, demonstrating the end-to-end "walkNBest -> rerank" path works.
  auto model = std::make_shared<RuleBasedMockModel>();
  NeuralRescorer rescorer(model, /*lambda=*/5.0);
  const auto& best = rescorer.rerank(candidates);
  ASSERT_EQ(best.valuesAsStrings(),
            (std::vector<std::string>{"依", "你"}));
}

TEST(NeuralRescorerTest, LambdaZeroIsIdentity) {
  ReadingGrid grid = BuildYiNiGrid();
  auto candidates = grid.walkNBest(5);
  ASSERT_GE(candidates.size(), 2u);

  // lambda=0 ignores the model entirely; result must equal the plain walk top.
  auto model = std::make_shared<RuleBasedMockModel>();
  NeuralRescorer rescorer(model, /*lambda=*/0.0);
  const auto& best = rescorer.rerank(candidates);
  ASSERT_EQ(best.valuesAsStrings(), candidates[0].valuesAsStrings());
}

TEST(NeuralRescorerTest, PrefixCacheSharesWork) {
  ReadingGrid grid = BuildYiNiGrid();
  auto candidates = grid.walkNBest(5);
  ASSERT_GE(candidates.size(), 3u);

  // Many candidates end in "你"; once a "...你" prefix is computed for one
  // candidate, identical prefixes in others should hit the cache. Verify at
  // least some steps were served from cache.
  auto model = std::make_shared<ConstantMockModel>();
  NeuralRescorer rescorer(model, 1.0);
  rescorer.rerank(candidates);
  // Across all candidates there are repeated single-char prefixes; with several
  // candidates sharing the second position value "你" we expect cache hits.
  std::cout << "[ cache ] computed=" << rescorer.lastStepsComputed()
            << " cached=" << rescorer.lastStepsCached() << "\n";
  ASSERT_GT(rescorer.lastStepsCached(), 0u);
}

TEST(NeuralRescorerTest, EndToEndWithGridHomophoneProvider) {
  // Full connectivity: real grid -> walkNBest -> rerank, with the selective-
  // softmax homophone set supplied live from the grid (not a stub). Verifies
  // the whole pipeline links and that the homophone set reaches the model.
  ReadingGrid grid = BuildYiNiGrid();
  auto candidates = grid.walkNBest(5);
  ASSERT_GE(candidates.size(), 2u);

  auto model = std::make_shared<HomophoneRecordingModel>();
  NeuralRescorer rescorer(model, /*lambda=*/5.0);
  auto provider = GridHomophoneProvider(&grid);

  const auto& best = rescorer.rerank(candidates, provider);

  // The end-to-end choice must flip to 依你.
  ASSERT_EQ(best.valuesAsStrings(),
            (std::vector<std::string>{"依", "你"}));

  // The homophone set for position 0 must be the ㄧ family the grid knows.
  ASSERT_FALSE(model->receivedHomophones.empty());
  bool sawYiFamily = false;
  for (const auto& set : model->receivedHomophones) {
    bool hasYi = std::find(set.begin(), set.end(), "一") != set.end();
    bool hasI = std::find(set.begin(), set.end(), "依") != set.end();
    if (hasYi && hasI) {
      sawYiFamily = true;
      break;
    }
  }
  ASSERT_TRUE(sawYiFamily)
      << "selective-softmax homophone set was not wired through from the grid";
}

TEST(NeuralRescorerTest, UserOverrideIsNotReranked) {
  // If the top candidate carries explicit user intent (a manual override /
  // UOM suggestion), the rescorer must NOT flip it, even with a model that
  // strongly prefers a different value. This protects user-chosen characters
  // and user-phrase boosts from being overruled by statistics.
  ReadingGrid grid = BuildYiNiGrid();

  // Force the first syllable to 醫 as if the user had picked it. This sets the
  // node's override flag, which walkNBest propagates into NBestPath.overridden.
  ASSERT_TRUE(grid.overrideCandidate(
      0, "醫", ReadingGrid::Node::OverrideType::kOverrideValueWithHighScore));

  auto candidates = grid.walkNBest(5);
  ASSERT_FALSE(candidates.empty());
  // The overridden walk puts 醫你 on top, and the top path must be flagged.
  ASSERT_EQ(candidates[0].valuesAsStrings(),
            (std::vector<std::string>{"醫", "你"}));
  ASSERT_TRUE(candidates[0].hasUserOverride());

  // A model that loves 依 must NOT be allowed to flip the user's 醫.
  auto model = std::make_shared<RuleBasedMockModel>();
  NeuralRescorer rescorer(model, /*lambda=*/100.0);
  const auto& best = rescorer.rerank(candidates);
  ASSERT_EQ(best.valuesAsStrings(),
            (std::vector<std::string>{"醫", "你"}));
}

TEST(NeuralRescorerTest, Timing) {
  // Build a realistic 10-syllable grid and time the full walkNBest -> rerank
  // pipeline with a constant model (isolates orchestration cost from model
  // math). This is the end-to-end "take top-N + re-rank" overhead.
  constexpr char kData[] = R"(
ㄍㄠ 高 -7.171551
ㄎㄜ 科 -7.171052
ㄐㄧˋ 計 -7.926683
ㄐㄧˋ 技 -8.450826
ㄍㄨㄥ 公 -7.877973
ㄍㄨㄥ 工 -7.822167
ㄙ 思 -9.006414
ㄙ 斯 -8.091803
ㄉㄜ˙ 的 -3.516024
ㄋㄧㄢˊ 年 -6.086515
ㄓㄨㄥ 中 -5.809297
ㄓㄨㄥ 鐘 -9.877580
ㄐㄧㄤˇ 講 -9.164384
ㄐㄧㄤˇ 獎 -8.690941
ㄐㄧㄣ 金 -7.290109
ㄐㄧㄣ 今 -8.034095
)";
  ReadingGrid grid(std::make_shared<SimpleLM>(kData));
  grid.setReadingSeparator("");
  const std::vector<std::string> sentence = {
      "ㄍㄠ", "ㄎㄜ", "ㄐㄧˋ", "ㄍㄨㄥ", "ㄙ",
      "ㄉㄜ˙", "ㄋㄧㄢˊ", "ㄓㄨㄥ", "ㄐㄧㄤˇ", "ㄐㄧㄣ"};
  for (const auto& r : sentence) {
    grid.insertReading(r);
  }

  auto model = std::make_shared<ConstantMockModel>();
  NeuralRescorer rescorer(model, 1.0);

  constexpr int kIters = 2000;
  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < kIters; ++i) {
    auto candidates = grid.walkNBest(10);
    rescorer.clearCache();  // simulate a fresh sentence each iteration
    volatile size_t idx = rescorer.rerankBestIndex(candidates);
    (void)idx;
  }
  auto t1 = std::chrono::steady_clock::now();
  double us =
      std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() /
      1000.0 / kIters;
  std::cout << "[ timing ] walkNBest(10)+rerank (constant model): " << us
            << " us/call\n";
  // Generous ceiling; orchestration must stay well within a keystroke budget.
  ASSERT_LT(us, 3000.0);
}

}  // namespace McBopomofo
