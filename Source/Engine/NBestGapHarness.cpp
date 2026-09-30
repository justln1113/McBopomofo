// Copyright (c) 2025 and onwards The McBopomofo Authors.
//
// Standalone measurement harness (NOT shipped). Drives the REAL production
// language model + ReadingGrid walk/walkNBest + NeuralRescorer over a list of
// bopomofo reading sequences, and dumps per-candidate scores as CSV so we can
// measure, on genuine n-best paths, the combined-score gap a personalization
// term would have to clear. Validates the synthetic single-swap headroom
// measurement (measure_headroom.py) against the live engine.
//
// Input file (TSV, one sentence per line):  <gold Han>\t<reading1> <reading2> ...
// Output (CSV to stdout):
//   sid,cand_idx,is_pick,is_plaintop,n_cand,unigram,model,combined,value,gold
//
// Build (from repo root):
//   cmake -B Source/Engine/build -DENABLE_TEST=ON -DCMAKE_BUILD_TYPE=Release
//   cmake --build Source/Engine/build --target NBestGapHarness
//   ./Source/Engine/build/NBestGapHarness \
//       Source/Data/data.txt \
//       Source/Data/rescorer-training/harness_readings.tsv \
//       Source/Data/rescorer-weights.bin \
//       Source/Data/rescorer-vocab.txt > gaps.csv

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "LstmRescorerModel.h"
#include "McBopomofoLM.h"
#include "NeuralRescorer.h"
#include "TransformerRescorerModel.h"
#include "UTF8Helper.h"
#include "gramambular2/reading_grid.h"

using Formosa::Gramambular2::LanguageModel;
using Formosa::Gramambular2::ReadingGrid;

namespace {

std::vector<std::string> SplitSpace(const std::string& s) {
  std::vector<std::string> out;
  std::istringstream iss(s);
  std::string tok;
  while (iss >> tok) {
    out.push_back(tok);
  }
  return out;
}

ReadingGrid::GuideHooks EagerHooks(
    std::shared_ptr<McBopomofo::LstmRescorerModel> model, double lambda) {
  ReadingGrid::GuideHooks hooks;
  hooks.lambda = lambda;
  hooks.initialState = [model]() -> std::shared_ptr<void> {
    return std::make_shared<McBopomofo::RescorerModelState>(
        model->initialState());
  };
  hooks.extend = [model](const std::shared_ptr<void>& prev,
                         const std::string& value,
                         std::shared_ptr<void>* next) -> double {
    auto st = std::static_pointer_cast<McBopomofo::RescorerModelState>(prev);
    auto res = model->step(*st, value, {});
    *next = std::make_shared<McBopomofo::RescorerModelState>(
        std::move(res.second));
    return res.first;
  };
  hooks.covers = [model](const std::string& v) { return model->coversValue(v); };
  return hooks;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 5) {
    std::fprintf(stderr,
                 "usage: %s <data.txt> <readings.tsv> <weights.bin> "
                 "<vocab.txt> [nbest=5]\n",
                 argv[0]);
    return 1;
  }
  const std::string dataPath = argv[1];
  const std::string readingsPath = argv[2];
  const std::string weightsPath = argv[3];
  const std::string vocabPath = argv[4];
  const size_t kNBest = argc > 5 ? std::stoul(argv[5]) : 5;

  // Generative mode, enabled by env vars so eval_scorecard.py needs no changes:
  //   GUIDED_BEAM=<width per cell>  GUIDED_HOMO=<unigrams per node>  LAMBDA=<w>
  const char* envBeam = std::getenv("GUIDED_BEAM");
  const char* envHomo = std::getenv("GUIDED_HOMO");
  const char* envLambda = std::getenv("LAMBDA");
  const size_t guidedBeam = envBeam ? std::stoul(envBeam) : 0;
  const size_t guidedHomophones = envHomo ? std::stoul(envHomo) : 8;
  const double guidedLambda = envLambda ? std::stod(envLambda) : 1.0;

  auto lm = std::make_shared<McBopomofo::McBopomofoLM>();
  lm->loadLanguageModel(dataPath.c_str());
  if (!lm->isDataModelLoaded()) {
    std::fprintf(stderr, "failed to load language model from %s\n",
                 dataPath.c_str());
    return 1;
  }

  std::shared_ptr<McBopomofo::LstmRescorerModel> model;
  try {
    model = McBopomofo::LstmRescorerModel::Load(weightsPath, vocabPath);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "failed to load rescorer model: %s\n", e.what());
    return 1;
  }

  std::shared_ptr<McBopomofo::TransformerRescorerModel> tfModel;
  if (const char* tfw = std::getenv("TF_WEIGHTS")) {
    try {
      tfModel = McBopomofo::TransformerRescorerModel::Load(tfw, vocabPath);
    } catch (const std::exception& e) {
      std::fprintf(stderr, "failed to load transformer: %s\n", e.what());
      return 1;
    }
  }
  if (tfModel && std::getenv("TF_VERIFY")) {
    // Print sum log P per sentence (gold text, one char per value) and exit.
    std::ifstream vin(readingsPath);
    std::string vl;
    while (std::getline(vin, vl)) {
      auto tab = vl.find('\t');
      std::string gold = tab == std::string::npos ? vl : vl.substr(0, tab);
      std::vector<std::string> chars;
      for (const auto& c : McBopomofo::Split(gold)) chars.push_back(c);
      std::printf("%.6f\t%s\n", tfModel->scoreValues(chars), gold.c_str());
    }
    return 0;
  }
  std::ifstream in(readingsPath);
  if (!in) {
    std::fprintf(stderr, "cannot open %s\n", readingsPath.c_str());
    return 1;
  }

  std::printf("sid,cand_idx,is_pick,is_plaintop,n_cand,usec,unigram,model,"
              "combined,value,gold\n");

  std::string line;
  int sid = 0;
  int processed = 0, skipped = 0;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    auto tab = line.find('\t');
    std::string gold = tab == std::string::npos ? "" : line.substr(0, tab);
    std::string rd = tab == std::string::npos ? line : line.substr(tab + 1);
    std::vector<std::string> readings = SplitSpace(rd);
    if (readings.empty()) {
      ++skipped;
      continue;
    }

    std::shared_ptr<LanguageModel> lmBase = lm;
    ReadingGrid grid(lmBase);
    grid.setReadingSeparator("-");
    bool ok = true;
    for (const auto& r : readings) {
      if (!grid.insertReading(r)) {
        ok = false;
        break;
      }
    }
    if (!ok) {
      ++skipped;
      continue;
    }

    if (guidedBeam > 0) {
      // Generative mode: candidate 0 is the plain unigram walk (so plain CER is
      // still measured); the rest come from lattice search guided by the LSTM.
      auto g0 = std::chrono::steady_clock::now();
      std::vector<ReadingGrid::NBestPath> plainTop = grid.walkNBest(1);
      if (plainTop.empty()) {
        ++skipped;
        continue;
      }
      ReadingGrid::GuideHooks hooks = tfModel ? tfModel->MakeGuideHooks(guidedLambda) : getenv("GUIDED_EAGER")
          ? EagerHooks(model, guidedLambda)
          : McBopomofo::LstmRescorerModel::MakeGuideHooks(model, guidedLambda);
      std::vector<ReadingGrid::NBestPath> gp =
          grid.walkGuided(kNBest, guidedBeam, guidedHomophones, hooks);
      auto g1 = std::chrono::steady_clock::now();
      const auto gusec =
          std::chrono::duration_cast<std::chrono::microseconds>(g1 - g0)
              .count();
      std::vector<ReadingGrid::NBestPath> all;
      all.push_back(plainTop[0]);
      for (auto& p : gp) all.push_back(std::move(p));
      for (size_t i = 0; i < all.size(); ++i) {
        std::string value;
        for (const auto& v : all[i].values) value += v;
        const double unigram = all[i].score;
        const double combined = i == 0 ? unigram : all[i].guidedScore;
        const double mscore =
            i == 0 || guidedLambda == 0.0 ? 0.0
                                          : (combined - unigram) / guidedLambda;
        // pick = best guided path (index 1); fall back to plain if none.
        const bool pick = gp.empty() ? i == 0 : i == 1;
        std::printf("%d,%zu,%d,%d,%zu,%lld,%.5f,%.5f,%.5f,%s,%s\n", sid, i,
                    pick ? 1 : 0, i == 0 ? 1 : 0, all.size(), (long long)gusec,
                    unigram, mscore, combined, value.c_str(), gold.c_str());
      }
      ++sid;
      ++processed;
      continue;
    }

    // Time the rescorer-relevant conversion cost: n-best generation + rerank.
    auto t0 = std::chrono::steady_clock::now();
    std::vector<ReadingGrid::NBestPath> paths = grid.walkNBest(kNBest);
    if (paths.empty()) {
      ++skipped;
      continue;
    }

    McBopomofo::NeuralRescorer rescorer(model);
    const double lambda = rescorer.lambda();
    std::vector<double> outScores;
    size_t best = rescorer.rerankBestIndex(paths, nullptr, &outScores);
    auto t1 = std::chrono::steady_clock::now();
    const auto usec =
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

    for (size_t i = 0; i < paths.size(); ++i) {
      const double combined = outScores[i];
      const double unigram = paths[i].score;
      const double mscore = lambda != 0.0 ? (combined - unigram) / lambda : 0.0;
      std::string value;
      for (const auto& v : paths[i].values) value += v;
      std::printf("%d,%zu,%d,%d,%zu,%lld,%.5f,%.5f,%.5f,%s,%s\n", sid, i,
                  i == best ? 1 : 0, i == 0 ? 1 : 0, paths.size(),
                  (long long)usec, unigram, mscore, combined, value.c_str(),
                  gold.c_str());
    }
    ++sid;
    ++processed;
  }
  std::fprintf(stderr, "processed %d sentences, skipped %d\n", processed,
               skipped);
  return 0;
}
