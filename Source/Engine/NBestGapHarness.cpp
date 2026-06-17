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

#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "LstmRescorerModel.h"
#include "McBopomofoLM.h"
#include "NeuralRescorer.h"
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

  std::ifstream in(readingsPath);
  if (!in) {
    std::fprintf(stderr, "cannot open %s\n", readingsPath.c_str());
    return 1;
  }

  std::printf("sid,cand_idx,is_pick,is_plaintop,n_cand,unigram,model,"
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

    std::vector<ReadingGrid::NBestPath> paths = grid.walkNBest(kNBest);
    if (paths.empty()) {
      ++skipped;
      continue;
    }

    McBopomofo::NeuralRescorer rescorer(model);
    const double lambda = rescorer.lambda();
    std::vector<double> outScores;
    size_t best = rescorer.rerankBestIndex(paths, nullptr, &outScores);

    for (size_t i = 0; i < paths.size(); ++i) {
      const double combined = outScores[i];
      const double unigram = paths[i].score;
      const double mscore = lambda != 0.0 ? (combined - unigram) / lambda : 0.0;
      std::string value;
      for (const auto& v : paths[i].values) value += v;
      std::printf("%d,%zu,%d,%d,%zu,%.5f,%.5f,%.5f,%s,%s\n", sid, i,
                  i == best ? 1 : 0, i == 0 ? 1 : 0, paths.size(), unigram,
                  mscore, combined, value.c_str(), gold.c_str());
    }
    ++sid;
    ++processed;
  }
  std::fprintf(stderr, "processed %d sentences, skipped %d\n", processed,
               skipped);
  return 0;
}
