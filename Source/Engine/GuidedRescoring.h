// Shared settings and path selection for generative rescoring
// (ReadingGrid::walkGuided with TransformerRescorerModel), used by KeyHandler
// and the engine inspector so both run exactly the same pipeline.

#ifndef SRC_ENGINE_GUIDEDRESCORING_H_
#define SRC_ENGINE_GUIDEDRESCORING_H_

#include <cstddef>
#include <string>
#include <vector>

#include "gramambular2/reading_grid.h"

namespace McBopomofo {

// Beam 4 with up to 32 homophones per node: held-out CER 5.99% at ~16 ms per
// sentence on Apple Silicon; beam 8 is slightly more accurate at about 1.6x
// the latency.
constexpr size_t kGuidedPaths = 6;
constexpr size_t kGuidedBeam = 4;
constexpr size_t kGuidedHomophones = 32;
constexpr double kGuidedLambda = 1.0;

// Picks the first guided path that keeps every position of `walk` that carries
// explicit user intent (a manual selection / UOM override, or a user phrase)
// at the same span with the same value. Returns paths.size() if none does.
inline size_t FirstPathHonouringUserIntent(
    const Formosa::Gramambular2::ReadingGrid::WalkResult& walk,
    const std::vector<Formosa::Gramambular2::ReadingGrid::NBestPath>& paths) {
  struct Pin {
    size_t start;
    size_t length;
    std::string value;
  };
  std::vector<Pin> pins;
  size_t start = 0;
  for (const auto& node : walk.nodes) {
    if (node->isOverridden() || node->currentUnigram().isFromUserPhrase()) {
      pins.push_back({start, node->spanningLength(), node->value()});
    }
    start += node->spanningLength();
  }
  for (size_t i = 0; i < paths.size(); ++i) {
    const auto& path = paths[i];
    bool honours = true;
    for (const Pin& pin : pins) {
      bool found = false;
      size_t pos = 0;
      for (size_t j = 0; j < path.values.size() && pos <= pin.start; ++j) {
        if (pos == pin.start && path.spanningLengths[j] == pin.length &&
            path.values[j] == pin.value) {
          found = true;
          break;
        }
        pos += path.spanningLengths[j];
      }
      if (!found) {
        honours = false;
        break;
      }
    }
    if (honours) {
      return i;
    }
  }
  return paths.size();
}

}  // namespace McBopomofo

#endif  // SRC_ENGINE_GUIDEDRESCORING_H_
