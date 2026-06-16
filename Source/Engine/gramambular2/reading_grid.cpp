// Copyright (c) 2022 and onwards Lukhnos Liu.
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

#include "reading_grid.h"

#include <algorithm>
#include <chrono>
#include <limits>
#include <memory>
#include <stack>
#include <string>
#include <utility>
#include <vector>

namespace Formosa::Gramambular2 {

void ReadingGrid::clear() {
  cursor_ = 0;
  readings_.clear();
  spans_.clear();
}

void ReadingGrid::setCursor(size_t cursor) {
  assert(cursor <= readings_.size());
  cursor_ = cursor;
}

void ReadingGrid::setReadingSeparator(const std::string& separator) {
  separator_ = separator;
}

bool ReadingGrid::insertReading(const std::string& reading) {
  if (reading.empty() || reading == separator_) {
    return false;
  }

  if (!lm_.hasUnigrams(reading)) {
    return false;
  }

  readings_.insert(readings_.begin() + static_cast<ptrdiff_t>(cursor_),
                   reading);
  expandGridAt(cursor_);
  update();

  // Cursor must only move after update().
  ++cursor_;
  return true;
}

bool ReadingGrid::deleteReadingBeforeCursor() {
  if (!cursor_) {
    return false;
  }

  readings_.erase(readings_.begin() + static_cast<ptrdiff_t>(cursor_ - 1),
                  readings_.begin() + static_cast<ptrdiff_t>(cursor_));
  // Cursor must decrement for grid-shrinking and update to work.
  --cursor_;
  shrinkGridAt(cursor_);
  update();
  return true;
}

bool ReadingGrid::deleteReadingAfterCursor() {
  if (cursor_ == readings_.size()) {
    return false;
  }

  readings_.erase(readings_.begin() + static_cast<ptrdiff_t>(cursor_),
                  readings_.begin() + static_cast<ptrdiff_t>(cursor_ + 1));
  shrinkGridAt(cursor_);
  update();
  return true;
}

std::optional<ReadingGrid::NodePtr> ReadingGrid::findInSpan(
    size_t cursor, const std::function<bool(const NodePtr&)>& predicate) const {
  assert(cursor <= readings_.size());
  std::vector<ReadingGrid::NodeInSpan> nodes =
      overlappingNodesAt(cursor == readings_.size() ? cursor - 1 : cursor);

  auto nodesIt = std::find_if(
      nodes.cbegin(), nodes.cend(),
      [&](const NodeInSpan& nodeInSpan) { return predicate(nodeInSpan.node); });

  return nodesIt == nodes.end()
             ? std::nullopt
             : std::optional<ReadingGrid::NodePtr>(nodesIt->node);
}

namespace {

int64_t GetEpochNowInMicroseconds() {
  auto now = std::chrono::system_clock::now();
  int64_t timestamp =
      std::chrono::time_point_cast<std::chrono::microseconds>(now)
          .time_since_epoch()
          .count();
  return timestamp;
}

}  // namespace

// Find the weightiest path in the grid graph. The path represents the most
// likely hidden chain of events from the observations.
// We use the Viterbi algorithm to compute such path.
// Instead of computing the path with the shortest distance, though, we compute
// the path with the longest distance (so the weightiest), since with log
// probability a larger value means a larger probability. The algorithm runs in
// O(|V| + |E|) time for G = (V, E) where G is a DAG. This means the walk is
// fairly economical even when the grid is large.
ReadingGrid::WalkResult ReadingGrid::walk() {
  WalkResult result;
  if (spans_.empty()) {
    return result;
  }
  int64_t start = GetEpochNowInMicroseconds();

  // Defines a state in the DP table. This structure tracks the maximum
  // accumulated score and the back-pointer required for path reconstruction in
  // the Viterbi algorithm.
  struct State {
    size_t fromIndex = 0;
    ReadingGrid::NodePtr fromNode = nullptr;
    double maxScore = -std::numeric_limits<double>::infinity();
  };

  const size_t readingLen = readings_.size();
  std::vector<State> viterbi(readingLen + 1);
  viterbi[0].maxScore = 0.0;

  // Iterate through the grid and compute the maximum accumulated score for each
  // reachable position. Since the grid is a lattice where edges only point
  // forward, processing nodes in index order is equivalent to processing them
  // in topological order.
  size_t reachableStates = 0;
  size_t evaluatedEdges = 0;
  for (size_t i = 0; i < readingLen; ++i) {
    ++reachableStates;

    const ReadingGrid::Span& span = spans_[i];
    const size_t maxSpanLen = span.maxLength();

    for (size_t spanLen = 1; spanLen <= maxSpanLen; ++spanLen) {
      const ReadingGrid::NodePtr& node = span.nodeOf(spanLen);
      if (node == nullptr) {
        continue;
      }
      ++evaluatedEdges;

      // Performs a relaxation on a transition. This updates the destination
      // state if the path through the current node yields a higher score than
      // the previously known best path. This is the core operation of the
      // Viterbi algorithm, adapted for finding the maximum likelihood path.
      double score = viterbi[i].maxScore + node->score();
      State& target = viterbi[i + spanLen];
      if (score > target.maxScore) {
        target.maxScore = score;
        target.fromNode = node;
        target.fromIndex = i;
      }
    }
  }
  // Vertices are the reachable states
  // Edges are the candidate word transitions
  result.vertices = reachableStates;
  result.edges = evaluatedEdges;

  // Reconstruct the most likely path by tracing back from the end of the grid
  // to the root using the back-pointers
  size_t totalReadingLen = 0;
  for (size_t curr = readingLen; curr > 0; curr = viterbi[curr].fromIndex) {
    assert(viterbi[curr].fromNode != nullptr);
    totalReadingLen += viterbi[curr].fromNode->spanningLength();
    result.nodes.emplace_back(std::move(viterbi[curr].fromNode));
  }
  std::reverse(result.nodes.begin(), result.nodes.end());
  assert(totalReadingLen == readingLen);
  result.totalReadings = totalReadingLen;

  result.elapsedMicroseconds = GetEpochNowInMicroseconds() - start;
  return result;
}

std::vector<ReadingGrid::NBestPath> ReadingGrid::walkNBest(size_t k) {
  std::vector<NBestPath> results;
  if (spans_.empty() || k == 0) {
    return results;
  }

  // A partial path ending at some grid position. Crucially each entry commits to
  // a specific unigram of a node (not just the node), so that homophones of the
  // *same* node (一 vs 依 for reading ㄧ) become distinct competing paths -- the
  // whole reason this exists.
  struct BeamEntry {
    double score = -std::numeric_limits<double>::infinity();
    size_t fromIndex = 0;
    // Index into viterbi[fromIndex] identifying which predecessor partial path
    // this entry extends, enabling back-tracking of each distinct path.
    size_t fromBeam = 0;
    NodePtr node = nullptr;       // nullptr only for the seed entry at cell 0
    std::string value;           // the chosen unigram value at this node
    bool fromUserPhrase = false;  // chosen unigram came from a user phrase
  };

  // Beam width. We keep a few more partial paths per cell than k, because
  // distinct full paths can share suffixes and collapse during de-duplication;
  // a slightly wider beam keeps enough diversity to still return k of them.
  const size_t beamWidth = std::max<size_t>(k + 2, 4);
  // Per node, how many top homophones to branch on. Bounded so a node with a
  // huge homophone list (e.g. ㄧ) doesn't blow up the beam; the unigrams are
  // already score-ranked, so the first few are the only plausible alternatives.
  const size_t maxUnigramsPerNode = beamWidth;

  const size_t readingLen = readings_.size();
  std::vector<std::vector<BeamEntry>> viterbi(readingLen + 1);
  viterbi[0].push_back(BeamEntry{0.0, 0, 0, nullptr, ""});

  // Prune a cell to the beam width. Must be applied to a cell *before* it is
  // used as a source for forward expansion, otherwise the beam multiplies cell
  // by cell and blows up exponentially with sentence length.
  auto pruneCell = [&](std::vector<BeamEntry>& cell) {
    std::stable_sort(cell.begin(), cell.end(),
                     [](const BeamEntry& a, const BeamEntry& b) {
                       return a.score > b.score;
                     });
    if (cell.size() > beamWidth) {
      cell.resize(beamWidth);
    }
  };

  for (size_t i = 0; i < readingLen; ++i) {
    // Cell i has all its incoming transitions by now (edges only point
    // forward and we visit positions in order), so prune it before it feeds
    // the next cells. This bounds every source cell to beamWidth.
    pruneCell(viterbi[i]);
    if (viterbi[i].empty()) {
      continue;
    }

    const Span& span = spans_[i];
    const size_t maxSpanLen = span.maxLength();

    for (size_t spanLen = 1; spanLen <= maxSpanLen; ++spanLen) {
      const NodePtr& node = span.nodeOf(spanLen);
      if (node == nullptr) {
        continue;
      }

      std::vector<BeamEntry>& target = viterbi[i + spanLen];
      if (node->isOverridden()) {
        // The user (or UOM) pinned this node to a specific value. Honor it: do
        // not branch on homophones here, and use the node's override-adjusted
        // score (node->score()) so the override actually wins the walk -- the
        // raw unigram scores would otherwise ignore the override entirely.
        const bool overriddenFromUser = node->currentUnigram().isFromUserPhrase();
        for (size_t b = 0; b < viterbi[i].size(); ++b) {
          target.push_back(BeamEntry{viterbi[i][b].score + node->score(), i, b,
                                     node, node->value(), overriddenFromUser});
        }
        continue;
      }

      const auto& unigrams = node->unigrams();
      const size_t uCount = std::min(unigrams.size(), maxUnigramsPerNode);
      // For every partial path reaching i, branch on each top homophone.
      for (size_t b = 0; b < viterbi[i].size(); ++b) {
        const double base = viterbi[i][b].score;
        for (size_t u = 0; u < uCount; ++u) {
          target.push_back(BeamEntry{base + unigrams[u].score(), i, b, node,
                                     unigrams[u].value(),
                                     unigrams[u].isFromUserPhrase()});
        }
      }
    }
  }

  // The terminal cell is never a source, so prune it here.
  pruneCell(viterbi[readingLen]);

  // Reconstruct one full path per surviving terminal beam entry.
  std::vector<std::string> seenKeys;
  for (size_t t = 0; t < viterbi[readingLen].size() && results.size() < k;
       ++t) {
    NBestPath path;
    size_t curr = readingLen;
    size_t beam = t;
    while (curr > 0) {
      const BeamEntry& entry = viterbi[curr][beam];
      if (entry.node == nullptr) {
        break;
      }
      path.values.push_back(entry.value);
      path.readings.push_back(entry.node->reading());
      path.spanningLengths.push_back(entry.node->spanningLength());
      path.overridden.push_back(entry.node->isOverridden());
      path.fromUserPhrase.push_back(entry.fromUserPhrase);
      curr = entry.fromIndex;
      beam = entry.fromBeam;
    }
    std::reverse(path.values.begin(), path.values.end());
    std::reverse(path.readings.begin(), path.readings.end());
    std::reverse(path.spanningLengths.begin(), path.spanningLengths.end());
    std::reverse(path.overridden.begin(), path.overridden.end());
    std::reverse(path.fromUserPhrase.begin(), path.fromUserPhrase.end());
    // Total score is the terminal entry's accumulated score.
    path.score = viterbi[readingLen][t].score;

    // De-duplicate: different beam entries can backtrack to the same sequence
    // of values once suffixes merge.
    std::string key;
    for (const std::string& v : path.values) {
      key += v;
      key += '\x1f';
    }
    if (std::find(seenKeys.begin(), seenKeys.end(), key) != seenKeys.end()) {
      continue;
    }
    seenKeys.push_back(std::move(key));
    results.push_back(std::move(path));
  }

  return results;
}

ReadingGrid::WalkResult ReadingGrid::walkResultFromPath(
    const NBestPath& path) const {
  WalkResult result;
  size_t total = 0;
  for (size_t i = 0; i < path.values.size(); ++i) {
    size_t span = i < path.spanningLengths.size() ? path.spanningLengths[i] : 1;
    const std::string reading =
        i < path.readings.size() ? path.readings[i] : std::string();
    // A single unigram carrying the chosen value, so node->value() returns it
    // and the node reports isOverridden() == false (it is a plain synthesized
    // node, not a user/UOM override of a real grid node).
    std::vector<LanguageModel::Unigram> unigrams;
    unigrams.emplace_back(path.values[i], 0.0, path.values[i]);
    result.nodes.push_back(
        std::make_shared<Node>(reading, span, std::move(unigrams)));
    total += span;
  }
  result.totalReadings = total;
  return result;
}

std::vector<ReadingGrid::Candidate> ReadingGrid::candidatesAt(size_t loc) {
  std::vector<ReadingGrid::Candidate> result;
  if (readings_.empty()) {
    return result;
  }

  if (loc > readings_.size()) {
    return result;
  }

  std::vector<NodeInSpan> nodes =
      overlappingNodesAt(loc == readings_.size() ? loc - 1 : loc);

  // Sort nodes by reading length.
  std::stable_sort(
      nodes.begin(), nodes.end(), [](const auto& n1, const auto& n2) {
        return n1.node->spanningLength() > n2.node->spanningLength();
      });

  for (const NodeInSpan& nodeInSpan : nodes) {
    for (const LanguageModel::Unigram& unigram : nodeInSpan.node->unigrams()) {
      result.emplace_back(nodeInSpan.node->reading(), unigram.value(),
                          unigram.rawValue());
    }
  }
  return result;
}

bool ReadingGrid::overrideCandidate(
    size_t loc, const ReadingGrid::Candidate& candidate,
    ReadingGrid::Node::OverrideType overrideType) {
  return overrideCandidate(loc, &candidate.reading, candidate.value,
                           overrideType);
}

bool ReadingGrid::overrideCandidate(
    size_t loc, const std::string& candidate,
    ReadingGrid::Node::OverrideType overrideType) {
  return overrideCandidate(loc, nullptr, candidate, overrideType);
}

void ReadingGrid::expandGridAt(size_t loc) {
  if (!loc || loc == spans_.size()) {
    spans_.insert(spans_.begin() + static_cast<ptrdiff_t>(loc), Span());
    return;
  }
  spans_.insert(spans_.begin() + static_cast<ptrdiff_t>(loc), Span());
  removeAffectedNodes(loc);
}

void ReadingGrid::shrinkGridAt(size_t loc) {
  if (loc == spans_.size()) {
    return;
  }
  spans_.erase(spans_.begin() + static_cast<ptrdiff_t>(loc));
  removeAffectedNodes(loc);
}

void ReadingGrid::removeAffectedNodes(size_t loc) {
  // Because of the expansion, certain spans now have "broken" nodes. We need
  // to remove those. For example, before:
  //
  // Span index 0   1   2   3
  //                (---)
  //                (-------)
  //            (-----------)
  //
  // After we've inserted a span at 2:
  //
  // Span index 0   1   2   3   4
  //                (---)
  //                (----   ----)
  //            (--------   ----)
  //
  // Similarly for shrinkage, before:
  //
  // Span index 0   1   2   3
  //                (---)
  //                (-------)
  //            (-----------)
  //
  // After we've deleted the span at 2:
  //
  // Span index 0   1   2   3   4
  //                (---)
  //                XXXXX
  //            XXXXXXXXX
  //
  if (spans_.empty()) {
    return;
  }
  size_t affectedLength = kMaximumSpanLength - 1;
  size_t begin = loc <= affectedLength ? 0 : loc - affectedLength;
  size_t end = loc >= 1 ? loc - 1 : 0;
  for (size_t i = begin; i <= end; ++i) {
    spans_[i].removeNodesOfOrLongerThan(loc - i + 1);
  }
}

void ReadingGrid::insert(size_t loc, const ReadingGrid::NodePtr& node) {
  assert(loc < spans_.size());
  spans_[loc].add(node);
}

std::string ReadingGrid::combineReading(
    std::vector<std::string>::const_iterator begin,
    std::vector<std::string>::const_iterator end) {
  std::string result;
  for (auto iter = begin; iter != end;) {
    result += *iter;
    ++iter;
    if (iter != end) {
      result += separator_;
    }
  }
  return result;
}

bool ReadingGrid::hasNodeAt(size_t loc, size_t readingLen,
                            const std::string& reading) {
  if (loc > spans_.size()) {
    return false;
  }
  const NodePtr& n = spans_[loc].nodeOf(readingLen);
  if (n == nullptr) {
    return false;
  }
  return reading == n->reading();
}

void ReadingGrid::update() {
  size_t begin =
      (cursor_ <= kMaximumSpanLength) ? 0 : cursor_ - kMaximumSpanLength;
  size_t end = cursor_ + kMaximumSpanLength;
  end = std::min(end, readings_.size());

  for (size_t pos = begin; pos < end; pos++) {
    for (size_t len = 1; len <= kMaximumSpanLength && pos + len <= end; len++) {
      std::string combinedReading =
          combineReading(readings_.begin() + static_cast<ptrdiff_t>(pos),
                         readings_.begin() + static_cast<ptrdiff_t>(pos + len));

      if (!hasNodeAt(pos, len, combinedReading)) {
        auto unigrams = lm_.getUnigrams(combinedReading);
        if (unigrams.empty()) {
          continue;
        }

        insert(pos, std::make_shared<Node>(combinedReading, len, unigrams));
      }
    }
  }
}

bool ReadingGrid::overrideCandidate(
    size_t loc, const std::string* reading, const std::string& value,
    ReadingGrid::Node::OverrideType overrideType) {
  if (loc > readings_.size()) {
    return false;
  }

  std::vector<NodeInSpan> overlappingNodes =
      overlappingNodesAt(loc == readings_.size() ? loc - 1 : loc);
  NodeInSpan overridden;
  for (NodeInSpan& nis : overlappingNodes) {
    if (reading != nullptr && nis.node->reading() != *reading) {
      continue;
    }

    if (nis.node->selectOverrideUnigram(value, overrideType)) {
      overridden = nis;
      break;
    }
  }

  if (overridden.node == nullptr) {
    // Nothing gets overridden.
    return false;
  }

  for (size_t i = overridden.spanIndex;
       i < overridden.spanIndex + overridden.node->spanningLength() &&
       i < spans_.size();
       ++i) {
    // We also need to reset *all* nodes that share the same location in the
    // span. For example, if previously the two walked nodes are "A BC" where
    // A and BC are two nodes with overrides. The user now chooses "DEF" which
    // is a node that shares the same span location with "A". The node with BC
    // will be reset as it's part of the overlapping node, but A is not.
    std::vector<NodeInSpan> nodes = overlappingNodesAt(i);
    for (NodeInSpan& nis : nodes) {
      if (nis.node != overridden.node) {
        nis.node->reset();
      }
    }
  }
  return true;
}

std::vector<ReadingGrid::NodeInSpan> ReadingGrid::overlappingNodesAt(
    size_t loc) const {
  std::vector<ReadingGrid::NodeInSpan> results;

  if (spans_.empty() || loc >= spans_.size()) {
    return results;
  }

  // First, get all nodes from the span at location.
  for (size_t i = 1, len = spans_[loc].maxLength(); i <= len; ++i) {
    NodePtr ptr = spans_[loc].nodeOf(i);
    if (ptr != nullptr) {
      ReadingGrid::NodeInSpan element{std::move(ptr), loc};
      results.emplace_back(std::move(element));
    }
  }

  size_t begin = loc - std::min(loc, kMaximumSpanLength - 1);
  for (size_t i = begin; i < loc; ++i) {
    size_t beginLen = loc - i + 1;
    size_t endLen = spans_[i].maxLength();
    for (size_t j = beginLen; j <= endLen; ++j) {
      NodePtr ptr = spans_[i].nodeOf(j);
      if (ptr != nullptr) {
        ReadingGrid::NodeInSpan element{std::move(ptr), i};
        results.emplace_back(std::move(element));
      }
    }
  }

  return results;
}

LanguageModel::Unigram ReadingGrid::Node::currentUnigram() const {
  return unigrams_.empty() ? LanguageModel::Unigram{} : *unigramIter_;
}

std::string ReadingGrid::Node::value() const {
  return unigrams_.empty() ? "" : unigramIter_->value();
}

double ReadingGrid::Node::score() const {
  if (unigrams_.empty()) {
    return 0;
  }

  switch (overrideType_) {
    case OverrideType::kOverrideValueWithHighScore:
      return kOverridingScore;
    case OverrideType::kOverrideValueWithScoreFromTopUnigram:
      return unigrams_[0].score();
    case OverrideType::kNone:
    default:
      return unigramIter_->score();
  }
}

bool ReadingGrid::Node::isOverridden() const {
  return overrideType_ != OverrideType::kNone;
}

void ReadingGrid::Node::reset() {
  unigramIter_ = unigrams_.begin();
  overrideType_ = OverrideType::kNone;
}

bool ReadingGrid::Node::selectOverrideUnigram(
    const std::string& value, ReadingGrid::Node::OverrideType type) {
  assert(type != ReadingGrid::Node::OverrideType::kNone);
  for (auto it = unigrams_.begin(), end = unigrams_.end(); it != end; ++it) {
    if (value == it->value()) {
      unigramIter_ = it;
      overrideType_ = type;
      return true;
    }
  }
  return false;
}

std::vector<ReadingGrid::NodePtr>::const_iterator
ReadingGrid::WalkResult::findNodeAt(size_t cursor,
                                    size_t* outCursorPastNode) const {
  if (nodes.empty()) {
    return nodes.cend();
  }

  if (cursor > totalReadings) {
    return nodes.cend();
  }

  if (cursor == 0) {
    auto it = nodes.cbegin();
    if (outCursorPastNode != nullptr) {
      *outCursorPastNode = (*it)->spanningLength();
    }
    return it;
  }

  // Covers both the "cursor is right at end" and "cursor is one reading before
  // the end" cases.
  if (cursor >= totalReadings - 1) {
    if (outCursorPastNode != nullptr) {
      *outCursorPastNode = totalReadings;
    }
    return std::next(nodes.cbegin(), static_cast<ptrdiff_t>(nodes.size() - 1));
  }

  size_t accumulated = 0;
  for (auto i = nodes.cbegin(); i != nodes.cend(); ++i) {
    accumulated += (*i)->spanningLength();
    if (accumulated > cursor) {
      if (outCursorPastNode != nullptr) {
        *outCursorPastNode = accumulated;
      }
      return i;
    }
  }

  // Shouldn't happen.
  return nodes.cend();
}

std::vector<std::string> ReadingGrid::WalkResult::valuesAsStrings() const {
  std::vector<std::string> result;
  for (const NodePtr& node : nodes) {
    result.emplace_back(node->value());
  }
  return result;
}

std::vector<std::string> ReadingGrid::WalkResult::readingsAsStrings() const {
  std::vector<std::string> result;
  for (const NodePtr& node : nodes) {
    result.emplace_back(node->reading());
  }
  return result;
}

void ReadingGrid::Span::clear() {
  nodes_.fill(nullptr);
  maxLength_ = 0;
}

void ReadingGrid::Span::add(const ReadingGrid::NodePtr& node) {
  assert(node->spanningLength() > 0 &&
         node->spanningLength() <= kMaximumSpanLength);
  nodes_[node->spanningLength() - 1] = node;
  maxLength_ = std::max(maxLength_, node->spanningLength());
}

void ReadingGrid::Span::removeNodesOfOrLongerThan(size_t length) {
  assert(length > 0 && length <= kMaximumSpanLength);
  for (size_t i = length - 1; i < kMaximumSpanLength; ++i) {
    nodes_[i] = nullptr;
  }
  maxLength_ = 0;
  if (length == 1) {
    return;
  }

  size_t i = length - 2;
  while (true) {
    if (nodes_[i] != nullptr) {
      maxLength_ = i + 1;
      return;
    }

    if (i == 0) {
      return;
    }

    --i;
  }
}

const ReadingGrid::NodePtr& ReadingGrid::Span::nodeOf(size_t length) const {
  assert(length > 0 && length <= kMaximumSpanLength);
  return nodes_[length - 1];
}

std::vector<LanguageModel::Unigram>
ReadingGrid::ScoreRankedLanguageModel::getUnigrams(const std::string& reading) {
  auto unigrams = lm_->getUnigrams(reading);
  std::stable_sort(
      unigrams.begin(), unigrams.end(),
      [](const auto& u1, const auto& u2) { return u1.score() > u2.score(); });
  return unigrams;
}

bool ReadingGrid::ScoreRankedLanguageModel::hasUnigrams(
    const std::string& reading) {
  return lm_->hasUnigrams(reading);
}

}  // namespace Formosa::Gramambular2
