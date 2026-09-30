// Context rules that the dictionary alone cannot express, applied to the
// lattice before a walk. Extracted from BopoMindFlow service/src/engine.cpp
// (Engine::ApplyRules, commit aa8ae3d) so that other front ends can share them.

#ifndef SRC_ENGINE_CONVERSIONRULES_H_
#define SRC_ENGINE_CONVERSIONRULES_H_

#include "gramambular2/reading_grid.h"

namespace McBopomofo {

struct ConversionRules {
  // Demote single-character 一 read ㄧˊ before a non-fourth tone, 一 read ㄧˋ
  // before a fourth tone, and 不 read ㄅㄨˊ before a non-fourth tone: the
  // sandhi cannot apply there, so ㄧˊ ㄓˊ becomes 移植 rather than 一直.
  // Citation tones (ㄧ, ㄅㄨˋ) and dictionary words are left alone, and so is
  // a syllable with nothing (or only a letter / punctuation) after it.
  bool sandhiFilter = true;
  // Demote candidates containing 她 or 妳 so that 他 / 你 win. They stay in
  // the candidate list.
  bool neutralPronouns = true;
};

// Score penalty for a demoted unigram. Large enough that it never wins on the
// model score alone, small enough that demoted unigrams still keep an order.
constexpr double kConversionRulePenalty = -20.0;

// Applies `rules` to every node of `grid` (a score adjustment through
// ReadingGrid::Node::adjustScores, never a removal). User-phrase unigrams and
// overridden nodes are skipped, and a unigram already at or below half the
// penalty is left as is, so calling this again after every insertReading /
// deleteReading is safe. The sandhi demotion is re-evaluated on every call and
// undone when the following syllable no longer calls for it (the grid keeps
// nodes across edits). Call it before walk / walkNBest / walkGuided.
void ApplyConversionRules(Formosa::Gramambular2::ReadingGrid& grid,
                          const ConversionRules& rules);

}  // namespace McBopomofo

#endif  // SRC_ENGINE_CONVERSIONRULES_H_
