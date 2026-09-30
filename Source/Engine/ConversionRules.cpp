#include "ConversionRules.h"

#include <string>
#include <vector>

namespace McBopomofo {

namespace {

bool IsFourthTone(const std::string& reading) {
  static const std::string kFourth = "ˋ";
  return reading.size() >= kFourth.size() &&
         reading.compare(reading.size() - kFourth.size(), kFourth.size(),
                         kFourth) == 0;
}

}  // namespace

void ApplyConversionRules(Formosa::Gramambular2::ReadingGrid& grid,
                          const ConversionRules& rules) {
  using Formosa::Gramambular2::LanguageModel;
  if (!rules.sandhiFilter && !rules.neutralPronouns) return;
  static const std::string kYi = "一";
  static const std::string kBu = "不";
  static const std::string kShe = "她";
  static const std::string kNi = "妳";
  static const std::string kYi2 = "ㄧˊ";
  static const std::string kYi4 = "ㄧˋ";
  static const std::string kBu2 = "ㄅㄨˊ";
  const std::vector<std::string>& readings = grid.readings();
  const auto& spans = grid.spans();
  for (size_t i = 0; i < spans.size(); ++i) {
    // Tone sandhi only concerns single-character nodes; dictionary words
    // carry their own (already correct) readings. It needs a following
    // syllable: at the end of the buffer or before a letter / punctuation
    // reading ("_..."), there is no context to judge, so nothing is demoted.
    const bool hasNext = i + 1 < readings.size() &&
                         !readings[i + 1].empty() &&
                         readings[i + 1].front() != '_';
    const bool nextFourth = hasNext && IsFourthTone(readings[i + 1]);
    for (size_t n = 1; n <= spans[i].maxLength(); ++n) {
      const auto& node = spans[i].nodeOf(n);
      if (node == nullptr || node->isOverridden()) continue;
      const std::string& r = node->reading();
      // The grid is updated in place and keeps nodes across edits, so the
      // sandhi decision is re-made on every call: a node whose neighbour
      // changed gets its earlier demotion undone.
      const bool sandhiNode = rules.sandhiFilter && n == 1 &&
                              (r == kYi2 || r == kYi4 || r == kBu2);
      bool demoteYi = false;
      bool demoteBu = false;
      if (sandhiNode) {
        demoteYi = hasNext && ((r == kYi2 && !nextFourth) ||
                               (r == kYi4 && nextFourth));
        demoteBu = hasNext && r == kBu2 && !nextFourth;
      }
      const bool pronouns =
          rules.neutralPronouns && !r.empty() && r.front() != '_';
      if (!sandhiNode && !pronouns) continue;
      node->adjustScores([&](const LanguageModel::Unigram& u) {
        if (u.isFromUserPhrase()) return 0.0;
        const std::string& v = u.value();
        const bool demoted = u.score() <= kConversionRulePenalty / 2;
        if (sandhiNode && (v == kYi || v == kBu)) {
          const bool want = v == kYi ? demoteYi : demoteBu;
          if (want && !demoted) return kConversionRulePenalty;
          if (!want && demoted) return -kConversionRulePenalty;
          return 0.0;
        }
        if (demoted) return 0.0;  // done
        double d = 0;
        if (pronouns && (v.find(kShe) != std::string::npos ||
                         v.find(kNi) != std::string::npos)) {
          d += kConversionRulePenalty;
        }
        return d;
      });
    }
  }
}

}  // namespace McBopomofo
