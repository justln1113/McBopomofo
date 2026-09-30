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
    // carry their own (already correct) readings.
    const bool nextFourth =
        i + 1 < readings.size() && IsFourthTone(readings[i + 1]);
    for (size_t n = 1; n <= spans[i].maxLength(); ++n) {
      const auto& node = spans[i].nodeOf(n);
      if (node == nullptr || node->isOverridden()) continue;
      const std::string& r = node->reading();
      bool demoteYi = false;
      bool demoteBu = false;
      if (rules.sandhiFilter && n == 1) {
        demoteYi = (r == kYi2 && !nextFourth) || (r == kYi4 && nextFourth);
        demoteBu = (r == kBu2 && !nextFourth);
      }
      const bool pronouns =
          rules.neutralPronouns && !r.empty() && r.front() != '_';
      if (!demoteYi && !demoteBu && !pronouns) continue;
      node->adjustScores([&](const LanguageModel::Unigram& u) {
        if (u.isFromUserPhrase()) return 0.0;
        if (u.score() <= kConversionRulePenalty / 2) return 0.0;  // done
        const std::string& v = u.value();
        double d = 0;
        if ((demoteYi && v == kYi) || (demoteBu && v == kBu)) {
          d += kConversionRulePenalty;
        }
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
