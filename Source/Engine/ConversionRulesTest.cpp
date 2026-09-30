#include "ConversionRules.h"

#include <gtest/gtest.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace McBopomofo {

using Formosa::Gramambular2::LanguageModel;
using Formosa::Gramambular2::ReadingGrid;

namespace {

class MapLM : public LanguageModel {
 public:
  void add(const std::string& reading, const std::string& value, double score,
           bool user = false) {
    db_[reading].emplace_back(value, score, value, user);
  }
  std::vector<Unigram> getUnigrams(const std::string& reading) override {
    auto it = db_.find(reading);
    return it == db_.end() ? std::vector<Unigram>() : it->second;
  }
  bool hasUnigrams(const std::string& reading) override {
    return db_.count(reading) > 0;
  }

 private:
  std::map<std::string, std::vector<Unigram>> db_;
};

std::shared_ptr<MapLM> MakeLM() {
  auto lm = std::make_shared<MapLM>();
  lm->add("ㄧˊ", "一", -2.0);
  lm->add("ㄧˊ", "移", -3.0);
  lm->add("ㄓˊ", "直", -3.0);
  lm->add("ㄉㄧㄥˋ", "定", -3.0);
  lm->add("ㄊㄚ", "她", -2.0);
  lm->add("ㄊㄚ", "他", -2.5);
  return lm;
}

std::string Walk(ReadingGrid& grid) {
  std::string out;
  for (const auto& v : grid.walk().valuesAsStrings()) out += v;
  return out;
}

ReadingGrid Grid(const std::shared_ptr<MapLM>& lm,
                 const std::vector<std::string>& readings) {
  ReadingGrid grid(lm);
  for (const auto& r : readings) grid.insertReading(r);
  return grid;
}

}  // namespace

TEST(ConversionRulesTest, SandhiYiBeforeNonFourthToneIsDemoted) {
  auto lm = MakeLM();
  auto grid = Grid(lm, {"ㄧˊ", "ㄓˊ"});
  EXPECT_EQ(Walk(grid), "一直");
  ApplyConversionRules(grid, ConversionRules());
  EXPECT_EQ(Walk(grid), "移直");
}

TEST(ConversionRulesTest, SandhiYiBeforeFourthToneIsKept) {
  auto lm = MakeLM();
  auto grid = Grid(lm, {"ㄧˊ", "ㄉㄧㄥˋ"});
  ApplyConversionRules(grid, ConversionRules());
  EXPECT_EQ(Walk(grid), "一定");
}

TEST(ConversionRulesTest, NeutralPronounsAndSwitches) {
  auto lm = MakeLM();
  auto grid = Grid(lm, {"ㄊㄚ"});
  ConversionRules off;
  off.neutralPronouns = false;
  ApplyConversionRules(grid, off);
  EXPECT_EQ(Walk(grid), "她");
  ApplyConversionRules(grid, ConversionRules());
  EXPECT_EQ(Walk(grid), "他");
}

TEST(ConversionRulesTest, ApplyingTwiceDoesNotStack) {
  auto lm = MakeLM();
  auto grid = Grid(lm, {"ㄊㄚ"});
  ApplyConversionRules(grid, ConversionRules());
  ApplyConversionRules(grid, ConversionRules());
  const auto& node = grid.spans()[0].nodeOf(1);
  ASSERT_NE(node, nullptr);
  for (const auto& u : node->unigrams()) {
    if (u.value() == "她") {
      EXPECT_DOUBLE_EQ(u.score(), -2.0 + kConversionRulePenalty);
    }
  }
}

TEST(ConversionRulesTest, UserPhraseAndOverrideAreKept) {
  auto lm = MakeLM();
  lm->add("ㄋㄧˇ", "妳", -1.0, /*user=*/true);
  lm->add("ㄋㄧˇ", "你", -2.0);
  auto grid = Grid(lm, {"ㄋㄧˇ"});
  ApplyConversionRules(grid, ConversionRules());
  EXPECT_EQ(Walk(grid), "妳");

  auto grid2 = Grid(lm, {"ㄊㄚ"});
  ASSERT_TRUE(grid2.overrideCandidate(0, std::string("她")));
  ApplyConversionRules(grid2, ConversionRules());
  EXPECT_EQ(Walk(grid2), "她");
}

}  // namespace McBopomofo
