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

#import "EngineInspector.h"

#import "LanguageModelManager+Privates.h"
#import "LstmRescorerModel.h"
#import "Mandarin.h"
#import "McBopomofo-Swift.h"
#import "McBopomofoLM.h"
#import "NeuralRescorer.h"
#import "UserOverrideModel.h"
#import "reading_grid.h"

#include <algorithm>
#include <chrono>
#include <memory>
#include <string>
#include <utility>
#include <vector>

@implementation EISCandidate
@synthesize value;
@synthesize reading;
@synthesize score;
@synthesize fromUserPhrase;
@synthesize chosen;
@synthesize uomSuggested;
@synthesize uomForced;
@end

@implementation EISNode
@synthesize value;
@synthesize reading;
@synthesize spanningLength;
@synthesize score;
@synthesize overridden;
@synthesize fromUserPhrase;
@synthesize uomOverride;
@synthesize uomForced;
@synthesize uomValue;
@synthesize hasNeuralLogProb;
@synthesize neuralLogProb;
@synthesize candidates;
@end

@implementation EISPath
@synthesize values;
@synthesize readings;
@synthesize unigramScore;
@synthesize modelScore;
@synthesize combinedScore;
@synthesize hasModelScore;
@synthesize rescorerPick;
@synthesize plainWalkTop;
@end

@implementation EISResult
@synthesize asciiInput;
@synthesize readings;
@synthesize invalidReadingIndexes;
@synthesize incompleteBuffer;
@synthesize hasNonAsciiInput;
@synthesize plainWalk;
@synthesize rescoredWalk;
@synthesize nBest;
@synthesize rescorerEnabled;
@synthesize rescorerLoaded;
@synthesize lambda;
@synthesize rescorerPickIndex;
@synthesize walkMicroseconds;
@synthesize rescoreMicroseconds;
@end

namespace {

using Formosa::Gramambular2::LanguageModel;
using Formosa::Gramambular2::ReadingGrid;

// Loads the bundled rescorer model exactly once. The weights are immutable, so
// the inspector shares a single LstmRescorerModel across calls; each inspect
// uses its own NeuralRescorer (which holds the per-run prefix cache). Returns
// nullptr if the model isn't bundled or fails to load.
std::shared_ptr<McBopomofo::LstmRescorerModel> SharedRescorerModel() {
  static std::shared_ptr<McBopomofo::LstmRescorerModel> sModel;
  static dispatch_once_t onceToken;
  dispatch_once(&onceToken, ^{
    NSBundle* bundle = [NSBundle bundleForClass:[EngineInspector class]];
    NSString* weightsPath = [bundle pathForResource:@"rescorer-weights"
                                             ofType:@"bin"];
    NSString* vocabPath = [bundle pathForResource:@"rescorer-vocab"
                                           ofType:@"txt"];
    if (weightsPath == nil || vocabPath == nil) {
      return;
    }
    try {
      sModel = McBopomofo::LstmRescorerModel::Load(weightsPath.UTF8String,
                                                   vocabPath.UTF8String);
    } catch (const std::exception& e) {
      NSLog(@"EngineInspector: failed to load rescorer model: %s", e.what());
      sModel = nullptr;
    }
  });
  return sModel;
}

const Formosa::Mandarin::BopomofoKeyboardLayout* CurrentKeyboardLayout() {
  using Layout = Formosa::Mandarin::BopomofoKeyboardLayout;
  switch (Preferences.keyboardLayout) {
    case KeyboardLayoutEten:
      return Layout::ETenLayout();
    case KeyboardLayoutHsu:
      return Layout::HsuLayout();
    case KeyboardLayoutEten26:
      return Layout::ETen26Layout();
    case KeyboardLayoutHanyuPinyin:
      return Layout::HanyuPinyinLayout();
    case KeyboardLayoutIBM:
      return Layout::IBMLayout();
    case KeyboardLayoutStandard:
    default:
      return Layout::StandardLayout();
  }
}

NSString* NSStr(const std::string& s) {
  NSString* r = [NSString stringWithUTF8String:s.c_str()];
  return r != nil ? r : @"";
}

// Builds the score-ranked candidate list for a node's (possibly combined)
// reading, marking which one matches the chosen value and which one the
// UserOverrideModel currently suggests for this position.
NSArray<EISCandidate*>* BuildCandidates(McBopomofo::McBopomofoLM* lm,
                                        const std::string& reading,
                                        const std::string& chosenValue,
                                        const std::string& uomValue,
                                        bool uomForced) {
  std::vector<LanguageModel::Unigram> unigrams = lm->getUnigrams(reading);
  std::stable_sort(unigrams.begin(), unigrams.end(),
                   [](const LanguageModel::Unigram& a,
                      const LanguageModel::Unigram& b) {
                     return a.score() > b.score();
                   });
  NSMutableArray<EISCandidate*>* arr =
      [NSMutableArray arrayWithCapacity:unigrams.size()];
  bool marked = false;
  for (const auto& u : unigrams) {
    EISCandidate* c = [[EISCandidate alloc] init];
    c.value = NSStr(u.value());
    c.reading = NSStr(reading);
    c.score = u.score();
    c.fromUserPhrase = u.isFromUserPhrase();
    c.chosen = (!marked && u.value() == chosenValue);
    if (c.chosen) {
      marked = true;
    }
    c.uomSuggested = (!uomValue.empty() && u.value() == uomValue);
    c.uomForced = c.uomSuggested && uomForced;
    [arr addObject:c];
  }
  return arr;
}

EISNode* BuildNode(McBopomofo::McBopomofoLM* lm, const ReadingGrid::NodePtr& node,
                   const std::string& uomValue, bool uomForced) {
  EISNode* n = [[EISNode alloc] init];
  std::string value = node->value();
  n.value = NSStr(value);
  n.reading = NSStr(node->reading());
  n.spanningLength = node->spanningLength();
  n.score = node->score();
  n.overridden = node->isOverridden();
  n.uomOverride = !uomValue.empty();
  n.uomForced = uomForced;
  n.uomValue = uomValue.empty() ? nil : NSStr(uomValue);
  n.hasNeuralLogProb = NO;
  n.neuralLogProb = 0;
  n.candidates = BuildCandidates(lm, node->reading(), value, uomValue, uomForced);
  n.fromUserPhrase = NO;
  for (EISCandidate* c in n.candidates) {
    if (c.chosen) {
      n.fromUserPhrase = c.fromUserPhrase;
      break;
    }
  }
  return n;
}

// Builds display nodes for a walk, annotating each with the UserOverrideModel's
// current suggestion for that position (so the UI can show override / decay
// provenance). `uom` may be null and `now` is a UNIX timestamp for decay.
NSArray<EISNode*>* BuildNodes(McBopomofo::McBopomofoLM* lm,
                              McBopomofo::UserOverrideModel* uom,
                              const ReadingGrid::WalkResult& walk, double now) {
  NSMutableArray<EISNode*>* arr =
      [NSMutableArray arrayWithCapacity:walk.nodes.size()];
  size_t offset = 0;
  for (const auto& node : walk.nodes) {
    std::string uomValue;
    bool uomForced = false;
    if (uom != nullptr) {
      McBopomofo::UserOverrideModel::Suggestion suggestion =
          uom->suggest(walk, offset, now);
      if (!suggestion.empty()) {
        uomValue = suggestion.candidate;
        uomForced = suggestion.forceHighScoreOverride;
      }
    }
    [arr addObject:BuildNode(lm, node, uomValue, uomForced)];
    offset += node->spanningLength();
  }
  return arr;
}

// The result of parsing an ASCII key stream through the reading buffer.
struct ParsedKeys {
  std::vector<std::string> syllables;  // committed readings, in order
  std::string partial;                 // trailing in-progress syllable, if any
  bool hasNonAscii = false;
};

// Replays an ASCII key stream through the engine's BopomofoReadingBuffer using
// the same syllable-commit rule as KeyHandler: a syllable commits when it has a
// tone marker (and isn't a lone tone marker), or when Space/Enter is pressed
// (first tone). Validity (hasUnigrams) is NOT checked here.
ParsedKeys ParseKeys(NSString* asciiKeys) {
  ParsedKeys out;
  Formosa::Mandarin::BopomofoReadingBuffer buffer(CurrentKeyboardLayout());
  NSUInteger length = asciiKeys.length;
  for (NSUInteger i = 0; i < length; ++i) {
    unichar ch = [asciiKeys characterAtIndex:i];
    if (ch >= 128) {
      out.hasNonAscii = true;
      continue;
    }
    char key = (char)ch;
    BOOL isWhitespace =
        (key == ' ' || key == '\n' || key == '\r' || key == '\t');
    if (isWhitespace) {
      if (!buffer.isEmpty()) {
        std::string reading = buffer.syllable().composedString();
        buffer.clear();
        if (!reading.empty()) {
          out.syllables.push_back(reading);
        }
      }
      continue;
    }
    // A stray tone marker starts a new syllable when the next key is not a tone.
    if (buffer.hasToneMarkerOnly()) {
      Formosa::Mandarin::BopomofoReadingBuffer probe(buffer.keyboardLayout());
      probe.combineKey(key);
      if (!probe.hasToneMarkerOnly()) {
        buffer.clear();
      }
    }
    buffer.combineKey(key);
    if (buffer.hasToneMarker() && !buffer.hasToneMarkerOnly()) {
      std::string reading = buffer.syllable().composedString();
      buffer.clear();
      if (!reading.empty()) {
        out.syllables.push_back(reading);
      }
    }
  }
  if (!buffer.isEmpty()) {
    out.partial = buffer.composedString();
  }
  return out;
}

// Drives the model over a value sequence to get each block's neural log prob
// (full-sentence log-likelihood term). No prefix cache -- the sequences are
// short and few.
std::vector<double> NeuralLogProbs(
    const std::shared_ptr<McBopomofo::LstmRescorerModel>& model,
    const std::vector<std::string>& values) {
  std::vector<double> out;
  out.reserve(values.size());
  McBopomofo::RescorerModelState state = model->initialState();
  for (const auto& v : values) {
    auto [logProb, next] = model->step(state, v, {});
    out.push_back(logProb);
    state = std::move(next);
  }
  return out;
}

}  // namespace

@implementation EngineInspector

+ (EISResult*)inspect:(NSString*)asciiInput nBest:(NSUInteger)k {
  if (k == 0) {
    k = 5;
  }
  EISResult* result = [[EISResult alloc] init];
  result.asciiInput = asciiInput != nil ? asciiInput : @"";
  result.readings = @[];
  result.invalidReadingIndexes = @[];
  result.plainWalk = @[];
  result.nBest = @[];
  result.rescorerPickIndex = -1;

  McBopomofo::McBopomofoLM* lm = [LanguageModelManager languageModelMcBopomofo];
  if (lm == nullptr) {
    return result;
  }

  // 1. Parse the ASCII key stream into readings using the engine's own buffer,
  //    then check each committed syllable against the language model.
  ParsedKeys parsed = ParseKeys(result.asciiInput);
  NSMutableArray<NSString*>* readings = [NSMutableArray array];
  NSMutableArray<NSNumber*>* invalid = [NSMutableArray array];
  std::vector<std::string> validReadings;
  for (const auto& reading : parsed.syllables) {
    if (!lm->hasUnigrams(reading)) {
      [invalid addObject:@(readings.count)];
    } else {
      validReadings.push_back(reading);
    }
    [readings addObject:NSStr(reading)];
  }

  result.readings = readings;
  result.invalidReadingIndexes = invalid;
  result.hasNonAsciiInput = parsed.hasNonAscii;
  if (!parsed.partial.empty()) {
    result.incompleteBuffer = NSStr(parsed.partial);
  }

  if (validReadings.empty()) {
    return result;
  }

  // 2. Build the grid from the valid readings (same wiring as KeyHandler).
  std::shared_ptr<LanguageModel> emptyShared;
  std::shared_ptr<LanguageModel> lmShared(emptyShared, lm);
  ReadingGrid grid(lmShared);
  grid.setReadingSeparator("-");
  for (const auto& reading : validReadings) {
    grid.insertReading(reading);
  }

  // 3. Plain Viterbi walk. Annotate nodes with the live UserOverrideModel's
  //    current suggestions (decay is evaluated against `now`).
  McBopomofo::UserOverrideModel* uom = [LanguageModelManager userOverrideModel];
  double now = [[NSDate date] timeIntervalSince1970];
  ReadingGrid::WalkResult walk = grid.walk();
  result.walkMicroseconds = walk.elapsedMicroseconds;
  result.plainWalk = BuildNodes(lm, uom, walk, now);

  // 4. N-best alternatives.
  std::vector<ReadingGrid::NBestPath> paths = grid.walkNBest(k);

  // 5. Rescore (if enabled and the model is bundled).
  result.rescorerEnabled = Preferences.neuralRescorerEnabled;
  std::shared_ptr<McBopomofo::LstmRescorerModel> model;
  if (result.rescorerEnabled) {
    model = SharedRescorerModel();
  }
  result.rescorerLoaded = (model != nullptr);

  std::vector<double> pathModelScores(paths.size(), 0.0);
  size_t best = 0;
  double lambda = 1.0;

  if (model != nullptr && !paths.empty()) {
    auto start = std::chrono::steady_clock::now();
    McBopomofo::NeuralRescorer rescorer(model);
    lambda = rescorer.lambda();
    // The engine's actual pick (honors the user-override guard).
    best = rescorer.rerankBestIndex(paths);
    // Independent per-path model scores for display.
    for (size_t i = 0; i < paths.size(); ++i) {
      const auto& values = paths[i].values;
      double sum = 0.0;
      for (double lp : NeuralLogProbs(model, values)) {
        sum += lp;
      }
      pathModelScores[i] = sum;
    }
    auto end = std::chrono::steady_clock::now();
    result.rescoreMicroseconds =
        (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(
            end - start)
            .count();
    result.rescorerPickIndex = (NSInteger)best;

    // Rescored walk (built even when best == 0, so the neural coloring shows).
    ReadingGrid::WalkResult rescored = grid.walkResultFromPath(paths[best]);
    NSArray<EISNode*>* rescoredNodes = BuildNodes(lm, uom, rescored, now);
    std::vector<double> segLogProbs = NeuralLogProbs(model, paths[best].values);
    for (NSUInteger i = 0;
         i < rescoredNodes.count && i < segLogProbs.size(); ++i) {
      rescoredNodes[i].hasNeuralLogProb = YES;
      rescoredNodes[i].neuralLogProb = segLogProbs[i];
    }
    result.rescoredWalk = rescoredNodes;
  }
  result.lambda = lambda;

  // 6. N-best table.
  NSMutableArray<EISPath*>* nbest =
      [NSMutableArray arrayWithCapacity:paths.size()];
  for (size_t i = 0; i < paths.size(); ++i) {
    EISPath* p = [[EISPath alloc] init];
    NSMutableArray<NSString*>* values =
        [NSMutableArray arrayWithCapacity:paths[i].values.size()];
    for (const auto& v : paths[i].values) {
      [values addObject:NSStr(v)];
    }
    NSMutableArray<NSString*>* pathReadings =
        [NSMutableArray arrayWithCapacity:paths[i].readings.size()];
    for (const auto& r : paths[i].readings) {
      [pathReadings addObject:NSStr(r)];
    }
    p.values = values;
    p.readings = pathReadings;
    p.unigramScore = paths[i].score;
    p.plainWalkTop = (i == 0);
    if (model != nullptr) {
      p.hasModelScore = YES;
      p.modelScore = pathModelScores[i];
      p.combinedScore = paths[i].score + lambda * pathModelScores[i];
      p.rescorerPick = (i == best);
    }
    [nbest addObject:p];
  }
  result.nBest = nbest;

  return result;
}

+ (NSString*)readingDisplayForKeys:(NSString*)asciiKeys {
  if (asciiKeys.length == 0) {
    return @"";
  }
  ParsedKeys parsed = ParseKeys(asciiKeys);
  NSMutableArray<NSString*>* parts =
      [NSMutableArray arrayWithCapacity:parsed.syllables.size() + 1];
  for (const auto& syllable : parsed.syllables) {
    [parts addObject:NSStr(syllable)];
  }
  if (!parsed.partial.empty()) {
    [parts addObject:NSStr(parsed.partial)];
  }
  return [parts componentsJoinedByString:@" "];
}

+ (NSString*)keysByDeletingLastSyllable:(NSString*)asciiKeys {
  NSUInteger length = asciiKeys.length;
  if (length == 0) {
    return @"";
  }
  // Replay the keys, tracking the start index of the syllable currently being
  // built and of the last syllable emitted (committed or left in progress).
  Formosa::Mandarin::BopomofoReadingBuffer buffer(CurrentKeyboardLayout());
  NSUInteger unitStart = 0;
  NSUInteger lastEmittedStart = 0;
  BOOL emittedAny = NO;
  BOOL building = NO;
  for (NSUInteger i = 0; i < length; ++i) {
    unichar ch = [asciiKeys characterAtIndex:i];
    if (ch >= 128) {
      continue;
    }
    char key = (char)ch;
    BOOL isWhitespace =
        (key == ' ' || key == '\n' || key == '\r' || key == '\t');
    if (isWhitespace) {
      if (!buffer.isEmpty()) {
        lastEmittedStart = unitStart;
        emittedAny = YES;
        buffer.clear();
        building = NO;
      }
      unitStart = i + 1;
      continue;
    }
    if (!building) {
      unitStart = i;
      building = YES;
    }
    if (buffer.hasToneMarkerOnly()) {
      Formosa::Mandarin::BopomofoReadingBuffer probe(buffer.keyboardLayout());
      probe.combineKey(key);
      if (!probe.hasToneMarkerOnly()) {
        buffer.clear();
      }
    }
    buffer.combineKey(key);
    if (buffer.hasToneMarker() && !buffer.hasToneMarkerOnly()) {
      lastEmittedStart = unitStart;
      emittedAny = YES;
      buffer.clear();
      building = NO;
      unitStart = i + 1;
    }
  }
  if (building) {
    lastEmittedStart = unitStart;
    emittedAny = YES;
  }
  if (!emittedAny) {
    return @"";
  }
  return [asciiKeys substringToIndex:lastEmittedStart];
}

@end
