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

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

// A pure-Objective-C bridge that exposes the engine's internal scoring to the
// SwiftUI debug inspector. It reuses the shared, already-loaded language model
// and re-runs the same pipeline KeyHandler uses (BopomofoReadingBuffer parse ->
// ReadingGrid walk -> walkNBest -> NeuralRescorer rerank), returning a plain
// data tree the UI renders. The engine is never mutated.

// One candidate value at a grid node.
@interface EISCandidate : NSObject
@property(nonatomic, copy) NSString *value;
@property(nonatomic, copy) NSString *reading;
@property(nonatomic) double score;            // unigram log probability (weight)
@property(nonatomic) BOOL fromUserPhrase;     // boosted: came from a user phrase
@property(nonatomic) BOOL chosen;             // chosen by the plain walk here
@property(nonatomic) BOOL uomSuggested;       // the UserOverrideModel suggests this
@property(nonatomic) BOOL uomForced;          // ...as a high-score (forced) override
@end

// One node (word block) on a walked path.
@interface EISNode : NSObject
@property(nonatomic, copy) NSString *value;
@property(nonatomic, copy) NSString *reading;
@property(nonatomic) NSUInteger spanningLength;
@property(nonatomic) double score;            // chosen unigram log probability
@property(nonatomic) BOOL overridden;         // manual selection / UOM intent
@property(nonatomic) BOOL fromUserPhrase;     // boosted by a user phrase
@property(nonatomic) BOOL uomOverride;        // UserOverrideModel suggests an override here
@property(nonatomic) BOOL uomForced;          // ...as a high-score (forced) override
@property(nonatomic, copy, nullable) NSString *uomValue;  // the UOM-suggested value
@property(nonatomic) BOOL hasNeuralLogProb;   // YES only on the rescored walk
@property(nonatomic) double neuralLogProb;    // this block's neural logP
@property(nonatomic, strong) NSArray<EISCandidate *> *candidates;  // all unigrams
@end

// One n-best candidate sentence.
@interface EISPath : NSObject
@property(nonatomic, strong) NSArray<NSString *> *values;
@property(nonatomic, strong) NSArray<NSString *> *readings;
@property(nonatomic) double unigramScore;     // sum of chosen unigram scores
@property(nonatomic) double modelScore;       // neural sentence log-likelihood
@property(nonatomic) double combinedScore;    // unigram + lambda * model
@property(nonatomic) BOOL hasModelScore;      // NO when rescorer is off/unloaded
@property(nonatomic) BOOL rescorerPick;       // the path the rescorer selected
@property(nonatomic) BOOL plainWalkTop;       // index 0 (== plain walk)
@end

// The full inspection result for one input.
@interface EISResult : NSObject
@property(nonatomic, copy) NSString *asciiInput;
@property(nonatomic, strong) NSArray<NSString *> *readings;  // parsed syllables
@property(nonatomic, strong) NSArray<NSNumber *> *invalidReadingIndexes;
@property(nonatomic, copy, nullable) NSString *incompleteBuffer;  // trailing partial
@property(nonatomic) BOOL hasNonAsciiInput;  // CJK detected -> prompt English input
@property(nonatomic, strong) NSArray<EISNode *> *plainWalk;       // original Viterbi
@property(nonatomic, strong, nullable) NSArray<EISNode *> *rescoredWalk;
@property(nonatomic, strong) NSArray<EISPath *> *nBest;
@property(nonatomic) BOOL rescorerEnabled;
@property(nonatomic) BOOL rescorerLoaded;
@property(nonatomic) double lambda;
@property(nonatomic) NSInteger rescorerPickIndex;
@property(nonatomic) uint64_t walkMicroseconds;
@property(nonatomic) uint64_t rescoreMicroseconds;
@end

@interface EngineInspector : NSObject
// Parses an ASCII key stream (standard Bopomofo layout, honoring the user's
// configured layout) into readings, builds a grid from the shared language
// model, and returns the walk / n-best / rescore breakdown. `k` is the number
// of n-best candidates to enumerate (pass 0 for a sensible default).
+ (EISResult *)inspect:(NSString *)asciiInput nBest:(NSUInteger)k;

// A cheap, synchronous ASCII -> Bopomofo conversion for live display in the
// input field: runs only the reading buffer (no grid / walk / model), returning
// the committed syllables plus any trailing in-progress syllable, space-joined.
// Honors the user's configured keyboard layout.
+ (NSString *)readingDisplayForKeys:(NSString *)asciiKeys;

// Returns the key stream with the last syllable's keys removed (the trailing
// in-progress syllable if any, else the last committed syllable). Used by the
// input field's "delete syllable" shortcut. Returns empty if there is at most
// one syllable.
+ (NSString *)keysByDeletingLastSyllable:(NSString *)asciiKeys;
@end

NS_ASSUME_NONNULL_END
