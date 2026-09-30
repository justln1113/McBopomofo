// Char-level causal Transformer LM used for generative rescoring. See
// Source/Data/rescorer-training (train_tf.py) for the matching trainer.

#ifndef SRC_ENGINE_TRANSFORMERRESCORERMODEL_H_
#define SRC_ENGINE_TRANSFORMERRESCORERMODEL_H_

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "gramambular2/reading_grid.h"

namespace McBopomofo {

// Pre-LN decoder-only Transformer with learned positions and tied output
// embeddings. Beam-search hypotheses share their prefix through a persistent
// linked list of TokNode: each node stores only its own K/V per layer, and
// attention walks the parent chain, so extending a hypothesis never copies the
// prefix. Nodes are materialized lazily (see MakeGuideHooks).
class TransformerRescorerModel
    : public std::enable_shared_from_this<TransformerRescorerModel> {
 public:
  // Throws std::runtime_error on malformed input.
  static std::shared_ptr<TransformerRescorerModel> Load(
      const std::string& weightsPath, const std::string& vocabPath);

  [[nodiscard]] bool coversValue(const std::string& value) const;

  // Hooks for ReadingGrid::walkGuided.
  Formosa::Gramambular2::ReadingGrid::GuideHooks MakeGuideHooks(double lambda);

  // Sum of log P(char | prefix) over the concatenated values (no <eos>), for
  // verification against the PyTorch reference.
  double scoreValues(const std::vector<std::string>& values);

  struct TokNode;
  struct Arena;

  [[nodiscard]] uint32_t dim() const { return d_; }
  [[nodiscard]] uint32_t layers() const { return layers_; }

 private:
  TransformerRescorerModel() = default;

  std::vector<int> tokenize(const std::string& value) const;
  void materialize(TokNode* node) const;
  void materializeBatch(const std::vector<TokNode*>& nodes) const;
  void logNormalizerBatch(const std::vector<TokNode*>& nodes) const;
  double logNormalizer(TokNode* node) const;
  double logit(const TokNode* node, int token) const;
  TokNode* newNode(Arena* arena, TokNode* parent, int token) const;
  double extend(Arena* arena, TokNode* prev, const std::string& value,
                TokNode** next) const;

  struct Layer {
    std::vector<float> ln1w, ln1b, qkvW, qkvB, projW, projB;
    std::vector<float> ln2w, ln2b, ff1W, ff1B, ff2W, ff2B;
  };

  uint32_t vocab_ = 0, d_ = 0, layers_ = 0, heads_ = 0, ff_ = 0, maxLen_ = 0;
  std::vector<float> embed_;  // [V, d]
  std::vector<float> pos_;    // [maxLen, d]
  std::vector<Layer> layer_;
  std::vector<float> normW_, normB_;

  std::unordered_map<std::string, int> tokenToId_;
  int bosId_ = -1, unkId_ = -1;
};

}  // namespace McBopomofo

#endif  // SRC_ENGINE_TRANSFORMERRESCORERMODEL_H_
