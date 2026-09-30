#include "TransformerRescorerModel.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <deque>
#include <fstream>
#include <limits>
#include <stdexcept>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#elif defined(__AVX2__) || defined(_M_X64)
#include <immintrin.h>
#endif

#include "UTF8Helper.h"

namespace McBopomofo {

namespace {

constexpr char kMagic[8] = {'M', 'C', 'B', 'T', 'R', 'F', '1', '\0'};

// Dot product; every matrix product below goes through it. NEON on arm64
// (the LstmRescorerModel kernel with two accumulators), AVX2+FMA on x86,
// scalar otherwise.
// MCB_RESCORER_FORCE_SCALAR forces the scalar loop (for comparisons).
inline float Dot(const float* a, const float* b, size_t n) {
#if defined(__ARM_NEON) && !defined(MCB_RESCORER_FORCE_SCALAR)
  float32x4_t acc0 = vdupq_n_f32(0.0f);
  float32x4_t acc1 = vdupq_n_f32(0.0f);
  size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    acc0 = vfmaq_f32(acc0, vld1q_f32(a + i), vld1q_f32(b + i));
    acc1 = vfmaq_f32(acc1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
  }
  for (; i + 4 <= n; i += 4) {
    acc0 = vfmaq_f32(acc0, vld1q_f32(a + i), vld1q_f32(b + i));
  }
  float sum = vaddvq_f32(vaddq_f32(acc0, acc1));
  for (; i < n; ++i) sum += a[i] * b[i];
  return sum;
#elif defined(__AVX2__) && (defined(__FMA__) || defined(_MSC_VER)) && \
    !defined(MCB_RESCORER_FORCE_SCALAR)
  __m256 acc0 = _mm256_setzero_ps();
  __m256 acc1 = _mm256_setzero_ps();
  size_t i = 0;
  for (; i + 16 <= n; i += 16) {
    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc0);
    acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8),
                           _mm256_loadu_ps(b + i + 8), acc1);
  }
  acc0 = _mm256_add_ps(acc0, acc1);
  for (; i + 8 <= n; i += 8) {
    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc0);
  }
  __m128 lo = _mm256_castps256_ps128(acc0);
  __m128 hi = _mm256_extractf128_ps(acc0, 1);
  lo = _mm_add_ps(lo, hi);
  lo = _mm_hadd_ps(lo, lo);
  lo = _mm_hadd_ps(lo, lo);
  float sum = _mm_cvtss_f32(lo);
  for (; i < n; ++i) sum += a[i] * b[i];
  return sum;
#else
  float sum = 0.0f;
  for (size_t i = 0; i < n; ++i) sum += a[i] * b[i];
  return sum;
#endif
}

// y[r] = b[r] + W[r,:] . x for W of shape [out, in], row-major.
void MatVec(const float* W, const float* b, const float* x, float* y,
            size_t out, size_t in) {
  for (size_t r = 0; r < out; ++r) y[r] = b[r] + Dot(W + r * in, x, in);
}

void LayerNorm(const float* x, const float* w, const float* b, float* y,
               size_t n) {
  double mean = 0.0;
  for (size_t i = 0; i < n; ++i) mean += x[i];
  mean /= static_cast<double>(n);
  double var = 0.0;
  for (size_t i = 0; i < n; ++i) {
    double c = x[i] - mean;
    var += c * c;
  }
  var /= static_cast<double>(n);
  const float inv = static_cast<float>(1.0 / std::sqrt(var + 1e-5));
  for (size_t i = 0; i < n; ++i) {
    y[i] = (x[i] - static_cast<float>(mean)) * inv * w[i] + b[i];
  }
}

inline float Gelu(float x) {
  return 0.5f * x * (1.0f + std::erf(x * 0.70710678118654752f));
}

class Reader {
 public:
  explicit Reader(std::vector<char> buf) : buf_(std::move(buf)) {}
  template <typename T>
  T get() {
    T v;
    need(sizeof(T));
    std::memcpy(&v, buf_.data() + pos_, sizeof(T));
    pos_ += sizeof(T);
    return v;
  }
  void floats(std::vector<float>* out, size_t n) {
    need(n * sizeof(float));
    out->resize(n);
    std::memcpy(out->data(), buf_.data() + pos_, n * sizeof(float));
    pos_ += n * sizeof(float);
  }
  void expect(const char* magic, size_t n) {
    need(n);
    if (std::memcmp(buf_.data() + pos_, magic, n) != 0) {
      throw std::runtime_error("TransformerRescorerModel: bad magic");
    }
    pos_ += n;
  }
  [[nodiscard]] bool atEnd() const { return pos_ == buf_.size(); }

 private:
  void need(size_t n) const {
    if (pos_ + n > buf_.size()) {
      throw std::runtime_error("TransformerRescorerModel: truncated weights");
    }
  }
  std::vector<char> buf_;
  size_t pos_ = 0;
};

}  // namespace

struct TransformerRescorerModel::TokNode {
  TokNode* parent = nullptr;  // owned by the same Arena
  std::vector<std::pair<int, TokNode*>> children;  // one node per next token
  int token = 0;
  int pos = 0;
  bool ready = false;
  std::vector<float> k, v;  // layers * d each, this token only
  std::vector<float> out;   // d: final-normed hidden -> next-token logits
  bool hasLogZ = false;
  double logZ = 0.0;
};

// Owns every node of one search; nodes point at each other with raw pointers, so
// there are no ownership cycles and shared prefixes/children are deduplicated.
struct TransformerRescorerModel::Arena {
  std::deque<TokNode> nodes;
};

std::shared_ptr<TransformerRescorerModel> TransformerRescorerModel::Load(
    const std::string& weightsPath, const std::string& vocabPath) {
  std::ifstream wf(weightsPath, std::ios::binary);
  if (!wf) throw std::runtime_error("cannot open " + weightsPath);
  std::vector<char> buf((std::istreambuf_iterator<char>(wf)),
                        std::istreambuf_iterator<char>());
  Reader r(std::move(buf));
  r.expect(kMagic, sizeof(kMagic));

  std::shared_ptr<TransformerRescorerModel> m(new TransformerRescorerModel());
  m->vocab_ = r.get<uint32_t>();
  m->d_ = r.get<uint32_t>();
  m->layers_ = r.get<uint32_t>();
  m->heads_ = r.get<uint32_t>();
  m->ff_ = r.get<uint32_t>();
  m->maxLen_ = r.get<uint32_t>();
  if (m->d_ == 0 || m->heads_ == 0 || m->d_ % m->heads_ != 0 ||
      m->layers_ == 0 || m->vocab_ == 0) {
    throw std::runtime_error("TransformerRescorerModel: bad dimensions");
  }
  const size_t D = m->d_, F = m->ff_;
  r.floats(&m->embed_, static_cast<size_t>(m->vocab_) * D);
  r.floats(&m->pos_, static_cast<size_t>(m->maxLen_) * D);
  m->layer_.resize(m->layers_);
  for (auto& L : m->layer_) {
    r.floats(&L.ln1w, D);
    r.floats(&L.ln1b, D);
    r.floats(&L.qkvW, 3 * D * D);
    r.floats(&L.qkvB, 3 * D);
    r.floats(&L.projW, D * D);
    r.floats(&L.projB, D);
    r.floats(&L.ln2w, D);
    r.floats(&L.ln2b, D);
    r.floats(&L.ff1W, F * D);
    r.floats(&L.ff1B, F);
    r.floats(&L.ff2W, D * F);
    r.floats(&L.ff2B, D);
  }
  r.floats(&m->normW_, D);
  r.floats(&m->normB_, D);
  if (!r.atEnd()) {
    throw std::runtime_error("TransformerRescorerModel: trailing bytes");
  }

  std::ifstream vf(vocabPath);
  if (!vf) throw std::runtime_error("cannot open " + vocabPath);
  std::string line;
  int id = 0;
  while (std::getline(vf, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    m->tokenToId_.emplace(line, id++);
  }
  if (static_cast<uint32_t>(id) != m->vocab_) {
    throw std::runtime_error("TransformerRescorerModel: vocab size mismatch");
  }
  auto bos = m->tokenToId_.find("<bos>");
  auto unk = m->tokenToId_.find("<unk>");
  if (bos == m->tokenToId_.end() || unk == m->tokenToId_.end()) {
    throw std::runtime_error("TransformerRescorerModel: missing <bos>/<unk>");
  }
  m->bosId_ = bos->second;
  m->unkId_ = unk->second;
  return m;
}

std::vector<int> TransformerRescorerModel::tokenize(
    const std::string& value) const {
  std::vector<int> ids;
  for (const std::string& ch : Split(value)) {
    auto it = tokenToId_.find(ch);
    ids.push_back(it == tokenToId_.end() ? unkId_ : it->second);
  }
  return ids;
}

bool TransformerRescorerModel::coversValue(const std::string& value) const {
  for (const std::string& ch : Split(value)) {
    if (tokenToId_.find(ch) == tokenToId_.end()) return false;
  }
  return true;
}

namespace {
// Y[b, r] = bias[r] + W[r,:] . X[b,:]  for a batch of B input rows. The weight row
// is the outer loop so it is read once and reused for all B rows.
void MatMulBatch(const float* W, const float* bias, const float* X, float* Y,
                 size_t B, size_t out, size_t in) {
  for (size_t r = 0; r < out; ++r) {
    const float* w = W + r * in;
    const float br = bias[r];
    for (size_t b = 0; b < B; ++b) {
      Y[b * out + r] = br + Dot(w, X + b * in, in);
    }
  }
}
}  // namespace

void TransformerRescorerModel::materialize(TokNode* node) const {
  materializeBatch({node});
}

void TransformerRescorerModel::materializeBatch(
    const std::vector<TokNode*>& nodesIn) const {
  std::vector<TokNode*> nodes;
  for (TokNode* n : nodesIn) {
    if (!n->ready) nodes.push_back(n);
  }
  std::sort(nodes.begin(), nodes.end());
  nodes.erase(std::unique(nodes.begin(), nodes.end()), nodes.end());
  const size_t B = nodes.size();
  if (B == 0) return;
  const size_t D = d_, H = heads_, dh = D / H, F = ff_;

  std::vector<std::vector<const TokNode*>> chains(B);
  for (size_t b = 0; b < B; ++b) {
    for (const TokNode* n = nodes[b]; n != nullptr; n = n->parent) {
      chains[b].push_back(n);
    }
    std::reverse(chains[b].begin(), chains[b].end());
  }

  std::vector<float> x(B * D), xn(B * D), qkv(B * 3 * D), ctx(B * D),
      attn(B * D), hid(B * F), y(B * D);
  size_t maxT = 0;
  for (const auto& c : chains) maxT = std::max(maxT, c.size());
  std::vector<float> scores(maxT);
  for (size_t b = 0; b < B; ++b) {
    TokNode* node = nodes[b];
    const size_t p =
        std::min<size_t>(static_cast<size_t>(node->pos), maxLen_ - 1);
    for (size_t i = 0; i < D; ++i) {
      x[b * D + i] =
          embed_[static_cast<size_t>(node->token) * D + i] + pos_[p * D + i];
    }
    node->k.assign(layers_ * D, 0.0f);
    node->v.assign(layers_ * D, 0.0f);
  }
  const float scale = 1.0f / std::sqrt(static_cast<float>(dh));

  for (size_t l = 0; l < layers_; ++l) {
    const Layer& L = layer_[l];
    for (size_t b = 0; b < B; ++b) {
      LayerNorm(&x[b * D], L.ln1w.data(), L.ln1b.data(), &xn[b * D], D);
    }
    MatMulBatch(L.qkvW.data(), L.qkvB.data(), xn.data(), qkv.data(), B, 3 * D,
                D);
    for (size_t b = 0; b < B; ++b) {
      TokNode* node = nodes[b];
      const float* q = &qkv[b * 3 * D];
      std::memcpy(node->k.data() + l * D, q + D, D * sizeof(float));
      std::memcpy(node->v.data() + l * D, q + 2 * D, D * sizeof(float));
      const auto& chain = chains[b];
      const size_t T = chain.size();
      for (size_t h = 0; h < H; ++h) {
        float maxScore = -std::numeric_limits<float>::infinity();
        for (size_t t = 0; t < T; ++t) {
          const float* kt = chain[t]->k.data() + l * D + h * dh;
          scores[t] = Dot(q + h * dh, kt, dh) * scale;
          maxScore = std::max(maxScore, scores[t]);
        }
        float sum = 0.0f;
        for (size_t t = 0; t < T; ++t) {
          scores[t] = std::exp(scores[t] - maxScore);
          sum += scores[t];
        }
        const float inv = 1.0f / sum;
        float* c = &ctx[b * D + h * dh];
        for (size_t j = 0; j < dh; ++j) c[j] = 0.0f;
        for (size_t t = 0; t < T; ++t) {
          const float w = scores[t] * inv;
          const float* vt = chain[t]->v.data() + l * D + h * dh;
          for (size_t j = 0; j < dh; ++j) c[j] += w * vt[j];
        }
      }
    }
    MatMulBatch(L.projW.data(), L.projB.data(), ctx.data(), attn.data(), B, D,
                D);
    for (size_t i = 0; i < B * D; ++i) x[i] += attn[i];

    for (size_t b = 0; b < B; ++b) {
      LayerNorm(&x[b * D], L.ln2w.data(), L.ln2b.data(), &xn[b * D], D);
    }
    MatMulBatch(L.ff1W.data(), L.ff1B.data(), xn.data(), hid.data(), B, F, D);
    for (size_t i = 0; i < B * F; ++i) hid[i] = Gelu(hid[i]);
    MatMulBatch(L.ff2W.data(), L.ff2B.data(), hid.data(), y.data(), B, D, F);
    for (size_t i = 0; i < B * D; ++i) x[i] += y[i];
  }
  for (size_t b = 0; b < B; ++b) {
    nodes[b]->out.resize(D);
    LayerNorm(&x[b * D], normW_.data(), normB_.data(), nodes[b]->out.data(), D);
    nodes[b]->ready = true;
  }
}

void TransformerRescorerModel::logNormalizerBatch(
    const std::vector<TokNode*>& nodesIn) const {
  std::vector<TokNode*> nodes;
  for (TokNode* n : nodesIn) {
    if (n->ready && !n->hasLogZ) nodes.push_back(n);
  }
  std::sort(nodes.begin(), nodes.end());
  nodes.erase(std::unique(nodes.begin(), nodes.end()), nodes.end());
  const size_t B = nodes.size();
  if (B == 0) return;
  static thread_local std::vector<float> logits;
  logits.resize(B * vocab_);
  for (size_t t = 0; t < vocab_; ++t) {
    const float* row = embed_.data() + t * d_;
    for (size_t b = 0; b < B; ++b) {
      logits[b * vocab_ + t] = Dot(row, nodes[b]->out.data(), d_);
    }
  }
  for (size_t b = 0; b < B; ++b) {
    const float* lg = &logits[b * vocab_];
    float maxLogit = -std::numeric_limits<float>::infinity();
    for (size_t t = 0; t < vocab_; ++t) maxLogit = std::max(maxLogit, lg[t]);
    double sum = 0.0;
    for (size_t t = 0; t < vocab_; ++t) {
      sum += std::exp(static_cast<double>(lg[t] - maxLogit));
    }
    nodes[b]->logZ = static_cast<double>(maxLogit) + std::log(sum);
    nodes[b]->hasLogZ = true;
  }
}
double TransformerRescorerModel::logit(const TokNode* node, int token) const {
  return Dot(embed_.data() + static_cast<size_t>(token) * d_, node->out.data(),
             d_);
}

double TransformerRescorerModel::logNormalizer(TokNode* node) const {
  if (node->hasLogZ) return node->logZ;
  static thread_local std::vector<float> logits;
  logits.resize(vocab_);
  float maxLogit = -std::numeric_limits<float>::infinity();
  for (size_t t = 0; t < vocab_; ++t) {
    float l = Dot(embed_.data() + t * d_, node->out.data(), d_);
    logits[t] = l;
    maxLogit = std::max(maxLogit, l);
  }
  double sum = 0.0;
  for (size_t t = 0; t < vocab_; ++t) {
    sum += std::exp(static_cast<double>(logits[t] - maxLogit));
  }
  node->logZ = static_cast<double>(maxLogit) + std::log(sum);
  node->hasLogZ = true;
  return node->logZ;
}

TransformerRescorerModel::TokNode* TransformerRescorerModel::newNode(
    Arena* arena, TokNode* parent, int token) const {
  if (parent != nullptr) {
    for (const auto& c : parent->children) {
      if (c.first == token) return c.second;
    }
  }
  arena->nodes.emplace_back();
  TokNode* n = &arena->nodes.back();
  n->parent = parent;
  n->token = token;
  n->pos = parent ? parent->pos + 1 : 0;
  if (parent != nullptr) parent->children.emplace_back(token, n);
  return n;
}

double TransformerRescorerModel::extend(Arena* arena, TokNode* prev,
                                        const std::string& value,
                                        TokNode** next) const {
  std::vector<int> ids = tokenize(value);
  if (ids.empty()) {
    *next = prev;
    return 0.0;
  }
  TokNode* cur = prev;
  double lp = 0.0;
  for (int id : ids) {
    materialize(cur);  // parent chain is ready by invariant
    lp += logit(cur, id) - logNormalizer(cur);
    cur = newNode(arena, cur, id);  // lazy; materialized only if extended
  }
  *next = cur;
  return lp;
}

Formosa::Gramambular2::ReadingGrid::GuideHooks
TransformerRescorerModel::MakeGuideHooks(double lambda) {
  Formosa::Gramambular2::ReadingGrid::GuideHooks hooks;
  hooks.lambda = lambda;
  auto self = shared_from_this();
  auto arena = std::make_shared<Arena>();
  // States handed to the grid alias the arena so it outlives every state.
  auto wrap = [arena](TokNode* n) -> std::shared_ptr<void> {
    return std::shared_ptr<void>(arena, static_cast<void*>(n));
  };
  hooks.initialState = [self, arena, wrap]() -> std::shared_ptr<void> {
    TokNode* bos = self->newNode(arena.get(), nullptr, self->bosId_);
    self->materialize(bos);
    return wrap(bos);
  };
  hooks.extend = [self, arena, wrap](const std::shared_ptr<void>& prevRaw,
                                     const std::string& value,
                                     std::shared_ptr<void>* next) -> double {
    TokNode* out = nullptr;
    double lp = self->extend(arena.get(), static_cast<TokNode*>(prevRaw.get()),
                             value, &out);
    *next = wrap(out);
    return lp;
  };
  hooks.prepare = [self](const std::vector<std::shared_ptr<void>>& states) {
    std::vector<TokNode*> nodes;
    nodes.reserve(states.size());
    for (const auto& s : states) nodes.push_back(static_cast<TokNode*>(s.get()));
    self->materializeBatch(nodes);
    self->logNormalizerBatch(nodes);
  };
  hooks.covers = [self](const std::string& v) { return self->coversValue(v); };
  return hooks;
}

double TransformerRescorerModel::scoreValues(
    const std::vector<std::string>& values) {
  Arena arena;
  TokNode* cur = newNode(&arena, nullptr, bosId_);
  materialize(cur);
  double total = 0.0;
  for (const auto& v : values) {
    TokNode* nx = nullptr;
    total += extend(&arena, cur, v, &nx);
    cur = nx;
  }
  return total;
}
}  // namespace McBopomofo
