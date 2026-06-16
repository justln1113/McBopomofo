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

#include "LstmRescorerModel.h"

#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>

#include "UTF8Helper.h"

namespace McBopomofo {

namespace {

constexpr char kMagic[8] = {'M', 'C', 'B', 'L', 'S', 'T', 'M', '1'};
constexpr uint32_t kDtypeFloat32 = 0;
constexpr uint32_t kDtypeInt8 = 1;

inline double Sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

// A forward cursor over the raw weight-file bytes. Every read is bounds-checked
// and throws on overrun so a truncated or malformed file fails loudly at load
// rather than reading garbage during inference.
class ByteReader {
 public:
  ByteReader(const char* data, size_t size) : data_(data), size_(size) {}

  void readMagic() {
    require(sizeof(kMagic));
    if (std::memcmp(data_ + pos_, kMagic, sizeof(kMagic)) != 0) {
      throw std::runtime_error("LstmRescorerModel: bad magic (not MCBLSTM1)");
    }
    pos_ += sizeof(kMagic);
  }

  uint32_t readU32() {
    require(sizeof(uint32_t));
    uint32_t v = 0;
    std::memcpy(&v, data_ + pos_, sizeof(uint32_t));  // file is little-endian
    pos_ += sizeof(uint32_t);
    return v;
  }

  float readF32() {
    require(sizeof(float));
    float v = 0;
    std::memcpy(&v, data_ + pos_, sizeof(float));
    pos_ += sizeof(float);
    return v;
  }

  // Reads `count` float32 values into `out`.
  void readFloats(size_t count, std::vector<float>* out) {
    require(count * sizeof(float));
    out->resize(count);
    std::memcpy(out->data(), data_ + pos_, count * sizeof(float));
    pos_ += count * sizeof(float);
  }

  // Reads a symmetric-per-tensor-quantized int8 tensor: a float32 scale, then
  // `count` int8 values, dequantized as q * scale.
  void readInt8Dequant(size_t count, std::vector<float>* out) {
    float scale = readF32();
    require(count);
    out->resize(count);
    for (size_t i = 0; i < count; ++i) {
      auto q = static_cast<int8_t>(data_[pos_ + i]);
      (*out)[i] = static_cast<float>(q) * scale;
    }
    pos_ += count;
  }

 private:
  void require(size_t n) const {
    if (pos_ + n > size_) {
      throw std::runtime_error("LstmRescorerModel: weight file truncated");
    }
  }

  const char* data_;
  size_t size_;
  size_t pos_ = 0;
};

}  // namespace

std::shared_ptr<LstmRescorerModel> LstmRescorerModel::Load(
    const std::string& weightsPath, const std::string& vocabPath) {
  std::ifstream wf(weightsPath, std::ios::binary | std::ios::ate);
  if (!wf) {
    throw std::runtime_error("LstmRescorerModel: cannot open weights file: " +
                             weightsPath);
  }
  std::streamsize size = wf.tellg();
  wf.seekg(0, std::ios::beg);
  std::vector<char> buffer(static_cast<size_t>(size));
  if (!wf.read(buffer.data(), size)) {
    throw std::runtime_error("LstmRescorerModel: cannot read weights file: " +
                             weightsPath);
  }

  auto model = std::shared_ptr<LstmRescorerModel>(new LstmRescorerModel());

  ByteReader reader(buffer.data(), buffer.size());
  reader.readMagic();
  uint32_t dtype = reader.readU32();
  model->vocab_ = reader.readU32();
  model->embed_ = reader.readU32();
  model->hidden_ = reader.readU32();
  reader.readU32();  // reserved

  if (dtype != kDtypeFloat32 && dtype != kDtypeInt8) {
    throw std::runtime_error("LstmRescorerModel: unknown dtype");
  }
  if (model->vocab_ == 0 || model->embed_ == 0 || model->hidden_ == 0) {
    throw std::runtime_error("LstmRescorerModel: zero dimension in header");
  }

  const size_t V = model->vocab_;
  const size_t E = model->embed_;
  const size_t H = model->hidden_;
  // Tensor element counts, in the exact order export_weights.py writes them.
  const std::pair<std::vector<float>*, size_t> tensors[] = {
      {&model->embedWeight_, V * E},
      {&model->wIh_, 4 * H * E},
      {&model->wHh_, 4 * H * H},
      {&model->bIh_, 4 * H},
      {&model->bHh_, 4 * H},
      {&model->projWeight_, V * H},
      {&model->projBias_, V},
  };
  for (const auto& [out, count] : tensors) {
    if (dtype == kDtypeInt8) {
      reader.readInt8Dequant(count, out);
    } else {
      reader.readFloats(count, out);
    }
  }

  // Load the id-ordered vocab. Each line is one token; export_weights.py escapes
  // an embedded newline as "\\n", so undo that.
  std::ifstream vf(vocabPath);
  if (!vf) {
    throw std::runtime_error("LstmRescorerModel: cannot open vocab file: " +
                             vocabPath);
  }
  std::string line;
  while (std::getline(vf, line)) {
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();  // tolerate CRLF
    }
    std::string tok;
    for (size_t i = 0; i < line.size(); ++i) {
      if (line[i] == '\\' && i + 1 < line.size() && line[i + 1] == 'n') {
        tok.push_back('\n');
        ++i;
      } else {
        tok.push_back(line[i]);
      }
    }
    int id = static_cast<int>(model->idToToken_.size());
    model->tokenToId_[tok] = id;
    model->idToToken_.push_back(tok);
  }
  if (model->idToToken_.size() != V) {
    throw std::runtime_error(
        "LstmRescorerModel: vocab.txt size does not match weights vocab dim");
  }

  auto bos = model->tokenToId_.find("<bos>");
  auto unk = model->tokenToId_.find("<unk>");
  if (bos == model->tokenToId_.end() || unk == model->tokenToId_.end()) {
    throw std::runtime_error(
        "LstmRescorerModel: vocab missing <bos> or <unk> token");
  }
  model->bosId_ = bos->second;
  model->unkId_ = unk->second;

  // Precompute the initial state: the LM is trained on [<bos>, chars..., <eos>],
  // so the "empty prefix" state is what the LSTM holds after consuming <bos>
  // from a zero hidden/cell. After this, proj(h) predicts the first real char.
  std::vector<float> zeroH(H, 0.0f);
  std::vector<float> zeroC(H, 0.0f);
  std::vector<float> h(H);
  std::vector<float> c(H);
  model->lstmStep(model->bosId_, zeroH.data(), zeroC.data(), h.data(),
                  c.data());
  model->initialState_.hidden.resize(2 * H);
  std::memcpy(model->initialState_.hidden.data(), h.data(), H * sizeof(float));
  std::memcpy(model->initialState_.hidden.data() + H, c.data(),
              H * sizeof(float));
  model->initialState_.context.clear();

  return model;
}

RescorerModelState LstmRescorerModel::initialState() { return initialState_; }

std::vector<int> LstmRescorerModel::tokenize(const std::string& value) const {
  std::vector<int> ids;
  for (const std::string& ch : Split(value)) {
    auto it = tokenToId_.find(ch);
    ids.push_back(it == tokenToId_.end() ? unkId_ : it->second);
  }
  return ids;
}

void LstmRescorerModel::lstmStep(int tokenId, const float* hPrev,
                                 const float* cPrev, float* hOut,
                                 float* cOut) const {
  const size_t H = hidden_;
  const size_t E = embed_;
  const float* x = embedWeight_.data() + static_cast<size_t>(tokenId) * E;

  for (size_t r = 0; r < H; ++r) {
    // Gate rows are packed [i (0..H), f (H..2H), g (2H..3H), o (3H..4H)].
    double preI = static_cast<double>(bIh_[r]) + bHh_[r];
    double preF = static_cast<double>(bIh_[H + r]) + bHh_[H + r];
    double preG = static_cast<double>(bIh_[2 * H + r]) + bHh_[2 * H + r];
    double preO = static_cast<double>(bIh_[3 * H + r]) + bHh_[3 * H + r];

    const float* wiI = wIh_.data() + (r) * E;
    const float* wiF = wIh_.data() + (H + r) * E;
    const float* wiG = wIh_.data() + (2 * H + r) * E;
    const float* wiO = wIh_.data() + (3 * H + r) * E;
    for (size_t e = 0; e < E; ++e) {
      double xe = x[e];
      preI += wiI[e] * xe;
      preF += wiF[e] * xe;
      preG += wiG[e] * xe;
      preO += wiO[e] * xe;
    }

    const float* whI = wHh_.data() + (r) * H;
    const float* whF = wHh_.data() + (H + r) * H;
    const float* whG = wHh_.data() + (2 * H + r) * H;
    const float* whO = wHh_.data() + (3 * H + r) * H;
    for (size_t k = 0; k < H; ++k) {
      double hk = hPrev[k];
      preI += whI[k] * hk;
      preF += whF[k] * hk;
      preG += whG[k] * hk;
      preO += whO[k] * hk;
    }

    double i = Sigmoid(preI);
    double f = Sigmoid(preF);
    double g = std::tanh(preG);
    double o = Sigmoid(preO);
    double cNew = f * cPrev[r] + i * g;
    cOut[r] = static_cast<float>(cNew);
    hOut[r] = static_cast<float>(o * std::tanh(cNew));
  }
}

double LstmRescorerModel::logSoftmaxFull(int tokenId, const float* h) const {
  const size_t H = hidden_;
  const size_t V = vocab_;
  // Full-vocabulary log-softmax: logit(tokenId) - log sum_t exp(logit(t)).
  // We score against the WHOLE vocabulary, not a selective set of the position's
  // homophones, because the disambiguating signal usually lies in the char
  // *after* a branch (你 reads fluently after 依 but not after 一); a selective
  // softmax at the branch alone would miss it and pick the wrong candidate.
  std::vector<double> logits(V);
  double maxLogit = -std::numeric_limits<double>::infinity();
  for (size_t t = 0; t < V; ++t) {
    const float* w = projWeight_.data() + t * H;
    double l = projBias_[t];
    for (size_t k = 0; k < H; ++k) {
      l += w[k] * h[k];
    }
    logits[t] = l;
    if (l > maxLogit) {
      maxLogit = l;
    }
  }
  double sumExp = 0.0;
  for (size_t t = 0; t < V; ++t) {
    sumExp += std::exp(logits[t] - maxLogit);
  }
  double logZ = maxLogit + std::log(sumExp);
  return logits[static_cast<size_t>(tokenId)] - logZ;
}

std::pair<double, RescorerModelState> LstmRescorerModel::step(
    const RescorerModelState& prevState, const std::string& nextValue,
    const std::vector<std::string>& homophones) {
  // homophones is unused: a full-likelihood model scores every position against
  // the whole vocabulary, so the per-position term is the same regardless of the
  // branch set. The parameter stays for the RescorerModel interface (mock /
  // bigram-style models may still use a selective set).
  (void)homophones;

  const size_t H = hidden_;
  // Resolve the prior (h, c). A malformed/empty state falls back to the initial
  // state rather than reading out of bounds.
  const std::vector<float>& prevHidden =
      prevState.hidden.size() == 2 * H ? prevState.hidden
                                       : initialState_.hidden;

  std::vector<int> ids = tokenize(nextValue);
  if (ids.empty()) {
    return {0.0, prevState};
  }

  std::vector<float> h(prevHidden.begin(), prevHidden.begin() + H);
  std::vector<float> c(prevHidden.begin() + H, prevHidden.begin() + 2 * H);
  std::vector<float> hNext(H);
  std::vector<float> cNext(H);

  // Sum the full-softmax log-probability of each char given its prefix: the
  // candidate value's contribution to the sentence log-likelihood.
  double logProb = 0.0;
  for (int id : ids) {
    logProb += logSoftmaxFull(id, h.data());
    lstmStep(id, h.data(), c.data(), hNext.data(), cNext.data());
    h.swap(hNext);
    c.swap(cNext);
  }

  RescorerModelState next;
  next.hidden.resize(2 * H);
  std::memcpy(next.hidden.data(), h.data(), H * sizeof(float));
  std::memcpy(next.hidden.data() + H, c.data(), H * sizeof(float));
  return {logProb, std::move(next)};
}

}  // namespace McBopomofo
