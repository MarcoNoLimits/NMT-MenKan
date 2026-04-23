#include "NMTWrapper.h"
#include <chrono>
#include <ctranslate2/models/model.h>
#include <ctranslate2/translator.h>
#include <iostream>
#include <sentencepiece_processor.h>

using namespace std;

namespace NMT {

namespace {

// Match scripts/evaluate_nmt_fast.py: Translator(..., inter_threads=8, intra_threads=0)
// — C++ maps intra_threads -> ReplicaPoolConfig::num_threads_per_replica (0 = auto, see
//   ctranslate2::set_num_threads). inter_threads maps to parallel replicas; a single
//   sequential TCP worker benefits most from intra (multi-core matmul), not multiple replicas.
#if defined(_M_ARM64) || defined(__aarch64__)
constexpr size_t kReplicasPerDevice = 1;
constexpr size_t kThreadsPerReplica = 1; // HoloLens / ARM: stable single-threaded compute
#elif defined(_WIN32)
// OPENBLAS_NUM_THREADS=1 in main.cpp caps BLAS; CT2 can still use OMP_NUM_THREADS for ops.
// kThreadsPerReplica=0 matches evaluate_nmt_fast.py intra_threads=0 (auto / OMP env).
constexpr size_t kReplicasPerDevice = 1;
constexpr size_t kThreadsPerReplica = 0;
#else
constexpr size_t kReplicasPerDevice = 1;
constexpr size_t kThreadsPerReplica =
    0; // Linux/macOS: 0 = intra_threads=0 (auto)
#endif

} // namespace

struct NMTWrapper::Impl {
  unique_ptr<ctranslate2::Translator> translator;
  unique_ptr<sentencepiece::SentencePieceProcessor> sp_processor;

  Impl(const std::string &model_path, const std::string &tokenizer_path) {
    ctranslate2::models::ModelLoader loader(model_path);
    loader.device = ctranslate2::Device::CPU;
    loader.num_replicas_per_device = kReplicasPerDevice;

    ctranslate2::ReplicaPoolConfig pool_config;
    pool_config.num_threads_per_replica = kThreadsPerReplica;

    translator =
        make_unique<ctranslate2::Translator>(loader, pool_config);

    // 2. Load the Tokenizer (SentencePiece)
    sp_processor = make_unique<sentencepiece::SentencePieceProcessor>();
    const auto status = sp_processor->Load(tokenizer_path);
    if (!status.ok()) {
      cerr << "❌ Failed to load SentencePiece model: " << status.ToString()
           << endl;
    }
  }

  string translate(const string &english_text) {
    if (!translator || !sp_processor)
      return "Error: Engine not initialized";

    using clock = std::chrono::steady_clock;
    auto t0 = clock::now();

    // --- A. TOKENIZE ---
    vector<string> tokens;
    sp_processor->Encode(english_text, &tokens);

    // --- THE FIX: ADD SOURCE LANGUAGE TAGS ---
    // NLLB *requires* the input to end with "</s>" and the source language
    // code. If we miss this, the model hallucinates (Holo Holo Holo...)
    tokens.push_back("</s>");
    tokens.push_back("eng_Latn");

    vector<vector<string>> batch_input = {tokens};

    auto t1 = clock::now();

    // --- B. SET OPTIONS (match evaluate_nmt_fast.py: beam_size=1, max_decoding_length=256) ---
    ctranslate2::TranslationOptions options;
    options.beam_size = 1;
    options.max_decoding_length = 256;

    // --- C. DEFINE TARGET PREFIX ---
    // This tells the model: "Start generating in Italian"
    vector<string> target_prefix = {"ita_Latn"};
    vector<vector<string>> batch_target_prefix = {target_prefix};

    // --- D. TRANSLATE ---
    ctranslate2::TranslationResult result = translator->translate_batch(
        batch_input, batch_target_prefix, options)[0];

    auto t2 = clock::now();

    // --- E. DETOKENIZE ---
    string italian_text;
    sp_processor->Decode(result.output(), &italian_text);

    auto t3 = clock::now();

    auto ms = [](clock::time_point a, clock::time_point b) {
      return std::chrono::duration_cast<std::chrono::milliseconds>(b - a).count();
    };
    cout << "⏱️ NMT: " << ms(t0, t3) << " ms total — tokenize " << ms(t0, t1)
         << " ms | CTranslate2 " << ms(t1, t2) << " ms | detokenize " << ms(t2, t3)
         << " ms\n";

    // --- F. CLEANUP (Remove the 'ita_Latn' tag) ---
    string tag = "ita_Latn";
    // Check if text starts with tag (handling potential spaces)
    if (italian_text.find(tag) == 0) {
      // Remove tag + any following space
      size_t remove_len = tag.length();
      if (italian_text.length() > remove_len &&
          italian_text[remove_len] == ' ') {
        remove_len++;
      }
      italian_text.erase(0, remove_len);
    }

    return italian_text;
  }
};

// --- PIMPL BOILERPLATE ---
NMTWrapper::NMTWrapper(const string &model_path, const string &tokenizer_path)
    : impl(make_unique<Impl>(model_path, tokenizer_path)) {}

NMTWrapper::~NMTWrapper() = default;

string NMTWrapper::translate(const string &text) {
  return impl->translate(text);
}
} // namespace NMT