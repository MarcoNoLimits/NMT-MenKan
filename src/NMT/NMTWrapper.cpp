#include "NMTWrapper.h"
#include <ctranslate2/translator.h>
#include <iostream>
#include <sentencepiece_processor.h>

using namespace std;

namespace NMT {

struct NMTWrapper::Impl {
  unique_ptr<ctranslate2::Translator> translator;
  unique_ptr<sentencepiece::SentencePieceProcessor> sp_processor;

  Impl(const std::string &model_path, const std::string &tokenizer_path) {
    // 1. Load the Translator (NLLB Int8)
    translator = make_unique<ctranslate2::Translator>(model_path,
                                                      ctranslate2::Device::CPU);

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

    // --- A. TOKENIZE ---
    vector<string> tokens;
    sp_processor->Encode(english_text, &tokens);

    // --- THE FIX: ADD SOURCE LANGUAGE TAGS ---
    // NLLB *requires* the input to end with "</s>" and the source language
    // code. If we miss this, the model hallucinates (Holo Holo Holo...)
    tokens.push_back("</s>");
    tokens.push_back("eng_Latn");

    vector<vector<string>> batch_input = {tokens};

    // --- B. SET OPTIONS ---
    ctranslate2::TranslationOptions options;

    // SPEED HACK: Change 2 to 1 (Greedy Search)
    // Beam search (2) is slightly better quality but 2x slower.
    // Greedy (1) is standard for real-time subtitles.
    options.beam_size = 1;

    // --- C. DEFINE TARGET PREFIX ---
    // This tells the model: "Start generating in Italian"
    vector<string> target_prefix = {"ita_Latn"};
    vector<vector<string>> batch_target_prefix = {target_prefix};

    // --- D. TRANSLATE ---
    ctranslate2::TranslationResult result = translator->translate_batch(
        batch_input, batch_target_prefix, options)[0];

    // --- E. DETOKENIZE ---
    string italian_text;
    sp_processor->Decode(result.output(), &italian_text);

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