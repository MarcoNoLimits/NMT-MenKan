#include "NMT/NMTWrapper.h"
#include <ctranslate2/translator.h>
#include <sentencepiece_processor.h>
#include <iostream>

using namespace std;

namespace NMT {

    struct NMTWrapper::Impl {
        std::unique_ptr<ctranslate2::Translator> translator;
        std::unique_ptr<sentencepiece::SentencePieceProcessor> sp_processor;

        Impl(const std::string& model_path, const std::string& tokenizer_path) {
            // 1. Load the Translator (NLLB Int8)
            // We use CPU because we are on HoloLens/Laptop without CUDA
            translator = std::make_unique<ctranslate2::Translator>(
                model_path, 
                ctranslate2::Device::CPU
            );

            // 2. Load the Tokenizer (SentencePiece)
            sp_processor = std::make_unique<sentencepiece::SentencePieceProcessor>();
            const auto status = sp_processor->Load(tokenizer_path);
            if (!status.ok()) {
                std::cerr << "❌ Failed to load SentencePiece model: " << status.ToString() << std::endl;
            }
        }

        std::string translate(const std::string& english_text) {
            if (!translator || !sp_processor) return "Error: Engine not initialized";

            // --- A. TOKENIZE ---
            std::vector<std::string> tokens;
            sp_processor->Encode(english_text, &tokens);

            // NLLB Specific: We might need to handle source lang tags here depending on training
            // For standard NLLB, we usually just input tokens.
            // But strict NLLB often wants: "sentence </s> eng_Latn"
            // For now, let's try raw tokens. If accuracy is low, we append "eng_Latn".

            std::vector<std::vector<std::string>> batch_input = { tokens };

            // --- B. SET OPTIONS ---
            ctranslate2::TranslationOptions options;
            options.beam_size = 2; // Low beam size for speed on HoloLens
            
            // --- C. DEFINE TARGET PREFIX (The Fix!) ---
            // Instead of options.target_prefix, we pass it as an argument.
            // "tur_Latn" tells NLLB to translate to Turkish.
            std::vector<std::string> target_prefix = { "tur_Latn" };
            std::vector<std::vector<std::string>> batch_target_prefix = { target_prefix };

            // --- D. TRANSLATE ---
            // Signature: translate_batch(source, target_prefix, options)
            ctranslate2::TranslationResult result = translator->translate_batch(
                batch_input, 
                batch_target_prefix, 
                options
            )[0];

            // --- E. DETOKENIZE ---
            std::string turkish_text;
            sp_processor->Decode(result.output(), &turkish_text);

            // --- F. CLEANUP (Remove the 'tur_Latn' tag) ---
            // NLLB outputs "tur_Latn Translation...", so we remove the first 9 chars
            // if the string starts with the tag.
            std::string tag = "tur_Latn ";
            if (turkish_text.rfind(tag, 0) == 0) { // Check if starts with tag
                turkish_text.erase(0, tag.length());
            }

            return turkish_text;
        }
    };

    // --- PIMPL BOILERPLATE ---
    NMTWrapper::NMTWrapper(const std::string& model_path, const std::string& tokenizer_path)
        : impl(std::make_unique<Impl>(model_path, tokenizer_path)) {}

    NMTWrapper::~NMTWrapper() = default;

    std::string NMTWrapper::translate(const std::string& text) {
        return impl->translate(text);
    }
}