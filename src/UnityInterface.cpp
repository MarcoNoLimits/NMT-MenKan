#include "NMTWrapper.h"
#include <memory>
#include <string>

// Global instance to keep the model loaded in memory for Unity
std::unique_ptr<NMT::NMTWrapper> g_translator;

// Helper to store the last translated string in memory so Unity can read it
std::string g_last_translation;

extern "C" {

// 1. Initialize the CTranslate2 Model
// Unity will pass the path to the NLLB folder (e.g.,
// Application.streamingAssetsPath + "/nllb_int8")
__declspec(dllexport) void InitModel(const char *model_path) {
  if (!g_translator) {
    std::string path(model_path);
    std::string tokenizer_path = path + "/sentencepiece.bpe.model";
    g_translator = std::make_unique<NMT::NMTWrapper>(path, tokenizer_path);
  }
}

// 2. Translate Text
// Unity passes English text, and receives a pointer to the translated Italian
// string
__declspec(dllexport) const char *TranslateText(const char *input_text) {
  if (!g_translator) {
    return "Error: Model not initialized.";
  }

  std::string input(input_text);

  try {
    g_last_translation = g_translator->translate(input);
    return g_last_translation.c_str();
  } catch (const std::exception &e) {
    g_last_translation = std::string("Error during translation: ") + e.what();
    return g_last_translation.c_str();
  }
}

// 3. Free resources when the Unity App closes
__declspec(dllexport) void CleanupModel() { g_translator.reset(); }
}
