#include <consoleapi2.h>
#include <iostream>
#include <chrono> // For timing
#include <winnls.h>
#include "NMT/NMTWrapper.h"

int main() {
    //_putenv("OPENBLAS_NUM_THREADS=1");

    // Force UTF-8 for Turkish characters
    #ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    #endif
    
    // Start Timer
    auto start = std::chrono::high_resolution_clock::now();

    std::cout << "⏳ Loading Model..." << std::endl;
    NMT::NMTWrapper engine("nllb_int8", "sentencepiece.bpe.model");
    
    // Stop Timer (Load Time)
    auto load_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> load_duration = load_end - start;
    std::cout << "✅ Model Loaded in: " << load_duration.count() << " seconds." << std::endl;

    // Translate
    std::string input = "Hello, this is a test for the hololens project.";
    auto trans_start = std::chrono::high_resolution_clock::now();
    std::string output = engine.translate(input);
    auto trans_end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> trans_duration = trans_end - trans_start;
    std::cout << "⚡ Translation took: " << trans_duration.count() << " seconds." << std::endl;
    std::cout << "🇹🇷 Output: " << output << std::endl;

    std::cin.get();
    return 0;
}