#include <iostream>
#include <cstdlib> // For _putenv
#include "NMT/NMTWrapper.h"

#ifdef _WIN32
#include <windows.h>
#endif

int main() {
    // --- 1. FIX THE CRASH (BLAS Memory Error) ---
    // Force OpenBLAS to use 1 thread. CTranslate2 handles the parallelism.
    // If we let OpenBLAS spawn threads too, it conflicts on Windows and crashes.
    _putenv("OPENBLAS_NUM_THREADS=1");

    // --- 2. FIX THE WEIRD TEXT (Encoding) ---
    // Force Windows Console to accept UTF-8 (so Emojis and Turkish chars work)
    #ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    #endif

    // NOTE: Ensure these paths are correct relative to where you run the .exe
    std::string model_path = "nllb_int8"; 
    std::string sp_model_path = "sentencepiece.bpe.model";

    std::cout << "🚀 Initializing NMT-MenKan Engine..." << std::endl;
    
    try {
        NMT::NMTWrapper engine(model_path, sp_model_path);
        
        std::string input = "Hello, this is a test for the HoloLens project.";
        std::cout << "en In: " << input << std::endl;

        std::string output = engine.translate(input);
        std::cout << "tr Out: " << output << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "🔥 Crash: " << e.what() << std::endl;
        return 1;
    }

    // Keep terminal open if running from double-click
    std::cout << "\n(Press Enter to exit)";
    std::cin.get();

    return 0;
}