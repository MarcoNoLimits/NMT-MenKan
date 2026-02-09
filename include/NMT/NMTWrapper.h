#pragma once

#include <string>
#include <memory>
#include <vector>

namespace NMT {

    class NMTWrapper {
    public:
        // Constructor: Initializes the Engine with paths to .bin and .model
        NMTWrapper(const std::string& model_path, const std::string& tokenizer_path);
        
        // Destructor: Cleans up the Pimpl pointer
        ~NMTWrapper();

        // Main translation function
        std::string translate(const std::string& english_text);

    private:
        // --- PIMPL IDIOM ---
        // This hides the heavy CTranslate2 headers from the rest of your app.
        // The actual logic lives inside the .cpp file.
        struct Impl; 
        std::unique_ptr<Impl> impl;
    };

}