# NMT-MenKan: Core NMT Engine Documentation
**Version:** 1.0 (Architecture Proof-of-Concept)
**Target:** Windows x64 (Laptop Demo) -> Path to HoloLens (ARM64)
**Engine:** CTranslate2 (Inference) + SentencePiece (Tokenization) + OpenBLAS (Math)

---

## 1. Project Overview
This project implements a standalone Offline Neural Machine Translation (NMT) engine using C++. It is designed to run the **NLLB-200** model to translate English speech to Turkish text for the Yeelen-Link accessibility glasses.

### Key Architecture Decisions
* **No Python Dependency:** The engine is pure C++, allowing it to run on embedded/restricted systems (like HoloLens UWP) where Python is not supported.
* **Static Linking:** We compile dependencies (Abseil, SentencePiece) from source to strictly control the ABI and prevent "DLL Hell."
* **CPU Optimization:** Configured for quantization (Int8) to run fast on consumer hardware without NVIDIA CUDA.

---

## 2. Prerequisites
Before building, ensure the following are installed:
1.  **Visual Studio 2022** (Community Edition) with "Desktop development with C++".
2.  **CMake** (Added to system PATH).
3.  **Git** (Added to system PATH).
4.  **PowerShell** (Terminal).

---

## 3. Directory Structure
Your project folder (`C:\GitHub\NMT-MenKan`) must look like this for the build scripts to work:

```text
NMT-MenKan/
├── build/                  # (Created automatically during build)
├── include/
│   └── NMT/
│       └── NMTWrapper.h    # The Header File
├── libs/                   # (Managed by CMake FetchContent, do not touch)
├── src/
│   ├── main.cpp            # The Entry Point / Test Runner
│   └── NMT/
│       └── NMTWrapper.cpp  # The Implementation Logic
├── vcpkg/                  # Package Manager
├── nllb_int8/              # (Your Model Folder)
│   └── model.bin
├── sentencepiece.bpe.model # (Your Tokenizer Model)
└── CMakeLists.txt          # The Build Configuration

---

## 4. The "Golden" Configuration Files
### A. CMakeLists.txt (The Build Script)
This is the critical file that fixes the "Unresolved External Symbol" errors by building Abseil and SentencePiece from source and linking them statically.

```cmake
CMake
cmake_minimum_required(VERSION 3.14)
project(NMT_MenKan)

# --- 1. CONFIGURATION ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Force MSVC's native OpenMP (ARM64 safe for future HoloLens port)
set(OPENMP_RUNTIME "COMP" CACHE STRING "Force MSVC OpenMP" FORCE)

# Optimization Flags
set(BUILD_CLI OFF CACHE BOOL "" FORCE)
set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(WITH_MKL OFF CACHE BOOL "Disable MKL (Intel Only)" FORCE) 
set(WITH_DNNL OFF CACHE BOOL "Disable DNNL (Intel Only)" FORCE)
set(WITH_OPENBLAS ON CACHE BOOL "Use OpenBLAS" FORCE)

# --- 2. FIND OPENBLAS (From vcpkg) ---
find_package(OpenBLAS CONFIG REQUIRED)
get_target_property(BLAS_INC OpenBLAS::OpenBLAS INTERFACE_INCLUDE_DIRECTORIES)
set(OPENBLAS_INCLUDE_DIR "${BLAS_INC}" CACHE PATH "Force OpenBLAS Include" FORCE)

# --- 3. FETCH ABSEIL (Build Source) ---
# Solves ABI mismatches by building Absl with the same compiler settings as the app.
include(FetchContent)
set(ABSL_PROPAGATE_CXX_STD ON)
FetchContent_Declare(
  abseil-cpp
  GIT_REPOSITORY [https://github.com/abseil/abseil-cpp.git](https://github.com/abseil/abseil-cpp.git)
  GIT_TAG 20240116.1 
)
FetchContent_MakeAvailable(abseil-cpp)

# --- 4. FETCH SENTENCEPIECE (Build Source) ---
# Builds static libraries to avoid DLL missing errors.
set(SPM_ENABLE_SHARED OFF CACHE BOOL "" FORCE)
set(SPM_ENABLE_TCMALLOC OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  sentencepiece
  GIT_REPOSITORY [https://github.com/google/sentencepiece.git](https://github.com/google/sentencepiece.git)
  GIT_TAG v0.2.0
)
FetchContent_MakeAvailable(sentencepiece)

# --- 5. FETCH CTRANSLATE2 (Build Source) ---
FetchContent_Declare(
  ctranslate2
  GIT_REPOSITORY [https://github.com/OpenNMT/CTranslate2.git](https://github.com/OpenNMT/CTranslate2.git)
  GIT_TAG v4.7.1
)
FetchContent_MakeAvailable(ctranslate2)

# --- 6. THE EXECUTABLE ---
add_executable(NMT_MenKan
    src/main.cpp
    src/NMT/NMTWrapper.cpp
)

# --- 7. INCLUDE PATHS ---
target_include_directories(NMT_MenKan PRIVATE 
    "${CMAKE_SOURCE_DIR}/src"
    "${CMAKE_SOURCE_DIR}/include"
    "${sentencepiece_SOURCE_DIR}/src" # Fixes "sentencepiece_processor.h not found"
)

# --- 8. LINKING (The Fix) ---
target_link_libraries(NMT_MenKan 
    PRIVATE 
    ctranslate2
    sentencepiece-static
    
    # Abseil Dependencies (Explicitly Linked)
    absl::base
    absl::strings
    absl::synchronization   # Fixes Mutex/SpinLock errors
    absl::hash              # Fixes MixingHashState errors
    absl::flat_hash_map     # Fixes Container errors
    absl::flags
    absl::flags_parse
    
    OpenBLAS::OpenBLAS
)

# Standard MSVC warning suppression
if(MSVC)
    target_compile_options(NMT_MenKan PRIVATE /W3 /EHsc)
endif()

# --- 9. POST-BUILD: COPY DLLs ---
# Automatically copies OpenBLAS.dll to the output folder.
add_custom_command(TARGET NMT_MenKan POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_SOURCE_DIR}/vcpkg/installed/x64-windows/bin/openblas.dll"
    $<TARGET_FILE_DIR:NMT_MenKan>
)
B. src/NMT/NMTWrapper.h (The Interface)
Uses the Pimpl Idiom to hide the heavy CTranslate2 headers from the rest of your application, speeding up compilation.

C++
#pragma once
#include <string>
#include <memory>
#include <vector>

namespace NMT {
    class NMTWrapper {
    public:
        NMTWrapper(const std::string& model_path, const std::string& tokenizer_path);
        ~NMTWrapper();
        std::string translate(const std::string& english_text);

    private:
        struct Impl; 
        std::unique_ptr<Impl> impl;
    };
}
C. src/NMT/NMTWrapper.cpp (The Logic)
Handles loading the model, tokenizing input, running inference, and cleaning up the output (removing tur_Latn tags).

C++
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
            // Load Engine (CPU Mode)
            translator = std::make_unique<ctranslate2::Translator>(
                model_path, ctranslate2::Device::CPU
            );
            // Load Tokenizer
            sp_processor = std::make_unique<sentencepiece::SentencePieceProcessor>();
            sp_processor->Load(tokenizer_path);
        }

        std::string translate(const std::string& english_text) {
            if (!translator || !sp_processor) return "Error: Engine not initialized";

            // 1. Tokenize
            std::vector<std::string> tokens;
            sp_processor->Encode(english_text, &tokens);
            std::vector<std::vector<std::string>> batch_input = { tokens };

            // 2. Setup Options
            ctranslate2::TranslationOptions options;
            options.beam_size = 2; // Fast speed
            std::vector<std::string> target_prefix = { "tur_Latn" }; // Target Language
            std::vector<std::vector<std::string>> batch_target_prefix = { target_prefix };

            // 3. Translate
            ctranslate2::TranslationResult result = translator->translate_batch(
                batch_input, batch_target_prefix, options
            )[0];

            // 4. Detokenize
            std::string turkish_text;
            sp_processor->Decode(result.output(), &turkish_text);

            // 5. Cleanup Tag (Remove 'tur_Latn')
            std::string tag = "tur_Latn ";
            if (turkish_text.rfind(tag, 0) == 0) {
                turkish_text.erase(0, tag.length());
            }
            return turkish_text;
        }
    };

    NMTWrapper::NMTWrapper(const std::string& m, const std::string& t) : impl(std::make_unique<Impl>(m, t)) {}
    NMTWrapper::~NMTWrapper() = default;
    std::string NMTWrapper::translate(const std::string& text) { return impl->translate(text); }
}
D. src/main.cpp (The Runner)
Includes critical fixes for UTF-8 encoding (so Turkish characters show up) and Threading (prevents OpenBLAS memory crash).

C++
#include <iostream>
#include <cstdlib>
#include "NMT/NMTWrapper.h"

#ifdef _WIN32
#include <windows.h>
#endif

int main() {
    // CRITICAL FIX: Prevent OpenBLAS from conflicting with CTranslate2 threads
    _putenv("OPENBLAS_NUM_THREADS=1");

    // CRITICAL FIX: Force Windows Console to UTF-8
    #ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    #endif

    std::string model_path = "nllb_int8"; 
    std::string sp_model_path = "sentencepiece.bpe.model";

    std::cout << "🚀 Initializing Yeelen-Link Engine..." << std::endl;
    
    try {
        NMT::NMTWrapper engine(model_path, sp_model_path);
        std::string input = "Hello, this is a test for the HoloLens project.";
        std::cout << "🇬🇧 In: " << input << std::endl;
        std::string output = engine.translate(input);
        std::cout << "🇹🇷 Out: " << output << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "🔥 Crash: " << e.what() << std::endl;
        return 1;
    }
    std::cout << "\n(Press Enter to exit)";
    std::cin.get();
    return 0;
}
5. How to Build & Run (The Recipe)
Open PowerShell in the NMT-MenKan folder and run these commands in order.

Step 1: Clean & Configure
PowerShell
# Delete old builds to avoid cache corruption
Remove-Item -Path build -Recurse -Force -ErrorAction SilentlyContinue

# Generate VS Solution (Point to your vcpkg toolchain)
cmake -B build -DCMAKE_TOOLCHAIN_FILE=C:/GitHub/NMT-MenKan/vcpkg/scripts/buildsystems/vcpkg.cmake
Step 2: Compile (The Long Wait)
This compiles CTranslate2, Abseil, SentencePiece, and your App. (Takes 5-10 mins first time).

PowerShell
cmake --build build --config Release
Step 3: Setup Runtime (Critical)
You must move your model files next to the new executable.

Navigate to C:\GitHub\NMT-MenKan\build\Release\

Copy ctranslate2.dll from build\_deps\ctranslate2-build\Release\ to build\Release\.

Copy your nllb_int8 folder here.

Copy sentencepiece.bpe.model here.

Final Folder Checklist:

[x] NMT_MenKan.exe

[x] openblas.dll

[x] ctranslate2.dll

[x] sentencepiece.bpe.model

[x] nllb_int8/ (Folder)

Step 4: Run
PowerShell
cd build/Release
.\NMT_MenKan.exe
6. Future Roadmap
Audio Integration: The next step is feeding the translate() function with text from a Speech-to-Text (STT) engine instead of a hardcoded string.

HoloLens Porting: To run on the glasses, we will change the compiler target from x64 (Laptop) to ARM64 (HoloLens) in CMake. The code we wrote is already compatible (no Intel-specific dependencies).