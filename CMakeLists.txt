cmake_minimum_required(VERSION 3.14)
project(NMT_MenKan)

# --- 1. CONFIGURATION ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(OPENMP_RUNTIME "COMP" CACHE STRING "Force MSVC OpenMP" FORCE)

# Optimization Flags
set(BUILD_CLI OFF CACHE BOOL "" FORCE)
set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(WITH_MKL OFF CACHE BOOL "Disable MKL" FORCE) 
set(WITH_DNNL OFF CACHE BOOL "Disable DNNL" FORCE)

# --- 2. SELECT COMPUTE BACKEND ---
if(CMAKE_GENERATOR_PLATFORM STREQUAL "ARM64")
    # HoloLens / ARM builds must use Google's Ruy processor for Int8 support
    set(WITH_RUY ON CACHE BOOL "Use Ruy" FORCE)
    set(WITH_OPENBLAS OFF CACHE BOOL "Disable OpenBLAS" FORCE)
else()
    # PC x64: OpenBLAS is only wired for float32 GEMM in CTranslate2; int8_float32 needs Ruy
    # (or MKL/DNNL). Without Ruy, int8 models fall back to float32 — same warning you see and
    # ~10x slower inference. Python wheels ship with Ruy/MKL; match that here.
    set(WITH_RUY ON CACHE BOOL "Use Ruy for int8 GEMM" FORCE)
    set(WITH_OPENBLAS ON CACHE BOOL "Use OpenBLAS for float32 GEMM" FORCE)
    find_package(OpenBLAS CONFIG REQUIRED)
    get_target_property(BLAS_INC OpenBLAS::OpenBLAS INTERFACE_INCLUDE_DIRECTORIES)
    set(OPENBLAS_INCLUDE_DIR "${BLAS_INC}" CACHE PATH "Force OpenBLAS Include" FORCE)
endif()

# --- 3. FETCH ABSEIL (Build from Source) ---
include(FetchContent)
set(ABSL_PROPAGATE_CXX_STD ON)

FetchContent_Declare(
  abseil-cpp
  GIT_REPOSITORY https://github.com/abseil/abseil-cpp.git
  GIT_TAG 20240116.1 
)
FetchContent_MakeAvailable(abseil-cpp)

# --- 4. FETCH SENTENCEPIECE (Build from Source) ---
set(SPM_ENABLE_SHARED OFF CACHE BOOL "" FORCE)
set(SPM_ENABLE_TCMALLOC OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
  sentencepiece
  GIT_REPOSITORY https://github.com/google/sentencepiece.git
  GIT_TAG v0.2.0
)
FetchContent_MakeAvailable(sentencepiece)

# --- 5. FETCH CTRANSLATE2 (Build from Source) ---
FetchContent_Declare(
  ctranslate2
  GIT_REPOSITORY https://github.com/OpenNMT/CTranslate2.git
  GIT_TAG v4.7.1
)
FetchContent_MakeAvailable(ctranslate2)

# --- 6. THE APP / NATIVE PLUGIN ---
add_library(NMT_MenKan_Plugin SHARED
    src/UnityInterface.cpp
    src/NMT/NMTWrapper.cpp
)

# --- THE FIX IS HERE ---
target_include_directories(NMT_MenKan_Plugin PRIVATE 
    "${CMAKE_SOURCE_DIR}/src"
    "${CMAKE_SOURCE_DIR}/include"
    "${CMAKE_SOURCE_DIR}/include/NMT"
    "${sentencepiece_SOURCE_DIR}/src" # <--- Adds path to sentencepiece_processor.h
)

# --- 7. LINK EVERYTHING ---
target_link_libraries(NMT_MenKan_Plugin 
    PRIVATE 
    ctranslate2
    sentencepiece-static
    
    # Abseil Deps
    absl::base
    absl::strings
    absl::synchronization
    absl::hash
    absl::flat_hash_map
    absl::flags
    absl::flags_parse
    Ws2_32
)

if(NOT CMAKE_GENERATOR_PLATFORM STREQUAL "ARM64")
    target_link_libraries(NMT_MenKan_Plugin PRIVATE OpenBLAS::OpenBLAS)
endif()

if(MSVC)
    target_compile_options(NMT_MenKan_Plugin PRIVATE /W3 /EHsc /wd4566)
endif()

# --- TCP test server (Windows PC): same NMT stack as evaluate_nmt_fast.py threading ---
add_executable(NMT_TCP_Server
    src/main.cpp
    src/NMT/NMTWrapper.cpp
)
target_include_directories(NMT_TCP_Server PRIVATE
    "${CMAKE_SOURCE_DIR}/src"
    "${CMAKE_SOURCE_DIR}/include"
    "${CMAKE_SOURCE_DIR}/include/NMT"
    "${sentencepiece_SOURCE_DIR}/src"
)
target_link_libraries(NMT_TCP_Server PRIVATE
    ctranslate2
    sentencepiece-static
    absl::base
    absl::strings
    absl::synchronization
    absl::hash
    absl::flat_hash_map
    absl::flags
    absl::flags_parse
    Ws2_32
)
if(NOT CMAKE_GENERATOR_PLATFORM STREQUAL "ARM64")
    target_link_libraries(NMT_TCP_Server PRIVATE OpenBLAS::OpenBLAS)
endif()
if(MSVC)
    # C4566: unicode in third-party ctranslate2 headers (code page 1252)
    target_compile_options(NMT_TCP_Server PRIVATE /W3 /EHsc /wd4566)
endif()

# Friendly name for docs / scripts (same binary as NMT_TCP_Server target)
set_target_properties(NMT_TCP_Server PROPERTIES OUTPUT_NAME NMT_MenKan)

# --- 8. DLL COPY ---
if(NOT CMAKE_GENERATOR_PLATFORM STREQUAL "ARM64")
    add_custom_command(TARGET NMT_MenKan_Plugin POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_SOURCE_DIR}/vcpkg/installed/x64-windows/bin/openblas.dll"
        $<TARGET_FILE_DIR:NMT_MenKan_Plugin>
    )
    add_custom_command(TARGET NMT_TCP_Server POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_SOURCE_DIR}/vcpkg/installed/x64-windows/bin/openblas.dll"
        $<TARGET_FILE_DIR:NMT_TCP_Server>
    )
endif()