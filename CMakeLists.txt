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
set(WITH_OPENBLAS ON CACHE BOOL "Use OpenBLAS" FORCE)

# --- 2. FIND OPENBLAS ---
find_package(OpenBLAS CONFIG REQUIRED)
get_target_property(BLAS_INC OpenBLAS::OpenBLAS INTERFACE_INCLUDE_DIRECTORIES)
set(OPENBLAS_INCLUDE_DIR "${BLAS_INC}" CACHE PATH "Force OpenBLAS Include" FORCE)

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

# --- 6. THE APP ---
add_executable(NMT_MenKan
    src/main.cpp
    src/NMT/NMTWrapper.cpp
)

# --- THE FIX IS HERE ---
target_include_directories(NMT_MenKan PRIVATE 
    "${CMAKE_SOURCE_DIR}/src"
    "${CMAKE_SOURCE_DIR}/include"
    "${sentencepiece_SOURCE_DIR}/src" # <--- Adds path to sentencepiece_processor.h
)

# --- 7. LINK EVERYTHING ---
target_link_libraries(NMT_MenKan 
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
    
    OpenBLAS::OpenBLAS
)

if(MSVC)
    target_compile_options(NMT_MenKan PRIVATE /W3 /EHsc)
endif()

# --- 8. DLL COPY ---
add_custom_command(TARGET NMT_MenKan POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_SOURCE_DIR}/vcpkg/installed/x64-windows/bin/openblas.dll"
    $<TARGET_FILE_DIR:NMT_MenKan>
)