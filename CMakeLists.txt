cmake_minimum_required(VERSION 3.18)

# Must set CUDA architectures before project() to avoid CMake using default sm_52
# Support modern GPUs (CUDA 13.0+):
# Ampere (80, 86), Ada Lovelace (89), Hopper (90), Blackwell (100, 120)
set(CMAKE_CUDA_ARCHITECTURES "80;86;89;90;100;120")

# For CUDA 13.0+, tell CMake to use a supported architecture for compiler identification
# Also use gcc-12 as host compiler for better compatibility with glibc
set(CMAKE_CUDA_FLAGS_INIT "-arch=sm_80 -ccbin=/usr/bin/g++-12")
set(CMAKE_CUDA_HOST_COMPILER "/usr/bin/g++-12")
# Use matching C++ compiler to avoid ABI issues
set(CMAKE_CXX_COMPILER "/usr/bin/g++-12")
set(CMAKE_C_COMPILER "/usr/bin/gcc-12")

project(xnt-miner LANGUAGES CXX CUDA)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Find nlohmann_json (installed via apt: libnlohmann-json-dev or vcpkg on Windows)
find_package(nlohmann_json QUIET)
if(NOT nlohmann_json_FOUND)
    message(STATUS "nlohmann_json not found via CMake, using local single-header include")
    # Use local single-header version if available
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/include/nlohmann/json.hpp")
        message(STATUS "Using local nlohmann/json.hpp from include/nlohmann/")
    else()
        message(STATUS "nlohmann_json not found, using system include path")
    endif()
endif()

# CUDA architecture detection
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # Support modern GPUs (CUDA 13.0+):
    # Ampere (80, 86), Ada Lovelace (89), Hopper (90), Blackwell (100, 120)
    # Note: Pascal (60), Volta (70), Turing (75) dropped in CUDA 13.0
    set(CMAKE_CUDA_ARCHITECTURES 80 86 89 90 100 120)
endif()

# Compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")
elseif(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2")
endif()

# CUDA flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")
# Allow __uint128_t in device code (NVCC supports it even with MSVC host compiler)
# Use NVCC flags to bypass MSVC preprocessor issues
if(MSVC)
    # Suppress MSVC warnings about unknown types in device code
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler /wd4324")
    # Allow extended device code features (enables __uint128_t support)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
    # Allow relaxed constexpr (helps with device code compilation)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
    # Use NVCC's device code compiler which natively supports __uint128_t
    # This tells NVCC to use its own type handling for device code
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
    # Define macro to help NVCC recognize __uint128_t in device code
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -D__ALLOW_DEVICE_UINT128__")
endif()

# Enable separable compilation for CUDA
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -lineinfo")
endif()

# Include directories
set(INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Find OpenSSL for SSL/TLS support
find_package(OpenSSL REQUIRED)

# Link libraries
set(LINK_LIBS
    CUDA::cudart
)

# Platform-specific libraries
if(WIN32)
    list(APPEND LINK_LIBS ws2_32)
else()
    list(APPEND LINK_LIBS pthread)
endif()

# Add OpenSSL libraries
list(APPEND LINK_LIBS OpenSSL::SSL OpenSSL::Crypto)

if(nlohmann_json_FOUND)
    list(APPEND LINK_LIBS nlohmann_json::nlohmann_json)
endif()

# ===== SOURCE FILES =====
set(HEADERS
    include/common.cuh
    include/tip5.cuh
    include/digest.cuh
    include/merkle.cuh
    include/pow.cuh
    include/kernels.cuh
    include/mining_client.h
    include/stratum_client.cuh
    include/network.cuh
    include/gpu_resources.cuh
    include/mining.cuh
    include/connection_multiplexer.cuh
)

set(SOURCES
    src/common.cu
    src/tip5.cu
    src/digest.cu
    src/merkle.cu
    src/pow.cu
    src/kernels.cu
    src/network.cu
    src/rpc_client.cu
    src/stratum_client.cu
    src/gpu_resources.cu
    src/connection_multiplexer.cu
    src/mining.cu
    src/main.cu
)

# ===== Build executable =====
# Check if all source files exist
set(FILES_EXIST TRUE)
foreach(src ${SOURCES})
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${src}")
        set(FILES_EXIST FALSE)
        message(STATUS "Source file missing: ${src}")
    endif()
endforeach()

if(FILES_EXIST)
    add_executable(xnt-miner ${SOURCES})

    target_include_directories(xnt-miner PRIVATE ${INCLUDE_DIRS})
    target_link_libraries(xnt-miner PRIVATE ${LINK_LIBS})

    # Set CUDA properties for separable compilation
    set_target_properties(xnt-miner PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )

    message(STATUS "Building xnt-miner from src/*.cu")
else()
    message(FATAL_ERROR "Not all source files exist. Please ensure all files in src/ are present.")
endif()

# ===== Installation =====
if(FILES_EXIST)
    install(TARGETS xnt-miner DESTINATION bin)
endif()

# Install headers (for library users)
install(FILES ${HEADERS} DESTINATION include/xnt-miner)

# ===== Print configuration summary =====
message(STATUS "")
message(STATUS "=======================================================")
message(STATUS "          XNT Miner Build Configuration")
message(STATUS "=======================================================")
message(STATUS "")
message(STATUS "Build Settings:")
message(STATUS "  CMAKE_BUILD_TYPE:        ${CMAKE_BUILD_TYPE}")
message(STATUS "  CMAKE_CXX_STANDARD:      ${CMAKE_CXX_STANDARD}")
message(STATUS "  CMAKE_CUDA_STANDARD:     ${CMAKE_CUDA_STANDARD}")
message(STATUS "  CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "")
message(STATUS "")
message(STATUS "Dependencies:")
message(STATUS "  CUDAToolkit_VERSION:     ${CUDAToolkit_VERSION}")
message(STATUS "  nlohmann_json_FOUND:     ${nlohmann_json_FOUND}")
message(STATUS "")
message(STATUS "Targets:")
if(FILES_EXIST)
    message(STATUS "  xnt-miner           - Executable")
endif()
message(STATUS "")
message(STATUS "=======================================================")
message(STATUS "")
