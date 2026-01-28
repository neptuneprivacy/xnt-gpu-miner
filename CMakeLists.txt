cmake_minimum_required(VERSION 3.18)

# Must set CUDA architectures before project() to avoid CMake using default sm_52
# Support modern GPUs (CUDA 13.0+):
# Ampere (80, 86), Ada Lovelace (89), Hopper (90), Blackwell (100)
set(CMAKE_CUDA_ARCHITECTURES "80;86;89;90")

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
    # Ampere (80, 86), Ada Lovelace (89), Hopper (90), Blackwell (100)
    # Note: Pascal (60), Volta (70), Turing (75) dropped in CUDA 13.0
    set(CMAKE_CUDA_ARCHITECTURES 80 86 89 90 100)
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

# Enable separable compilation for CUDA (required for modular build)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -lineinfo")
endif()

# Option to choose build type
option(BUILD_MODULAR "Build from modular source files" ON)
option(BUILD_MONOLITHIC "Build from monolithic source file" ON)

# Include directories
set(INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

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

if(nlohmann_json_FOUND)
    list(APPEND LINK_LIBS nlohmann_json::nlohmann_json)
endif()

# ===== MODULAR SOURCE FILES =====
set(MODULAR_HEADERS
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
)

set(MODULAR_SOURCES
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
    src/mining.cu
    src/main.cu
)

# ===== OPTION 1: Build from monolithic file =====
if(BUILD_MONOLITHIC)
    add_executable(xnt-miner xnt-miner.cu)

    target_include_directories(xnt-miner PRIVATE ${INCLUDE_DIRS})
    target_link_libraries(xnt-miner PRIVATE ${LINK_LIBS})

    # Set CUDA properties
    set_target_properties(xnt-miner PROPERTIES
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )

    # Add NVCC-specific compiler definitions for MSVC compatibility
    if(MSVC)
        target_compile_definitions(xnt-miner PRIVATE
            __NVCC__=1
            __CUDACC_VER_MAJOR__=12
            __CUDACC_VER_MINOR__=6
        )
    endif()

    message(STATUS "Building monolithic xnt-miner from xnt-miner.cu")
endif()

# ===== OPTION 2: Build from modular source files =====
if(BUILD_MODULAR)
    # Check if all source files exist
    set(MODULAR_FILES_EXIST TRUE)
    foreach(src ${MODULAR_SOURCES})
        if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${src}")
            set(MODULAR_FILES_EXIST FALSE)
            message(STATUS "Modular source file missing: ${src}")
        endif()
    endforeach()

    if(MODULAR_FILES_EXIST)
        add_executable(xnt-miner-modular ${MODULAR_SOURCES})

        target_include_directories(xnt-miner-modular PRIVATE ${INCLUDE_DIRS})
        target_link_libraries(xnt-miner-modular PRIVATE ${LINK_LIBS})

        # Set CUDA properties for separable compilation
        set_target_properties(xnt-miner-modular PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
        )

        message(STATUS "Building modular xnt-miner-modular from src/*.cu")
    else()
        message(STATUS "Skipping modular build - not all source files exist yet")
        message(STATUS "To enable modular build, create all files in src/")
    endif()
endif()

# ===== STATIC LIBRARY (for testing/embedding) =====
option(BUILD_LIBRARY "Build as static library" OFF)
if(BUILD_LIBRARY AND MODULAR_FILES_EXIST)
    # Build without main.cu for library
    list(FILTER MODULAR_SOURCES EXCLUDE REGEX "main\\.cu$")

    add_library(xnt-miner-lib STATIC ${MODULAR_SOURCES})

    target_include_directories(xnt-miner-lib PUBLIC ${INCLUDE_DIRS})
    target_link_libraries(xnt-miner-lib PUBLIC ${LINK_LIBS})

    set_target_properties(xnt-miner-lib PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )

    message(STATUS "Building static library xnt-miner-lib")
endif()

# ===== Installation =====
if(BUILD_MONOLITHIC)
    install(TARGETS xnt-miner DESTINATION bin)
endif()

if(BUILD_MODULAR AND MODULAR_FILES_EXIST)
    install(TARGETS xnt-miner-modular DESTINATION bin)
endif()

# Install headers (for library users)
install(FILES ${MODULAR_HEADERS} DESTINATION include/xnt-miner)

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
message(STATUS "Build Options:")
message(STATUS "  BUILD_MONOLITHIC:        ${BUILD_MONOLITHIC}")
message(STATUS "  BUILD_MODULAR:           ${BUILD_MODULAR}")
message(STATUS "  BUILD_LIBRARY:           ${BUILD_LIBRARY}")
message(STATUS "")
message(STATUS "Dependencies:")
message(STATUS "  CUDAToolkit_VERSION:     ${CUDAToolkit_VERSION}")
message(STATUS "  nlohmann_json_FOUND:     ${nlohmann_json_FOUND}")
message(STATUS "")
message(STATUS "Targets:")
if(BUILD_MONOLITHIC)
    message(STATUS "  xnt-miner           - Monolithic build")
endif()
if(BUILD_MODULAR AND MODULAR_FILES_EXIST)
    message(STATUS "  xnt-miner-modular   - Modular build")
endif()
if(BUILD_LIBRARY AND MODULAR_FILES_EXIST)
    message(STATUS "  xnt-miner-lib       - Static library")
endif()
message(STATUS "")
message(STATUS "=======================================================")
message(STATUS "")
