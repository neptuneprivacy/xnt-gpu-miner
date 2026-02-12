# XNT Miner Makefile
# Simple build without CMake

NVCC = nvcc
CXX = g++

# CUDA architecture: default = both RTX 4090 (sm_89) and RTX 5090 (sm_120).
# Set ARCH for a single-GPU build (faster compile, smaller binary).
#   sm_89  - Ada Lovelace (RTX 4090, 4080, 4070)
#   sm_120 - Blackwell (RTX 5090)
#   sm_90  - Hopper (H100, H200)
#   sm_86  - Ampere (RTX 3080, 3090)  |  sm_80  - Ampere (RTX 3070)
#   native - auto-detect from current GPU
# Usage: make              # 4090 + 5090  |  make ARCH=sm_89  # 4090 only
ifdef ARCH
    CUDA_ARCH = -arch=$(ARCH)
else
    # One binary for both RTX 4090 and RTX 5090
    CUDA_ARCH = -gencode arch=compute_89,code=sm_89 -gencode arch=compute_120,code=sm_120
endif

# Compiler flags
# Unity build: all device code is in mining_core.cu (single TU).
# -dc is required because main.cu references device symbols.
# -dlto is required to propagate __launch_bounds__ register constraints
# across __noinline__ device function calls (tip5_permutation etc.).
NVCC_FLAGS = -O3 --use_fast_math -std=c++17 $(CUDA_ARCH) \
             --ptxas-options=-v \
             --ptxas-options=-O3 \
             -Xnvlink=-suppress-stack-size-warning
NVCC_DC_FLAGS = -O3 --use_fast_math -std=c++17 $(CUDA_ARCH) \
                --ptxas-options=-v \
                --ptxas-options=-O3 \
                -dc
# Device link-time optimization: propagates __launch_bounds__ register limits
# across __noinline__ function calls. Only valid with single -arch (not -gencode).
DLTO ?= 1
ifdef ARCH
ifeq ($(DLTO),1)
    NVCC_FLAGS += -dlto
    NVCC_DC_FLAGS += -dlto
endif
endif
# Register limit: let __launch_bounds__ control per-kernel register allocation.
# Override globally with MAX_REG_COUNT=<n> on the command line if needed.
ifdef MAX_REG_COUNT
    NVCC_FLAGS += -maxrregcount=$(MAX_REG_COUNT)
    NVCC_DC_FLAGS += -maxrregcount=$(MAX_REG_COUNT)
endif
CXX_FLAGS = -O3 -march=native -std=c++17

# Include paths
INCLUDES = -I./include

# Libraries
LIBS = -lcudart -lpthread -lssl -lcrypto


# Output binary
TARGET = xnt-miner

# Source files - modular build
# mining_core.cu is a unity build: tip5, digest, merkle, kernels, pow, and mining
# (GpuWorker, MultiGpuManager, etc.) in one TU for maximum compiler optimization.
MODULAR_SRCS = src/common.cu \
               src/mining_core.cu \
               src/network.cu \
               src/rpc_client.cu \
               src/stratum_client.cu \
               src/connection_multiplexer.cu \
               src/main.cu

.PHONY: all clean modular unity test-rpc help rebuild

# Default: unity build (single TU for max performance)
all: unity

# Unity build: single translation unit, best runtime performance
# Use: make unity  or  make BUILD_UNITY=1
BUILD_UNITY ?= 0
unity:
	$(MAKE) BUILD_UNITY=1 $(TARGET)
	@echo "Built $(TARGET) (unity - single TU)"

# Build from modular sources (faster incremental compile)
modular:
	$(MAKE) BUILD_UNITY=0 $(TARGET)

# Clean rebuild — runs clean first, then build. Safe with -j (uses recursive make).
rebuild:
	$(MAKE) clean
	$(MAKE) all

# Object files for separable compilation
OBJS = $(MODULAR_SRCS:.cu=.o)
DEVICE_LINK_OBJ = device_link.o
UNITY_SRC = xnt-miner-unity.cu

# Regenerate xnt-miner-unity.cu from src/*.cu when any source changes
$(UNITY_SRC): $(MODULAR_SRCS)
	@echo "Regenerating $(UNITY_SRC)..."
	@echo '// xnt-miner-unity.cu — Single file: all source inlined' > $@
	@echo '// Auto-generated from src/*.cu — do not edit' >> $@
	@echo '' >> $@
	@echo '#define XNT_UNITY_BUILD 1' >> $@
	@echo '' >> $@
	@echo '// --- common.cu ---' >> $@
	@cat src/common.cu >> $@
	@echo '' >> $@
	@echo '// --- mining_core.cu ---' >> $@
	@cat src/mining_core.cu >> $@
	@echo '' >> $@
	@echo '// --- rpc_client.cu ---' >> $@
	@cat src/rpc_client.cu >> $@
	@echo '' >> $@
	@echo '// --- stratum_client.cu ---' >> $@
	@cat src/stratum_client.cu >> $@
	@echo '' >> $@
	@echo '// --- network.cu ---' >> $@
	@cat src/network.cu >> $@
	@echo '' >> $@
	@echo '// --- connection_multiplexer.cu ---' >> $@
	@cat src/connection_multiplexer.cu >> $@
	@echo '' >> $@
	@echo '// --- main.cu ---' >> $@
	@cat src/main.cu >> $@
	@echo "Done."

# Unity build: single TU (conditional)
ifeq ($(BUILD_UNITY),1)
$(TARGET): $(UNITY_SRC)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -I. -o $@ $(UNITY_SRC) $(LIBS)
# Modular build: link object files
else
$(TARGET): $(OBJS)
ifdef DLTO
	$(NVCC) $(NVCC_FLAGS) -o $@ $(OBJS) $(LIBS)
else
	$(NVCC) $(NVCC_FLAGS) -dlink -o $(DEVICE_LINK_OBJ) $(OBJS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $(OBJS) $(DEVICE_LINK_OBJ) $(LIBS)
endif
endif

# Compile each .cu file to .o with device code (separable compilation)
%.o: %.cu
	$(NVCC) $(NVCC_DC_FLAGS) $(INCLUDES) -c -o $@ $<

# Test RPC endpoint (standalone)
test-rpc: test_rpc_simple
	@echo "Built test_rpc_simple executable"

test_rpc_simple: test_rpc_simple.cpp
	$(CXX) $(CXX_FLAGS) -I./include -o $@ $<

# Clean build artifacts. Use "make rebuild" or "make clean && make" — NEVER "make clean & make"
clean:
	rm -f $(TARGET) test_rpc test_rpc_simple *.o src/*.o $(DEVICE_LINK_OBJ)

# Help
help:
	@echo "XNT Miner Build System"
	@echo ""
	@echo "Targets:"
	@echo "  all        - Unity build (default, max performance)"
	@echo "  unity      - Single TU build (best runtime performance)"
	@echo "  modular    - Modular build (faster incremental compile)"
	@echo "  rebuild    - Clean + build (safe, sequential)"
	@echo "  clean      - Remove build artifacts"
	@echo "  help       - Show this help"
	@echo ""
	@echo ""
	@echo "Examples:"
	@echo "  make                          # Unity build (default)"
	@echo "  make unity ARCH=sm_89        # Unity for 4090 only"
	@echo "  make modular                  # Modular (faster incremental)"
	@echo "  make rebuild ARCH=sm_89      # Clean rebuild for 4090"
	@echo "  make clean && make ARCH=sm_89 # Same (sequential)"
	@echo "  make MAX_REG_COUNT=48 # Build with max 48 registers (higher occupancy)"
	@echo "  make MAX_REG_COUNT=80 # Build with max 80 registers (if 64 causes spilling)"
	@echo ""
	@echo "CUDA architecture (default: 4090 + 5090; use ARCH= for single GPU):"
	@echo "  make              # RTX 4090 and RTX 5090 (default)"
	@echo "  make ARCH=sm_89   # RTX 4090 only"
	@echo "  make ARCH=sm_120  # RTX 5090 only"
	@echo "  make ARCH=sm_90   # Hopper  |  make ARCH=native  # Auto-detect"
