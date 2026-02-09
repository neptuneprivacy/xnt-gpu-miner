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
# Enable separable compilation for device functions across multiple files
# Register optimization flags:
#   -maxrregcount=64: Limit registers to improve occupancy (can be adjusted)
#   --ptxas-options=-v: Show register usage during compilation
#   --ptxas-options=-O3: Aggressive PTX optimization
# Note: Lower maxrregcount = better occupancy but may cause register spilling
#       Start with 64, increase if you see performance degradation
NVCC_FLAGS = -O3 --use_fast_math -std=c++17 $(CUDA_ARCH) \
             --ptxas-options=-v \
             --ptxas-options=-O3 \
             -Xnvlink=-suppress-stack-size-warning
NVCC_DC_FLAGS = -O3 --use_fast_math -std=c++17 $(CUDA_ARCH) \
                --ptxas-options=-v \
                --ptxas-options=-O3 \
                -dc
# Device link-time optimization (enables cross-TU inlining, better reg allocation).
# Only valid with single -arch=... ; nvcc does not allow -dlto with -gencode (multi-arch).
# Disable with DLTO=0 if build is too slow.
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
MODULAR_SRCS = src/common.cu \
               src/tip5.cu \
               src/digest.cu \
               src/merkle.cu \
               src/pow.cu \
               src/kernels.cu \
               src/network.cu \
               src/rpc_client.cu \
               src/stratum_client.cu \
               src/gpu_resources.cu \
               src/connection_multiplexer.cu \
               src/mining.cu \
               src/main.cu

.PHONY: all clean modular test-rpc help rebuild

# Default: build from modular sources
all: modular

# Build from modular sources
modular: $(TARGET)

# Clean rebuild — runs clean first, then build. Safe with -j (uses recursive make).
rebuild:
	$(MAKE) clean
	$(MAKE) all

# Object files for separable compilation
OBJS = $(MODULAR_SRCS:.cu=.o)
DEVICE_LINK_OBJ = device_link.o

$(TARGET): $(OBJS)
ifdef DLTO
	$(NVCC) $(NVCC_FLAGS) -o $@ $(OBJS) $(LIBS)
else
	$(NVCC) $(NVCC_FLAGS) -dlink -o $(DEVICE_LINK_OBJ) $(OBJS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $(OBJS) $(DEVICE_LINK_OBJ) $(LIBS)
endif

# Compile each .cu file to .o with device code
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
	@echo "  all        - Build from modular sources (default)"
	@echo "  modular    - Build from modular src/ files"
	@echo "  rebuild    - Clean + build (safe, sequential)"
	@echo "  clean      - Remove build artifacts"
	@echo "  help       - Show this help"
	@echo ""
	@echo ""
	@echo "Examples:"
	@echo "  make                          # Release build"
	@echo "  make rebuild ARCH=sm_89       # Clean rebuild for 4090"
	@echo "  make clean && make ARCH=sm_89 # Same (sequential)"
	@echo "  make MAX_REG_COUNT=48 # Build with max 48 registers (higher occupancy)"
	@echo "  make MAX_REG_COUNT=80 # Build with max 80 registers (if 64 causes spilling)"
	@echo ""
	@echo "CUDA architecture (default: 4090 + 5090; use ARCH= for single GPU):"
	@echo "  make              # RTX 4090 and RTX 5090 (default)"
	@echo "  make ARCH=sm_89   # RTX 4090 only"
	@echo "  make ARCH=sm_120  # RTX 5090 only"
	@echo "  make ARCH=sm_90   # Hopper  |  make ARCH=native  # Auto-detect"
