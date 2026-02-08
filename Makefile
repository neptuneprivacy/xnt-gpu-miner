# XNT Miner Makefile
# Simple build without CMake

NVCC = nvcc
CXX = g++

# CUDA architecture flags
# RTX 5090 has compute capability 12.0 (Blackwell) = sm_100
# Options: 
#   -arch=sm_100 (Blackwell/RTX 5090 - optimal)
#   -arch=sm_90  (Ada/Hopper - fallback compatibility)
#   -arch=native (auto-detect)
# Usage: make ARCH=sm_90 or make ARCH=sm_100
ifdef ARCH
    CUDA_ARCH = -arch=$(ARCH)
else
    CUDA_ARCH = -arch=sm_120  # Blackwell RTX 5090 (compute 12.0)
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
# Device link-time optimization (enables cross-TU inlining, better reg allocation)
# Disable with DLTO=0 if build is too slow
DLTO ?= 1
ifeq ($(DLTO),1)
    NVCC_FLAGS += -dlto
    NVCC_DC_FLAGS += -dlto
endif
# Register limit: 255 gives best perf on Blackwell (override with MAX_REG_COUNT=<n>)
MAX_REG_COUNT ?= 255
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

.PHONY: all clean modular test-rpc help

# Default: build from modular sources
all: modular

# Build from modular sources
modular: $(TARGET)

# Object files for separable compilation
OBJS = $(MODULAR_SRCS:.cu=.o)
# Device link object file
DEVICE_LINK_OBJ = device_link.o

$(TARGET): $(OBJS)
ifdef DLTO
	# With device LTO, let nvcc perform device linking internally to avoid duplicate registration stubs
	$(NVCC) $(NVCC_FLAGS) -o $@ $(OBJS) $(LIBS)
else
	# First, create device link object
	$(NVCC) $(NVCC_FLAGS) -dlink -o $(DEVICE_LINK_OBJ) $(OBJS)
	# Then link everything together into final executable
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

# Clean build artifacts
clean:
	rm -f $(TARGET) test_rpc test_rpc_simple *.o src/*.o

# Help
help:
	@echo "XNT Miner Build System"
	@echo ""
	@echo "Targets:"
	@echo "  all        - Build from modular sources (default)"
	@echo "  modular    - Build from modular src/ files"
	@echo "  clean      - Remove build artifacts"
	@echo "  help       - Show this help"
	@echo ""
	@echo ""
	@echo "Examples:"
	@echo "  make                  # Release build"
	@echo "  make DEBUG=1          # Debug build"
	@echo "  make clean && make    # Clean rebuild"
	@echo "  make MAX_REG_COUNT=48 # Build with max 48 registers (higher occupancy)"
	@echo "  make MAX_REG_COUNT=80 # Build with max 80 registers (if 64 causes spilling)"