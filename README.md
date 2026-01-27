# XNT GPU Miner

A high-performance CUDA-based solo miner for Neptune (XNT) using the TIP5 hash algorithm with JSON-RPC communication.

## Requirements

- NVIDIA GPU with ≥11GB VRAM (for HIGH_VRAM mode)
- CUDA Toolkit 11.0 or later
- C++17 compatible compiler
- nlohmann/json library (`apt install nlohmann-json3-dev`)
- xnt-core node with JSON-RPC enabled

## Quick Start

### 1. Start xnt-core with JSON-RPC

```bash
xnt-core \
  --network main \
  --peer 161.97.150.88:9898 \
  --peer 154.38.160.61:9898 \
  --listen-rpc 0.0.0.0:9897 \
  --rpc-modules Node,Chain,Mining
```

**Key flags:**
- `--listen-rpc <ADDRESS:PORT>`: Enable JSON-RPC HTTP server (default: `127.0.0.1:9897`)
- `--rpc-modules <MODULES>`: Enable RPC modules (e.g., `Node,Chain,Mining`)
- `--network <NETWORK>`: Network (`main`, `alpha`, `beta`, `testnet`, `regtest`)

### 2. Build the Miner

```bash
cd xnt-gpu-miner
make
```

**Build options:**
- `make clean`: Remove build artifacts

### 3. Start Mining

```bash
# Mine with all available GPUs (default: http://127.0.0.1:9897)
./xnt-miner -w YOUR_WALLET_ADDRESS

# Use specific GPU
./xnt-miner -w YOUR_WALLET_ADDRESS -d 0

# Connect to remote node
./xnt-miner -w YOUR_WALLET_ADDRESS --rpc-url http://192.168.1.100:9897
```

## Command-Line Options

| Option | Description |
|--------|-------------|
| `-w, --wallet ADDRESS` | Your Neptune wallet address (required) |
| `-d, --device ID` | Use specific GPU device ID (default: all GPUs) |
| `--rpc-url URL` | RPC endpoint URL (default: `http://127.0.0.1:9897`) |
| `-H, --host HOST` | RPC host (alternative to --rpc-url) |
| `-p, --port PORT` | RPC port (alternative to --rpc-url) |
| `-h, --help` | Show help message |

## RPC Endpoints

The miner communicates with xnt-core via JSON-RPC 2.0:

| Method | Purpose |
|--------|---------|
| `mining_getBlockTemplate` | Get block template for mining |
| `mining_submitBlock` | Submit solved block |
| `chain_height` | Get current block height |
| `chain_tipDigest` | Check template freshness (validate not stale) |

## Architecture

**Core Components:**
- **TIP5 Hash**: Goldilocks field arithmetic, S-box/MDS operations, sponge construction
- **Digest**: 5×64-bit (40 byte) structure with fixed/variable length hashing
- **Merkle Tree**: GPU-accelerated construction with 2^27 leafs (134M leafs, ~268M total nodes)
- **Proof-of-Work**: Puzzle representation, preprocessing, solution verification
- **Mining Kernels**: HIGH_VRAM (full tree) or LOW_VRAM (compute on-demand) modes
- **RPC Client**: JSON-RPC 2.0 over HTTP for template fetching and submission
- **GPU Resources**: Per-GPU state management, hash rate tracking, event-driven loop
- **Mining Controller**: Multi-GPU orchestration with 5-second template polling

## Project Structure

```
xnt-gpu-miner/
├── include/          # Header files (tip5, digest, pow, merkle, kernels, etc.)
├── src/              # Source files (main, mining, network, rpc_client, etc.)
├── CMakeLists.txt    # CMake build configuration
└── Makefile          # Simple make build
```

## License

See LICENSE file for details.
