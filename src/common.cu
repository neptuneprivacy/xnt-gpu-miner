#include "common.cuh"

std::atomic<bool> stop_mining{false};

std::string g_miner_wallet_address = "";
std::string g_miner_worker_name = "";
std::atomic<int> g_total_gpu_count{1};
int g_gpu_device_id = -1;
bool g_test_mode = false;
bool g_benchmark_mode = false;
int g_fetch_interval_sec = 5;

std::mutex g_log_mutex;

// Tuning parameters (optimized defaults)
int g_block_size = 256;              // Threads per block (optimal for LUT loading)
uint64_t g_batch_size = DEFAULT_BATCH_SIZE;  // Mining + benchmark default (0 = auto)
