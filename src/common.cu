#include "common.cuh"

std::atomic<bool> stop_mining{false};

std::string g_miner_wallet_address = "";
std::atomic<int> g_total_gpu_count{1};
int g_gpu_device_id = -1;

std::mutex g_log_mutex;
