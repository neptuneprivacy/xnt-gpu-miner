#include "gpu_resources.cuh"

// ===== GLOBAL VARIABLES =====
std::vector<GpuResources*> g_all_gpu_resources;
std::mutex g_all_gpu_resources_mutex;


std::string get_gpu_uuid(int device_id) {
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device_id);
    
    if (err != cudaSuccess) {
        return "unknown";
    }
    
    // Convert UUID bytes to hex string
    std::ostringstream oss;
    oss << std::hex << std::setfill('0');
    
    for (int i = 0; i < 16; ++i) {
        oss << std::setw(2) << static_cast<int>(static_cast<unsigned char>(prop.uuid.bytes[i]));
        if (i == 3 || i == 5 || i == 7 || i == 9) {
            oss << "-";
        }
    }
    
    return oss.str();
}

void broadcast_stop_to_all_gpus(int source_gpu_id, const char* reason) {
    std::lock_guard<std::mutex> lock(g_all_gpu_resources_mutex);
    
    int stopped_count = 0;
    for (GpuResources* gpu : g_all_gpu_resources) {
        if (gpu) {
            try {
                bool was_stopped = gpu->gpu_stop_flag.load();
                if (!was_stopped) {
                    gpu->gpu_stop_flag = true;
                    stopped_count++;
                    if (gpu->event_handler) {
                        gpu->event_handler->postEvent(EventType::STOP_MINING);
                    }
                }
            } catch (...) {
                continue;
            }
        }
    }
    
    if (stopped_count > 0) {
        LOG_DEBUG("[GPU " << source_gpu_id << "] " << reason
                  << " - Broadcasting stop to " << stopped_count << " GPU(s)");
    }
}

GpuResources* find_gpu_resources(int gpu_id) {
    std::lock_guard<std::mutex> lock(g_all_gpu_resources_mutex);
    
    for (GpuResources* gpu : g_all_gpu_resources) {
        if (gpu && gpu->gpu_id == gpu_id) {
            return gpu;
        }
    }
    
    return nullptr;
}

double get_total_hashrate() {
    std::lock_guard<std::mutex> lock(g_all_gpu_resources_mutex);
    
    double total = 0.0;
    for (GpuResources* gpu : g_all_gpu_resources) {
        if (gpu && gpu->is_mining()) {
            total += gpu->hash_tracker.get_average();
        }
    }
    
    return total;
}

uint64_t get_total_solutions_found() {
    std::lock_guard<std::mutex> lock(g_all_gpu_resources_mutex);
    
    uint64_t total = 0;
    for (GpuResources* gpu : g_all_gpu_resources) {
        if (gpu) {
            total += gpu->solutions_found.load();
        }
    }
    
    return total;
}

void cleanup_gpu_memory() {
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    
    if (error != cudaSuccess || device_count == 0) {
        return;
    }
    
    for (int gpu_id = 0; gpu_id < device_count; ++gpu_id) {
        cudaSetDevice(gpu_id);
        cudaDeviceSynchronize();
        cudaGetLastError();
        cudaDeviceReset();
    }
    
    {
        std::lock_guard<std::mutex> lock(g_all_gpu_resources_mutex);
        g_all_gpu_resources.clear();
    }
}