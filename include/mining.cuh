#ifndef XNT_MINING_CUH
#define XNT_MINING_CUH

#include "gpu_resources.cuh"
#include "kernels.cuh"
#include "network.cuh"

class UnifiedMiningController;
class MultiGpuManager;
struct PowPuzzle;

extern UnifiedMiningController* g_mining_controller;

class UnifiedMiningController {
public:
    std::string rpc_url;
    int gpu_id;
    GpuResources* gpu_resources;
    
private:
    std::thread fetcher_thread;
    std::thread mining_thread;
    std::atomic<bool> controller_running{false};
    
public:
    UnifiedMiningController(
        int gpu_id, 
        GpuResources* resources,
        const std::string& rpc_url = "http://127.0.0.1:9897");
    
    ~UnifiedMiningController();
    
    UnifiedMiningController(const UnifiedMiningController&) = delete;
    UnifiedMiningController& operator=(const UnifiedMiningController&) = delete;
    
    void start();
    void stop();
    bool isRunning() const { return controller_running.load(); }
    bool reconnect_to_node();
    
private:
    void fetcherLoop();
    void miningLoop();
    void handleNewPuzzle(const MiningEvent& event);
    void handleSolutionFound(const MiningEvent& event);
    void handleError(const MiningEvent& event);
    bool initializeCuda();
    bool connectToNode();
    json requestJob();
    bool processJobResponse(const json& job_response);
};

class MultiGpuManager {
private:
    std::vector<int> gpu_ids;
    std::vector<std::unique_ptr<GpuResources>> gpu_resources;
    std::vector<std::unique_ptr<UnifiedMiningController>> controllers;
    std::vector<std::thread> gpu_threads;
    std::string rpc_url;
    int single_gpu_id;
    
public:
    MultiGpuManager(
        const std::string& rpc_url = "http://127.0.0.1:9897",
        int specific_gpu = -1);
    
    ~MultiGpuManager();
    
    MultiGpuManager(const MultiGpuManager&) = delete;
    MultiGpuManager& operator=(const MultiGpuManager&) = delete;
    
    bool detectAndInitGpus();
    size_t getGpuCount() const { return gpu_ids.size(); }
    const std::vector<int>& getGpuIds() const { return gpu_ids; }
    
    void startAll();
    void stopAll();
    
    GpuResources* getGpuResources(size_t index);
    const std::vector<std::unique_ptr<GpuResources>>& getAllGpuResources() const {
        return gpu_resources;
    }
    
private:
    bool initializeGpu(int device_id);
    void gpuMiningThread(size_t index);
};

void startUnifiedMining(
    const std::string& rpc_url = "http://127.0.0.1:9897",
    int specific_gpu = -1);

void puzzleFetcher(GpuResources* gpu_res, UnifiedMiningController* controller);

bool preprocessPuzzle(const PowPuzzle& puzzle, GpuResources* gpu_res);

bool minePuzzleWithCuda(
    const PowPuzzle& puzzle, 
    GpuResources* gpu_res);

bool continuousMiningLoop(GpuResources* gpu_res, UnifiedMiningController* controller);

bool verifySolution(
    const Pow& pow,
    const PowPuzzle& puzzle,
    const Digest& commitment,
    const Digest& target);

uint64_t getNextNonceRange(GpuResources* gpu_res, uint64_t batch_size);
void resetNonceCounter(GpuResources* gpu_res, const std::string& puzzle_id);

void signal_handler(int signal);
void install_signal_handlers();

inline double get_timestamp_ms() {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::microseconds>(duration).count() / 1000.0;
}

inline double calculate_hashrate(uint64_t nonces, double elapsed_ms) {
    if (elapsed_ms <= 0) return 0.0;
    return (static_cast<double>(nonces) / elapsed_ms) * 1000.0;
}

constexpr int TARGET_BATCH_DURATION_MS = 900;
constexpr int MAX_BATCH_DURATION_MS = 2000;
constexpr int MIN_BATCH_DURATION_MS = 200;
constexpr int JOB_STALENESS_THRESHOLD_SEC = 60;
constexpr int HASHRATE_SUBMISSION_INTERVAL_SEC = 600;
constexpr int MIN_RECONNECT_DELAY_SEC = 5;
constexpr int MAX_RECONNECT_DELAY_SEC = 60;
constexpr int MAX_CONSECUTIVE_RECONNECTIONS = 10;
constexpr int PAUSE_THRESHOLD_SEC = 30;

#endif
