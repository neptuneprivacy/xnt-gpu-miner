#ifndef XNT_MINING_CUH
#define XNT_MINING_CUH

#include "gpu_resources.cuh"
#include "kernels.cuh"
#include "network.cuh"
#include <future>

class MultiGpuManager;
class GpuWorker;
struct PowPuzzle;
struct GpuWorkerHandle;

// ============================================================================
// GpuWorker - Per-GPU controller using shared ConnectionMultiplexer
// ============================================================================
// GpuWorker does NOT manage connections - it receives jobs via EventHandler
// and submits solutions through the ConnectionMultiplexer

class GpuWorker {
public:
    std::string endpoint;  // RPC URL or Stratum URL
    int gpu_id;
    GpuResources* gpu_resources;
    MiningMode mining_mode;  // Solo or Stratum
    
private:
    std::thread mining_thread;
    std::atomic<bool> worker_running{false};
    GpuWorkerHandle* worker_handle;  // Registration with multiplexer
    
public:
    GpuWorker(int gpu_id, GpuResources* resources);
    ~GpuWorker();
    std::atomic<bool> controller_running{false};
    std::string stratum_password;  // For stratum mode
    
public:
    UnifiedMiningController(
        int gpu_id, 
        GpuResources* resources,
        const std::string& endpoint = "http://127.0.0.1:9897",
        MiningMode mode = MiningMode::Solo,
        const std::string& stratum_pass = "x");
    
    ~UnifiedMiningController();
    
    GpuWorker(const GpuWorker&) = delete;
    GpuWorker& operator=(const GpuWorker&) = delete;
    
    void start();
    void stop();
    bool isRunning() const { return worker_running.load(); }
    
    // Submit solution through multiplexer (non-blocking, returns future)
    std::future<bool> submitSolution(const std::string& proposal_id,
                                      const Pow& pow_solution,
                                      const Digest& solution_hash,
                                      const json& template_obj);
    
private:
    bool initializeCuda();
    void fetcherLoop();
    void stratumFetcherLoop();  // Stratum-specific fetcher (push-based)
    void miningLoop();
    void handleNewPuzzle(const MiningEvent& event);
};

// ============================================================================
// MultiGpuManager - Coordinates multiple GPU workers with shared connection
// ============================================================================

class MultiGpuManager {
private:
    std::vector<int> gpu_ids;
    std::vector<std::unique_ptr<GpuResources>> gpu_resources;
    std::vector<std::unique_ptr<GpuWorker>> workers;
    std::vector<std::thread> gpu_threads;
    std::string endpoint;  // RPC URL or Stratum URL
    int single_gpu_id;
    MiningMode mining_mode;
    std::string stratum_password;
    
public:
    MultiGpuManager(
        const std::string& endpoint = "http://127.0.0.1:9897",
        int specific_gpu = -1,
        MiningMode mode = MiningMode::Solo,
        const std::string& stratum_pass = "x");
    
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
    void gpuWorkerThread(size_t index);
};

void startUnifiedMining(
    const std::string& endpoint = "http://127.0.0.1:9897",
    int specific_gpu = -1,
    MiningMode mode = MiningMode::Solo,
    const std::string& stratum_pass = "x");

// Continuous mining loop (uses GpuWorker)
bool continuousMiningLoop(GpuResources* gpu_res, GpuWorker* worker);

bool preprocessPuzzle(const PowPuzzle& puzzle, GpuResources* gpu_res);

bool minePuzzleWithCuda(
    const PowPuzzle& puzzle, 
    GpuResources* gpu_res);

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
