#ifndef XNT_CONNECTION_MULTIPLEXER_CUH
#define XNT_CONNECTION_MULTIPLEXER_CUH

#include "common.cuh"
#include "gpu_resources.cuh"
#include "pow.cuh"
#include "mining_client.h"
#include <memory>
#include <vector>
#include <queue>
#include <mutex>
#include <shared_mutex>
#include <condition_variable>
#include <atomic>
#include <thread>
#include <future>

class XntRpcClient;
struct PowPuzzle;

// ============================================================================
// SoloMiningClient (Thread-safe HTTP JSON-RPC wrapper)
// ============================================================================

class SoloMiningClient : public MiningClient {
private:
    std::unique_ptr<XntRpcClient> rpc_client;
    std::string rpc_url;
    mutable std::mutex client_mutex;  // Thread-safe access
    std::atomic<bool> connected{false};
    
public:
    SoloMiningClient(const std::string& rpc_url);
    ~SoloMiningClient() override;
    
    SoloMiningClient(const SoloMiningClient&) = delete;
    SoloMiningClient& operator=(const SoloMiningClient&) = delete;
    
    bool connect() override;
    void disconnect() override;
    bool is_connected() const override;
    bool reconnect() override {
        return MiningClient::reconnect();
    }
    
    MiningMode get_mode() const override { return MiningMode::Solo; }
    
    json getBlockTemplate() override;
    json getBlockTemplate(const std::string& wallet_address) override;
    
    bool submitSolution(
        const std::string& proposal_id,
        const Pow& pow_solution,
        const Digest& solution_hash,
        const json& template_obj) override;
    
    // Alias for compatibility
    bool submit_solution(
        const std::string& proposal_id,
        const Pow& pow_solution,
        const Digest& solution_hash,
        const json& template_obj) override {
        return submitSolution(proposal_id, pow_solution, solution_hash, template_obj);
    }
    
    std::string getTipDigest() override;
    uint64_t getChainHeight() override;
    
    bool is_push_based() const override { return false; }
};

// ============================================================================
// SolutionSubmission (Queue Entry)
// ============================================================================

struct SolutionSubmission {
    int gpu_id;
    std::string proposal_id;
    Pow pow_solution;
    Digest solution_hash;
    json template_obj;
    std::chrono::steady_clock::time_point timestamp;
    std::shared_ptr<std::promise<bool>> result_promise;
    
    SolutionSubmission(int gpu, const std::string& id, const Pow& pow,
                       const Digest& hash, const json& tmpl)
        : gpu_id(gpu)
        , proposal_id(id)
        , pow_solution(pow)
        , solution_hash(hash)
        , template_obj(tmpl)
        , timestamp(std::chrono::steady_clock::now())
        , result_promise(std::make_shared<std::promise<bool>>()) {}
};

// ============================================================================
// GpuWorkerHandle (Registration Token)
// ============================================================================

struct GpuWorkerHandle {
    int gpu_id;
    GpuResources* resources;
    std::atomic<bool> active{true};
    
    // Statistics per GPU (tracked by multiplexer)
    std::atomic<uint64_t> jobs_received{0};
    std::atomic<uint64_t> submissions_queued{0};
    
    GpuWorkerHandle(int id, GpuResources* res) 
        : gpu_id(id), resources(res) {}
};

// ============================================================================
// ConnectionMultiplexer (Singleton)
// ============================================================================

class ConnectionMultiplexer {
public:
    // Singleton access
    static ConnectionMultiplexer& getInstance();
    static void destroyInstance();
    
    // Prevent copying
    ConnectionMultiplexer(const ConnectionMultiplexer&) = delete;
    ConnectionMultiplexer& operator=(const ConnectionMultiplexer&) = delete;
    
    // ========== Initialization ==========
    
    bool initialize(const std::string& endpoint, const std::string& wallet_address, 
                    const std::string& stratum_password = "");
    void shutdown();
    bool isInitialized() const { return initialized.load(); }
    
    // ========== Worker Registration ==========
    
    GpuWorkerHandle* registerWorker(int gpu_id, GpuResources* resources);
    void unregisterWorker(GpuWorkerHandle* handle);
    size_t getActiveWorkerCount() const;
    
    // ========== Solution Submission ==========
    
    std::future<bool> submitSolution(int gpu_id, const std::string& proposal_id,
                                      const Pow& pow_solution, const Digest& solution_hash,
                                      const json& template_obj);
    
    // ========== Connection State ==========
    
    bool isConnected() const;
    bool isRunning() const { return running.load(); }
    
    // ========== Statistics ==========
    
    struct Stats {
        std::atomic<uint64_t> total_jobs_fetched{0};
        std::atomic<uint64_t> total_jobs_broadcast{0};
        std::atomic<uint64_t> total_solutions_submitted{0};
        std::atomic<uint64_t> total_solutions_accepted{0};
        std::atomic<uint64_t> total_solutions_rejected{0};
        std::atomic<uint64_t> reconnection_count{0};
        std::chrono::steady_clock::time_point last_job_time;
        std::chrono::steady_clock::time_point last_submission_time;
        mutable std::mutex time_mutex;
    };
    
    const Stats& getStats() const { return stats; }
    
private:
    // Private constructor (singleton)
    ConnectionMultiplexer();
    
public:
    // Public destructor needed for unique_ptr in destroyInstance
    ~ConnectionMultiplexer();
    
private:
    
    // ========== Internal Components ==========
    
    std::unique_ptr<MiningClient> client;
    std::string endpoint;
    std::string wallet_address;
    MiningMode mining_mode{MiningMode::Solo};
    
    // Worker registry
    std::vector<std::unique_ptr<GpuWorkerHandle>> workers;
    mutable std::shared_mutex workers_mutex;
    
    // Solution submission queue
    std::queue<std::unique_ptr<SolutionSubmission>> submission_queue;
    std::mutex submission_mutex;
    std::condition_variable submission_cv;
    
    // Thread management
    std::thread job_broadcaster_thread;
    std::thread solution_submitter_thread;
    std::thread health_monitor_thread;
    std::thread tip_monitor_thread;  // Fast tip change detection
    std::atomic<bool> running{false};
    std::atomic<bool> initialized{false};
    std::atomic<bool> connected{false};
    
    // Current job state
    std::string last_template_id;
    std::string current_tip_digest;  // Track current chain tip for stale detection
    std::atomic<bool> composing_new_block{false};  // True when tip changed, waiting for valid new proposal
    std::string first_proposal_prev_block;  // Track first proposal's prev_block after tip change
    std::atomic<int> proposals_seen_for_tip{0};  // Count proposals seen for current tip
    mutable std::mutex job_mutex;
    
    // Statistics
    Stats stats;
    
    // ========== Thread Functions ==========
    
    void jobBroadcasterLoop();
    void solutionSubmitterLoop();
    void healthMonitorLoop();
    void tipMonitorLoop();  // Fast tip monitoring for stale detection
    
    // ========== Internal Helpers ==========
    
    void broadcastJobToWorkers(const json& job);
    bool processSubmission(SolutionSubmission& submission);
    bool attemptReconnection();
    void updateAllWorkersConnectionState(bool is_connected);
    
    // Singleton instance
    static std::unique_ptr<ConnectionMultiplexer> instance;
    static std::mutex instance_mutex;
};

#endif // XNT_CONNECTION_MULTIPLEXER_CUH
