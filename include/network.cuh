#ifndef XNT_NETWORK_CUH
#define XNT_NETWORK_CUH

#include "common.cuh"
#include "pow.cuh"

// Forward declarations
class Pow;
struct Digest;

// ===== RPC CLIENT FORWARD DECLARATION =====
class XntRpcClient;

// ===== PUZZLE STRUCTURES =====
// AuthPaths is defined in pow.cuh

// POW puzzle from xnt-core node
struct PowPuzzle {
    std::string id;                    // Unique puzzle identifier
    std::string threshold;             // Target threshold (hex string)
    std::string total_guesser_reward;  // Reward for finding solution
    std::string prev_block;            // Previous block hash (hex string)
    AuthPaths auth_paths;              // MAST authentication paths
    int consensus_rule_set;            // Consensus rules (0=Reboot, 1=HardforkAlpha, 2=Xnt)
    
    PowPuzzle()
        : consensus_rule_set(CONSENSUS_XNT) {}
    
    bool is_valid() const {
        return !id.empty() && !threshold.empty() && 
               !auth_paths.pow.empty() && !auth_paths.header.empty() && !auth_paths.kernel.empty();
    }
};

// ===== SUBMISSION RESULT =====
struct SubmissionResult {
    bool success;              // Whether submission was accepted
    std::string message;       // Success/error message
    std::string error;         // Error details (if any)
    SubmissionResult()
        : success(false) {}
    
    SubmissionResult(bool s, const std::string& msg = "", const std::string& err = "")
        : success(s)
        , message(msg)
        , error(err) {}
};

// ===== NEPTUNE CUDA MINER CLIENT =====
// RPC-based client for communication with xnt-core node
// Uses JSON-RPC 2.0 over HTTP
class NeptuneCudaMinerClient {
private:
    // RPC client for node communication
    std::unique_ptr<XntRpcClient> rpc_client;
    
    // Node connection info (RPC URL for xnt-core node)
    std::string rpc_url;
    
    // Current job tracking
    std::string current_job_id;
    std::string worker_id;
    
    // Last puzzle cache (template from RPC)
    json last_template;
    std::chrono::steady_clock::time_point last_template_time;
    bool has_last_template;
    
    // Wallet address for getBlockTemplate calls
    std::string wallet_address;

public:
    // ===== CONSTRUCTORS/DESTRUCTOR =====
    
    NeptuneCudaMinerClient(
        const std::string& rpc_url = "http://127.0.0.1:9899",
        const std::string& wallet_addr = "");
    
    ~NeptuneCudaMinerClient();
    
    // Disable copying
    NeptuneCudaMinerClient(const NeptuneCudaMinerClient&) = delete;
    NeptuneCudaMinerClient& operator=(const NeptuneCudaMinerClient&) = delete;
    
    // ===== CONNECTION MANAGEMENT =====
    
    // Connect to the xnt-core node (test RPC connection)
    bool connect_to_node();
    
    // Disconnect from node (no-op for RPC)
    void disconnect() {}
    
    // Check if connected to node (test RPC connection)
    bool is_connected() const;
    
    // Check connection health (test RPC call)
    bool check_connection_health();
    
    // Get connection status string for display
    std::string get_connection_status() const;
    
    // ===== JOB MANAGEMENT =====
    
    // Get current job ID
    std::string get_current_job_id() const { return current_job_id; }
    
    // Set current job ID
    void set_current_job_id(const std::string& job_id) { current_job_id = job_id; }
    
    // Get worker ID
    std::string get_worker_id() const { return worker_id; }
    
    // Set worker ID
    void set_worker_id(const std::string& id) { worker_id = id; }
    
    // ===== SOLUTION SUBMISSION =====
    
    // Submit a solution to the xnt-core node via RPC
    bool submit_solution(
        const std::string& job_id,
        const Pow& pow_solution,
        const Digest& solution_hash);
    
    // Submit hashrate to node (not used in RPC mode)
    bool submit_hashrate_to_node(double hashrate_hs) { return true; }
    
    // ===== TEMPLATE MANAGEMENT =====
    
    // Get block template from RPC
    json getBlockTemplate();
    
    // Check if we have a cached template
    bool has_cached_puzzle() const { return has_last_template; }
    
    // Get cached template
    json get_cached_puzzle() const { return last_template; }
    
    // Cache a template
    void cache_puzzle(const json& template_obj);
    
    // Clear template cache
    void clear_puzzle_cache();
    
    // Get RPC client (for direct access if needed)
    XntRpcClient* get_rpc_client() const { return rpc_client.get(); }
};

// ===== PUZZLE PARSING FUNCTIONS =====

// Parse a PowPuzzle from JSON string
PowPuzzle parsePowPuzzle(const std::string& jsonStr);

// Parse a PowPuzzle from JSON object
PowPuzzle parsePowPuzzle(const json& j);

// Convert AuthPaths to PowMastPaths structure

// ===== SOLUTION SUBMISSION FUNCTIONS =====

// Submit a POW solution to the xnt-core node
// Returns SubmissionResult with success/failure status
SubmissionResult submitPowSolutionToNode(
    const std::string& proposal_id, 
    const Pow& pow_solution,
    const Digest& solution_hash);

// Submit solution through a specific client instance
bool submit_solution_to_node(
    const Pow& pow_solution, 
    const std::string& puzzle_id, 
    const Digest& solution_hash, 
    NeptuneCudaMinerClient* client);

// ===== GLOBAL CLIENT POINTER =====
// Global client pointer for solution submission from anywhere
extern NeptuneCudaMinerClient* g_neptune_miner_client;

// ===== UTILITY FUNCTIONS =====

// Convert digest to JSON limb array format
// Format: [limb0, limb1, limb2, limb3, limb4]
json digest_to_limb_array(const Digest& digest);

// Convert Merkle path to JSON array format
json merkle_path_to_json_array(const Digest* path, size_t height);

// ===== RECONNECTION SUPPORT =====


// Total solutions found across all sessions
extern uint64_t total_solutions_found;

#endif // XNT_NETWORK_CUH