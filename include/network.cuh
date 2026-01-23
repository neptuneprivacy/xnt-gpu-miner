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

// ===== NEPTUNE CUDA MINER CLIENT =====
// RPC-based client for communication with xnt-core node
// Uses JSON-RPC 2.0 over HTTP
class NeptuneCudaMinerClient {
private:
    // RPC client for node communication
    std::unique_ptr<XntRpcClient> rpc_client;
    
    // Node connection info (RPC URL for xnt-core node)
    std::string rpc_url;
    
    // Cached template from RPC (for solution submission)
    json last_template;
    bool has_last_template;
    
    // Wallet address for getBlockTemplate calls
    std::string wallet_address;

public:
    // ===== CONSTRUCTORS/DESTRUCTOR =====
    
    NeptuneCudaMinerClient(
        const std::string& rpc_url = "http://127.0.0.1:9897",
        const std::string& wallet_addr = "");
    
    ~NeptuneCudaMinerClient();
    
    // Disable copying
    NeptuneCudaMinerClient(const NeptuneCudaMinerClient&) = delete;
    NeptuneCudaMinerClient& operator=(const NeptuneCudaMinerClient&) = delete;
    
    // ===== CONNECTION MANAGEMENT =====
    
    // Connect to the xnt-core node (test RPC connection)
    bool connect_to_node();
    
    // Check if connected to node (test RPC connection)
    bool is_connected() const;
    
    // ===== SOLUTION SUBMISSION =====
    
    // Submit a solution to the xnt-core node via RPC
    bool submit_solution(
        const std::string& proposal_id,
        const Pow& pow_solution,
        const Digest& solution_hash);
    
    // ===== TEMPLATE MANAGEMENT =====
    
    // Get block template from RPC
    json getBlockTemplate();
    
    // Cache a template for solution submission
    void cache_puzzle(const json& template_obj);
    
    // Get RPC client (for direct access if needed)
    XntRpcClient* get_rpc_client() const { return rpc_client.get(); }
};

// ===== PUZZLE PARSING FUNCTIONS =====

// Parse a PowPuzzle from JSON string (used for event data)
PowPuzzle parsePowPuzzle(const std::string& jsonStr);

// ===== GLOBAL CLIENT POINTER =====
// Global client pointer for solution submission from anywhere
extern NeptuneCudaMinerClient* g_neptune_miner_client;

// Total solutions found across all sessions
extern uint64_t total_solutions_found;

#endif // XNT_NETWORK_CUH