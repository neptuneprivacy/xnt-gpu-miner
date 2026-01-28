#ifndef XNT_STRATUM_CLIENT_CUH
#define XNT_STRATUM_CLIENT_CUH

#include "common.cuh"
#include "mining_client.h"
#include "network.cuh"
#include <future>
#include <map>

class Pow;
struct Digest;

// Stratum protocol error codes
enum class StratumError {
    None,
    ConnectionFailed,
    ConnectionLost,
    Timeout,
    AuthenticationFailed,
    InvalidResponse,
    SubscribeFailed,
    SubmitRejected,
    Unknown
};

// Stratum job structure (parsed from mining.notify)
struct StratumJob {
    std::string job_id;
    std::string prev_block_hash;
    std::string threshold;           // Target difficulty
    std::string total_guesser_reward;
    AuthPaths auth_paths;
    int consensus_rule_set;
    uint64_t height;
    bool clean_jobs;                 // If true, discard previous jobs
    
    StratumJob() : consensus_rule_set(CONSENSUS_XNT), height(0), clean_jobs(false) {}
    
    bool is_valid() const {
        return !job_id.empty() && !prev_block_hash.empty() && !threshold.empty() &&
               !auth_paths.pow.empty() && !auth_paths.header.empty() && !auth_paths.kernel.empty();
    }
    
    // Convert to PowPuzzle for compatibility with existing mining code
    PowPuzzle to_pow_puzzle() const {
        PowPuzzle puzzle;
        puzzle.id = job_id;
        puzzle.threshold = threshold;
        puzzle.total_guesser_reward = total_guesser_reward;
        puzzle.prev_block = prev_block_hash;
        puzzle.auth_paths = auth_paths;
        puzzle.consensus_rule_set = consensus_rule_set;
        return puzzle;
    }
};

// Stratum client configuration
struct StratumConfig {
    std::string host;
    int port;
    std::string username;       // Wallet address or pool username
    std::string password;       // Pool password (optional)
    std::string worker_name;    // Worker identifier
    int connect_timeout_sec;
    int read_timeout_sec;
    int reconnect_delay_sec;
    int max_reconnect_attempts;
    
    StratumConfig()
        : port(3333)
        , password("x")
        , worker_name("xnt-miner")
        , connect_timeout_sec(30)
        , read_timeout_sec(60)
        , reconnect_delay_sec(5)
        , max_reconnect_attempts(10) {}
};

// TCP-based Stratum client implementing JSON-RPC over TCP
class StratumClient : public MiningClient {
private:
    StratumConfig config;
    socket_t sock;
    std::atomic<bool> connected{false};
    std::atomic<bool> authorized{false};
    std::atomic<bool> running{false};
    
    // Receive thread for async notifications
    std::thread receive_thread;
    
    // Job queue (thread-safe)
    std::queue<StratumJob> job_queue;
    std::mutex job_mutex;
    std::condition_variable job_cv;
    
    // Current job (for getBlockTemplate compatibility)
    StratumJob current_job;
    std::mutex current_job_mutex;
    
    // Request ID counter for JSON-RPC
    std::atomic<int> request_id_counter{1};
    
    // Pending responses (for request-response matching)
    std::map<int, std::promise<json>> pending_requests;
    std::mutex pending_mutex;
    
    // Error tracking
    mutable StratumError last_error{StratumError::None};
    mutable std::string last_error_message;
    
    // Subscription data
    std::string session_id;
    std::string extranonce1;
    int extranonce2_size;
    double current_difficulty;
    
    // Internal methods
    bool connect_tcp();
    void disconnect_tcp();
    void receive_loop();
    bool send_json(const json& message);
    bool send_line(const std::string& line);
    std::string read_line(int timeout_ms = 5000);
    void handle_message(const json& message);
    void handle_notification(const std::string& method, const json& params);
    void handle_response(int id, const json& result, const json& error);
    
    // Stratum protocol methods
    bool do_subscribe();
    bool do_authorize();
    
    // Parse stratum job notification
    StratumJob parse_job_notification(const json& params);
    
public:
    StratumClient(const StratumConfig& cfg);
    StratumClient(const std::string& url, const std::string& username, 
                  const std::string& password = "x");
    ~StratumClient() override;
    
    // Disable copy
    StratumClient(const StratumClient&) = delete;
    StratumClient& operator=(const StratumClient&) = delete;
    
    // MiningClient interface implementation
    bool connect() override;
    void disconnect() override;
    bool is_connected() const override;
    json getBlockTemplate() override;
    bool submit_solution(
        const std::string& proposal_id,
        const Pow& pow_solution,
        const Digest& solution_hash,
        const json& template_obj) override;
    MiningMode get_mode() const override { return MiningMode::Stratum; }
    bool wait_for_job(json& job, int timeout_ms = 5000) override;
    
    // Stratum-specific methods
    bool is_authorized() const { return authorized.load(); }
    double get_difficulty() const { return current_difficulty; }
    const std::string& get_session_id() const { return session_id; }
    
    // Error handling
    StratumError get_last_error() const { return last_error; }
    const std::string& get_last_error_message() const { return last_error_message; }
    
    // Statistics
    std::atomic<uint64_t> shares_submitted{0};
    std::atomic<uint64_t> shares_accepted{0};
    std::atomic<uint64_t> shares_rejected{0};
};

// Parse stratum URL into host and port
// Supported formats:
//   stratum://host:port
//   stratum+tcp://host:port
//   tcp://host:port
//   host:port
bool parse_stratum_url(const std::string& url, std::string& host, int& port);

#endif // XNT_STRATUM_CLIENT_CUH
