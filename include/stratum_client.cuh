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

// StratumPowMastPaths structure matching pool schema (Job.paths in schema.rs)
// This is separate from PowMastPaths in pow.cuh which uses Digest types for GPU computation
struct StratumPowMastPaths {
    std::vector<std::string> pow_kernel_body;      // pow.kernel_body
    std::vector<std::string> pow_type_scripts;     // pow.type_scripts
    std::vector<std::string> pow_kernel;           // pow.kernel
    std::vector<std::string> header_body;          // header.body
    std::vector<std::string> header_appendix;      // header.appendix
    std::vector<std::string> kernel;               // kernel
};

// Stratum job structure matching pool schema (Job in schema.rs)
struct StratumJob {
    std::string job_id;              // id: Digest (hex string)
    StratumPowMastPaths paths;       // paths: PowMastPaths (from schema.rs)
    std::string difficulty;          // difficulty: String
    bool clean_jobs;                 // If true, discard previous jobs (implicit on new job)
    
    StratumJob() : clean_jobs(true) {}
    
    bool is_valid() const {
        return !job_id.empty() && !difficulty.empty();
    }
    
    // Convert to PowPuzzle for compatibility with existing mining code
    PowPuzzle to_pow_puzzle() const {
        PowPuzzle puzzle;
        puzzle.id = job_id;
        puzzle.threshold = difficulty;
        // Map StratumPowMastPaths to legacy AuthPaths format
        puzzle.auth_paths.pow = paths.pow_kernel_body;
        puzzle.auth_paths.pow.insert(puzzle.auth_paths.pow.end(), 
            paths.pow_type_scripts.begin(), paths.pow_type_scripts.end());
        puzzle.auth_paths.pow.insert(puzzle.auth_paths.pow.end(),
            paths.pow_kernel.begin(), paths.pow_kernel.end());
        puzzle.auth_paths.header = paths.header_body;
        puzzle.auth_paths.header.insert(puzzle.auth_paths.header.end(),
            paths.header_appendix.begin(), paths.header_appendix.end());
        puzzle.auth_paths.kernel = paths.kernel;
        puzzle.consensus_rule_set = CONSENSUS_XNT;
        return puzzle;
    }
};

// Stratum client configuration (matches pool Login request)
struct StratumConfig {
    std::string host;
    int port;
    bool use_ssl;               // Use SSL/TLS for stratum connection
    std::string name;           // Worker name (login.name)
    std::string address;        // Wallet address (login.address)
    std::string password;       // Pool password (optional, login.password)
    std::string agent;          // Miner agent string (login.agent)
    int connect_timeout_sec;
    int read_timeout_sec;
    int reconnect_delay_sec;
    int max_reconnect_attempts;
    int keepalive_interval_sec; // Interval for keepalive requests
    
    StratumConfig()
        : port(3333)
        , use_ssl(false)
        , name("default")
        , password("")
        , agent("xnt-gpu-miner/1.0")
        , connect_timeout_sec(30)
        , read_timeout_sec(60)
        , reconnect_delay_sec(5)
        , max_reconnect_attempts(10)
        , keepalive_interval_sec(30) {}
};

// Forward declarations for OpenSSL
struct ssl_ctx_st;
struct ssl_st;
typedef ssl_ctx_st SSL_CTX;
typedef ssl_st SSL;

// TCP-based Stratum client implementing pool schema protocol
class StratumClient : public MiningClient {
private:
    StratumConfig config;
    socket_t sock;
    bool use_ssl{false};
    SSL_CTX* ssl_ctx{nullptr};
    SSL* ssl{nullptr};
    std::atomic<bool> connected{false};
    std::atomic<bool> logged_in{false};
    std::atomic<bool> running{false};
    
    // Receive thread for async notifications
    std::thread receive_thread;
    
    // Keepalive thread
    std::thread keepalive_thread;
    
    // Job queue (thread-safe)
    std::queue<StratumJob> job_queue;
    std::mutex job_mutex;
    std::condition_variable job_cv;
    
    // Current job (for getBlockTemplate compatibility)
    StratumJob current_job;
    std::mutex current_job_mutex;
    
    // Request ID counter for JSON-RPC
    std::atomic<uint64_t> request_id_counter{1};
    
    // Pending responses (for request-response matching)
    std::map<uint64_t, std::promise<json>> pending_requests;
    std::mutex pending_mutex;
    
    // Error tracking
    mutable StratumError last_error{StratumError::None};
    mutable std::string last_error_message;
    
    // Worker ID from login response
    size_t worker_id{0};
    
    // Track difficulty for stratum-v1 pools (mining.set_difficulty)
    std::string current_difficulty;
    
    // Internal methods
    bool connect_tcp();
    bool init_ssl();
    void cleanup_ssl();
    void disconnect_tcp();
    void receive_loop();
    void keepalive_loop();
    bool send_json(const json& message);
    bool send_line(const std::string& line);
    std::string read_line(int timeout_ms = 5000);
    void handle_message(const json& message);
    void handle_notification(const std::string& method, const json& params);
    void handle_response(uint64_t id, const json& result, const json& error);
    
    // Pool protocol methods
    bool do_login();
    bool do_stratum_v1_login();
    
    // Parse job notification from pool
    StratumJob parse_job_notification(const json& params);
    
public:
    StratumClient(const StratumConfig& cfg);
    StratumClient(const std::string& url, const std::string& address, 
                  const std::string& worker_name = "default",
                  const std::string& password = "");
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
    
    // Pool-specific methods
    bool is_logged_in() const { return logged_in.load(); }
    size_t get_worker_id() const { return worker_id; }
    
    // Send keepalive (can be called manually if needed)
    bool send_keepalive();
    
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
//   stratum+ssl://host:port (SSL not yet implemented, will use TCP)
//   tcp://host:port
//   host:port
bool parse_stratum_url(const std::string& url, std::string& host, int& port, bool& use_ssl);

#endif // XNT_STRATUM_CLIENT_CUH
