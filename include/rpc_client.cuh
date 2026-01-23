#ifndef XNT_RPC_CLIENT_CUH
#define XNT_RPC_CLIENT_CUH

#include "common.cuh"
#include "network.cuh"
#include <string>
#include <chrono>

// Forward declarations
class Pow;
struct Digest;

// ===== RPC ERROR TYPES =====
enum class RpcError {
    None,
    ConnectionFailed,
    Timeout,
    InvalidResponse,
    AuthenticationFailed,
    InvalidBlock,
    InsufficientWork,
    Unknown
};

// ===== RPC CLIENT CONFIGURATION =====
struct RpcConfig {
    std::string url;              // Full URL: http://host:port
    std::string user;             // Basic auth username (optional)
    std::string password;         // Basic auth password (optional)
    std::string cookie_file;      // Cookie file path (optional)
    int timeout_sec;              // Request timeout in seconds
    int poll_interval_sec;        // Polling interval for templates
    
    RpcConfig() 
        : url("http://127.0.0.1:9897")
        , timeout_sec(30)
        , poll_interval_sec(5) {}
};

// ===== XNT RPC CLIENT =====
// HTTP JSON-RPC 2.0 client for xnt-core node communication
class XntRpcClient {
private:
    RpcConfig config;
    int request_id_counter;
    
    // HTTP client functionality
    bool send_http_post(const std::string& url,
                       const std::string& body,
                       std::string& response,
                       int timeout_sec);
    
    bool make_rpc_request(const std::string& method, 
                         const json& params, 
                         json& response);
    
    // Authentication helpers
    std::string build_auth_header() const;
    std::string load_cookie() const;
    
    // Error handling
    RpcError parse_rpc_error(const json& response) const;
    std::string get_error_message(const json& response) const;

public:
    // ===== CONSTRUCTORS =====
    XntRpcClient(const RpcConfig& cfg = RpcConfig());
    XntRpcClient(const std::string& url,
                 const std::string& user = "",
                 const std::string& password = "");
    
    ~XntRpcClient() = default;
    
    // Disable copying
    XntRpcClient(const XntRpcClient&) = delete;
    XntRpcClient& operator=(const XntRpcClient&) = delete;
    
    // ===== CONFIGURATION =====
    void set_config(const RpcConfig& cfg) { config = cfg; }
    const RpcConfig& get_config() const { return config; }
    
    // ===== MINING ENDPOINTS =====
    
    /**
     * Get block template for mining
     * @param guesser_address Wallet address to receive mining rewards
     * @return JSON response with template, or empty json on error
     */
    json getBlockTemplate(const std::string& guesser_address);
    
    /**
     * Submit mined block with proof-of-work solution
     * @param template Block template from getBlockTemplate
     * @param pow Proof-of-work solution structure
     * @return JSON response with success status, or empty json on error
     */
    json submitBlock(const json& template_obj, const json& pow);
    
    // ===== CHAIN ENDPOINTS =====
    
    /**
     * Get current blockchain height
     * @return Block height, or 0 on error
     */
    uint64_t getChainHeight();
    
    /**
     * Get hash of the current tip block
     * @return Tip block digest (hex string), or empty string on error
     */
    std::string getTipDigest();
    
    /**
     * Get tip block header
     * @return JSON response with header, or empty json on error
     */
    json getTipHeader();
    
    // ===== CONNECTION HEALTH =====
    
    /**
     * Test RPC connection
     * @return true if connection successful
     */
    bool testConnection();
    
    // ===== ERROR HANDLING =====
    
    /**
     * Get last error code
     */
    RpcError getLastError() const { return last_error; }
    
    /**
     * Get last error message
     */
    std::string getLastErrorMessage() const { return last_error_message; }

private:
    mutable RpcError last_error;
    mutable std::string last_error_message;
};

// ===== TEMPLATE PARSING =====

/**
 * Parse RPC template response to PowPuzzle structure
 * @param template_response JSON response from getBlockTemplate
 * @return PowPuzzle structure
 */
PowPuzzle parseRpcTemplate(const json& template_response);

/**
 * Convert Pow solution to RPC submission format
 * @param pow_solution Pow structure
 * @param solution_hash Solution hash digest
 * @return JSON structure for mining_submitBlock pow parameter
 */
json powToRpcFormat(const Pow& pow_solution, const Digest& solution_hash);

#endif // XNT_RPC_CLIENT_CUH
