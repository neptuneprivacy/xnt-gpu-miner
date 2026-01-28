#ifndef XNT_RPC_CLIENT_CUH
#define XNT_RPC_CLIENT_CUH

#include "common.cuh"
#include "network.cuh"
#include <string>
#include <chrono>
#include <mutex>
#include <atomic>

class Pow;
struct Digest;

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

struct RpcConfig {
    std::string url;
    std::string user;
    std::string password;
    std::string cookie_file;
    int timeout_sec;
    int poll_interval_sec;
    
    RpcConfig() 
        : url("http://127.0.0.1:9897")
        , timeout_sec(30)
        , poll_interval_sec(5) {}
};

class XntRpcClient {
private:
    RpcConfig config;
    std::atomic<int> request_id_counter;
    mutable RpcError last_error;
    mutable std::string last_error_message;
    mutable std::mutex rpc_mutex;  // Thread-safety for RPC calls
    
    bool send_http_post(const std::string& url,
                        const std::string& body,
                        std::string& response,
                        int timeout_sec);
    
    bool make_rpc_request(const std::string& method, 
                          const json& params, 
                          json& response);
    
    std::string build_auth_header() const;
    std::string load_cookie() const;
    RpcError parse_rpc_error(const json& response) const;
    std::string get_error_message(const json& response) const;

public:
    XntRpcClient(const RpcConfig& cfg = RpcConfig());
    XntRpcClient(const std::string& url,
                 const std::string& user = "",
                 const std::string& password = "");
    
    ~XntRpcClient() = default;
    
    XntRpcClient(const XntRpcClient&) = delete;
    XntRpcClient& operator=(const XntRpcClient&) = delete;
    
    void set_config(const RpcConfig& cfg) { config = cfg; }
    const RpcConfig& get_config() const { return config; }
    
    json getBlockTemplate(const std::string& guesser_address);
    json submitBlock(const json& template_obj, const json& pow);
    
    uint64_t getChainHeight();
    std::string getTipDigest();
    json getTipHeader();
    
    bool testConnection();
    
    RpcError getLastError() const { return last_error; }
    std::string getLastErrorMessage() const { return last_error_message; }
};

PowPuzzle parseRpcTemplate(const json& template_response);
json powToRpcFormat(const Pow& pow_solution, const Digest& solution_hash);

#endif
