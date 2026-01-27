#ifndef XNT_NETWORK_CUH
#define XNT_NETWORK_CUH

#include "common.cuh"
#include "pow.cuh"

class Pow;
struct Digest;
class XntRpcClient;

struct PowPuzzle {
    std::string id;
    std::string threshold;
    std::string total_guesser_reward;
    std::string prev_block;
    AuthPaths auth_paths;
    int consensus_rule_set;
    
    PowPuzzle() : consensus_rule_set(CONSENSUS_XNT) {}
    
    bool is_valid() const {
        return !id.empty() && !threshold.empty() && 
               !auth_paths.pow.empty() && !auth_paths.header.empty() && !auth_paths.kernel.empty();
    }
};

class NeptuneCudaMinerClient {
private:
    std::unique_ptr<XntRpcClient> rpc_client;
    std::string rpc_url;
    json last_template;
    bool has_last_template;
    std::string wallet_address;

public:
    NeptuneCudaMinerClient(
        const std::string& rpc_url = "http://127.0.0.1:9897",
        const std::string& wallet_addr = "");
    
    ~NeptuneCudaMinerClient();
    
    NeptuneCudaMinerClient(const NeptuneCudaMinerClient&) = delete;
    NeptuneCudaMinerClient& operator=(const NeptuneCudaMinerClient&) = delete;
    
    bool connect_to_node();
    bool is_connected() const;
    
    bool submit_solution(
        const std::string& proposal_id,
        const Pow& pow_solution,
        const Digest& solution_hash,
        const json& template_obj);
    
    json getBlockTemplate();
    void cache_puzzle(const json& template_obj);
    XntRpcClient* get_rpc_client() const { return rpc_client.get(); }
};

PowPuzzle parsePowPuzzle(const std::string& jsonStr);

#endif
