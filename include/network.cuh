#ifndef XNT_NETWORK_CUH
#define XNT_NETWORK_CUH

#include "common.cuh"
#include "pow.cuh"
#include "mining_client.h"

class Pow;
struct Digest;
class XntRpcClient;
class StratumClient;

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

// Solo mining client (HTTP JSON-RPC) implementing MiningClient interface
class NeptuneCudaMinerClient : public MiningClient {
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
    
    // MiningClient interface implementation
    bool connect() override { return connect_to_node(); }
    void disconnect() override { /* HTTP is stateless, nothing to disconnect */ }
    bool is_connected() const override;
    MiningMode get_mode() const override { return MiningMode::Solo; }
    
    bool submit_solution(
        const std::string& proposal_id,
        const Pow& pow_solution,
        const Digest& solution_hash,
        const json& template_obj) override;
    
    json getBlockTemplate() override;
    // Also support the new interface with optional parameter
    json getBlockTemplate(const std::string& wallet_address) {
        (void)wallet_address;  // Not used in solo mode
        return getBlockTemplate();
    }
    
    // Solo-specific methods
    bool connect_to_node();
    void cache_puzzle(const json& template_obj);
    XntRpcClient* get_rpc_client() const { return rpc_client.get(); }
};

// Unified mining client that wraps either Solo or Stratum client
class UnifiedMinerClient {
private:
    std::unique_ptr<MiningClient> client;
    MiningMode mode;
    std::string endpoint;
    std::string wallet_address;
    std::string stratum_password;
    
public:
    UnifiedMinerClient(
        const std::string& endpoint,
        const std::string& wallet_addr,
        const std::string& stratum_pass = "x");
    
    ~UnifiedMinerClient() = default;
    
    UnifiedMinerClient(const UnifiedMinerClient&) = delete;
    UnifiedMinerClient& operator=(const UnifiedMinerClient&) = delete;
    
    // Initialize client based on detected mode
    bool initialize();
    
    // Delegated methods
    bool connect() { return client ? client->connect() : false; }
    void disconnect() { if (client) client->disconnect(); }
    bool is_connected() const { return client ? client->is_connected() : false; }
    json getBlockTemplate() { return client ? client->getBlockTemplate() : json(); }
    
    bool submit_solution(
        const std::string& proposal_id,
        const Pow& pow_solution,
        const Digest& solution_hash,
        const json& template_obj) {
        return client ? client->submit_solution(proposal_id, pow_solution, solution_hash, template_obj) : false;
    }
    
    bool wait_for_job(json& job, int timeout_ms = 5000) {
        return client ? client->wait_for_job(job, timeout_ms) : false;
    }
    
    MiningMode get_mode() const { return mode; }
    bool is_push_based() const { return client ? client->is_push_based() : false; }
    MiningClient* get_client() { return client.get(); }
    
    // For backward compatibility with existing code expecting NeptuneCudaMinerClient
    NeptuneCudaMinerClient* get_solo_client() {
        if (mode == MiningMode::Solo) {
            return dynamic_cast<NeptuneCudaMinerClient*>(client.get());
        }
        return nullptr;
    }
};

PowPuzzle parsePowPuzzle(const std::string& jsonStr);

#endif
