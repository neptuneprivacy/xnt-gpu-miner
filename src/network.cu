#include "network.cuh"
#include "rpc_client.cuh"
#include "stratum_client.cuh"
#include "pow.cuh"
#include "digest.cuh"
#include "common.cuh"

NeptuneCudaMinerClient::NeptuneCudaMinerClient(
    const std::string& rpc_url,
    const std::string& wallet_addr)
    : rpc_url(rpc_url)
    , has_last_template(false)
    , wallet_address(wallet_addr) {
    RpcConfig config;
    config.url = rpc_url;
    config.timeout_sec = 30;
    config.poll_interval_sec = g_fetch_interval_sec;
    rpc_client = std::make_unique<XntRpcClient>(config);
}

NeptuneCudaMinerClient::~NeptuneCudaMinerClient() {
}

bool NeptuneCudaMinerClient::connect_to_node() {
    if (!rpc_client) {
        return false;
    }
    return rpc_client->testConnection();
}

bool NeptuneCudaMinerClient::is_connected() const {
    if (!rpc_client) {
        return false;
    }
    
    return rpc_client->testConnection();
}

json NeptuneCudaMinerClient::getBlockTemplate() {
    if (!rpc_client || wallet_address.empty()) {
        return json();
    }
    
    return rpc_client->getBlockTemplate(wallet_address);
}

bool NeptuneCudaMinerClient::submit_solution(
    const std::string& proposal_id,
    const Pow& pow_solution,
    const Digest& solution_hash,
    const json& template_obj) {
    
    if (!rpc_client) {
        return false;
    }
    
    if (template_obj.is_null() || template_obj.empty()) {
        std::cout << "[SUBMIT] ERROR: No template provided for proposal " << proposal_id << std::endl;
        return false;
    }
    std::string tip_digest = rpc_client->getTipDigest();
    
    // Check if template is stale
    if (template_obj.contains("metadata") && !template_obj["metadata"].is_null()) {
        json metadata = template_obj["metadata"];
        std::string prev_block;
        if (metadata.contains("prevBlock") && !metadata["prevBlock"].is_null()) {
            prev_block = metadata.value("prevBlock", "");
        } else if (metadata.contains("prev_block") && !metadata["prev_block"].is_null()) {
            prev_block = metadata.value("prev_block", "");
        }

        if (!prev_block.empty() && tip_digest != prev_block) {
            std::string short_prev = prev_block.length() > 20 ? prev_block.substr(0, 12) + "..." + prev_block.substr(prev_block.length() - 8) : prev_block;
            std::string short_tip = tip_digest.length() > 20 ? tip_digest.substr(0, 12) + "..." + tip_digest.substr(tip_digest.length() - 8) : tip_digest;
            std::cout << "[SUBMIT] " << Color::RED << "✗ REJECTED: Template stale (prev_block=" << short_prev << " != tip=" << short_tip << ")" << Color::RESET << std::endl;
            return false;
        }
    }
    
    // Extract the block from template (RpcBlockTemplate has "block" and "metadata" fields)
    // SubmitBlockRequest expects: { template: RpcBlock, pow: RpcBlockPow }
    if (!template_obj.contains("block") || template_obj["block"].is_null()) {
        std::cout << "[SUBMIT] ERROR: Template missing 'block' field" << std::endl;
        return false;
    }
    
    json block_obj = template_obj["block"];
    
    // Ensure block has kernel.appendix with required structure
    // RpcBlockAppendix is serialized as an array of RpcClaim objects
    if (!block_obj.contains("kernel") || block_obj["kernel"].is_null()) {
        std::cout << "[SUBMIT] ERROR: Block missing 'kernel' field" << std::endl;
        return false;
    }
    
    json kernel_obj = block_obj["kernel"];
    
    // Check if appendix exists, if not create empty array
    // If it exists but is not an array, ensure it's properly structured
    if (!kernel_obj.contains("appendix") || kernel_obj["appendix"].is_null()) {
        std::cout << "[SUBMIT] WARNING: Block kernel missing 'appendix', adding empty array" << std::endl;
        kernel_obj["appendix"] = json::array();
        block_obj["kernel"] = kernel_obj;
    } else if (!kernel_obj["appendix"].is_array()) {
        std::cout << "[SUBMIT] WARNING: Block kernel 'appendix' is not an array, fixing structure" << std::endl;
        kernel_obj["appendix"] = json::array();
        block_obj["kernel"] = kernel_obj;
    }
    
    json pow_json = powToRpcFormat(pow_solution, solution_hash);
    
    json response = rpc_client->submitBlock(block_obj, pow_json);
    
    if (!response.empty() && response.contains("result")) {
        if (response["result"].is_boolean()) {
            bool success = response["result"].get<bool>();
            if (!success) {
                std::cout << "[SUBMIT] " << Color::RED << "✗ REJECTED: Block rejected" << Color::RESET << std::endl;
            }
            return success;
        }
        if (response["result"].contains("success")) {
            bool success = response["result"]["success"].get<bool>();
            if (!success) {
                std::cout << "[SUBMIT] " << Color::RED << "✗ REJECTED: Block rejected" << Color::RESET << std::endl;
            }
            return success;
        }
        return true;
    }
    
    if (response.contains("error") && !response["error"].is_null()) {
        json error = response["error"];
        std::string error_reason = "Unknown error";
        
        // Extract exact error reason from error data
        if (error.contains("data") && !error["data"].is_null()) {
            json error_data = error["data"];
            if (error_data.is_object() && !error_data.empty()) {
                // Get the first value from the error data object
                auto it = error_data.begin();
                if (it != error_data.end() && it.value().is_string()) {
                    error_reason = it.value().get<std::string>();
                } else if (it != error_data.end()) {
                    error_reason = it.value().dump();
                }
            } else if (error_data.is_string()) {
                error_reason = error_data.get<std::string>();
            }
        } else if (error.contains("message") && !error["message"].is_null()) {
            error_reason = error.value("message", "Unknown error");
        }
        
        std::cout << "[SUBMIT] " << Color::RED << "✗ REJECTED: " << error_reason << Color::RESET << std::endl;
    } else {
        std::cout << "[SUBMIT] " << Color::RED << "✗ REJECTED: No response from server" << Color::RESET << std::endl;
    }
    
    return false;
}

void NeptuneCudaMinerClient::cache_puzzle(const json& template_obj) {
    last_template = template_obj;
    has_last_template = true;
}

PowPuzzle parsePowPuzzle(const std::string& jsonStr) {
    try {
        json j = json::parse(jsonStr);
        if (j.contains("result") && j["result"].contains("template")) {
            return parseRpcTemplate(j);
        }
        return parseRpcTemplate(j);
    } catch (const std::exception& e) {
        LOG_DEBUG("Failed to parse puzzle JSON: " << e.what());
        return PowPuzzle();
    }
}

// ===== UnifiedMinerClient Implementation =====

UnifiedMinerClient::UnifiedMinerClient(
    const std::string& endpoint,
    const std::string& wallet_addr,
    const std::string& stratum_pass)
    : endpoint(endpoint)
    , wallet_address(wallet_addr)
    , stratum_password(stratum_pass) {
    mode = detect_mining_mode(endpoint);
}

bool UnifiedMinerClient::initialize() {
    if (client) {
        return true;  // Already initialized
    }
    
    if (mode == MiningMode::Stratum) {
        // Create stratum client
        std::string host;
        int port;
        bool use_ssl = false;
        if (!parse_stratum_url(endpoint, host, port, use_ssl)) {
            std::cerr << Color::RED << "Invalid stratum URL: " << endpoint << Color::RESET << std::endl;
            return false;
        }
        
        StratumConfig config;
        config.host = host;
        config.port = port;
        config.use_ssl = use_ssl;
        config.address = wallet_address;
        config.name = "xnt-miner";
        config.password = stratum_password;
        
        client = std::make_unique<StratumClient>(config);
        
        std::cout << "Initialized stratum client for " << host << ":" << port << std::endl;
    } else {
        // Create solo (HTTP RPC) client
        client = std::make_unique<NeptuneCudaMinerClient>(endpoint, wallet_address);
        std::cout << "Initialized solo mining client for " << endpoint << std::endl;
    }
    
    return client != nullptr;
}


