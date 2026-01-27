#include "network.cuh"
#include "rpc_client.cuh"
#include "pow.cuh"
#include "digest.cuh"

NeptuneCudaMinerClient::NeptuneCudaMinerClient(
    const std::string& rpc_url,
    const std::string& wallet_addr)
    : rpc_url(rpc_url)
    , has_last_template(false)
    , wallet_address(wallet_addr) {
    RpcConfig config;
    config.url = rpc_url;
    config.timeout_sec = 30;
    config.poll_interval_sec = 5;
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
    const Digest& solution_hash) {
    
    if (!rpc_client) {
        return false;
    }
    
    if (!has_last_template) {
        return false;
    }
    
    json template_obj = last_template;
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
            std::cout << "[SUBMIT] Template stale: prev_block=" << short_prev << " != tip=" << short_tip << std::endl;
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
    json pow_json = powToRpcFormat(pow_solution, solution_hash);
    
    // Log submission details
    std::string short_id = proposal_id.length() > 20 ? proposal_id.substr(0, 12) + "..." + proposal_id.substr(proposal_id.length() - 8) : proposal_id;
    std::cout << "[SUBMIT] Submitting solution for proposal: " << short_id << std::endl;
    
    json response = rpc_client->submitBlock(block_obj, pow_json);
    
    if (!response.empty() && response.contains("result")) {
        if (response["result"].is_boolean()) {
            bool success = response["result"].get<bool>();
            std::cout << "[SUBMIT] " << (success ? "✓ ACCEPTED" : "✗ REJECTED") << std::endl;
            return success;
        }
        if (response["result"].contains("success")) {
            bool success = response["result"]["success"].get<bool>();
            std::cout << "[SUBMIT] " << (success ? "✓ ACCEPTED" : "✗ REJECTED") << std::endl;
            return success;
        }
        std::cout << "[SUBMIT] ✓ ACCEPTED (result present)" << std::endl;
        return true;
    }
    
    if (response.contains("error") && !response["error"].is_null()) {
        json error = response["error"];
        std::string error_msg = (error.contains("message") && !error["message"].is_null())
            ? error.value("message", "Unknown error") : "Unknown error";
        std::string error_code = (error.contains("code") && !error["code"].is_null())
            ? std::to_string(error["code"].get<int>()) : "unknown";
        std::cout << "[SUBMIT] ✗ REJECTED: " << error_msg << " (code=" << error_code << ")" << std::endl;
        if (error.contains("data") && !error["data"].is_null()) {
            std::cout << "[SUBMIT]   Details: " << error["data"].dump() << std::endl;
        }
    } else {
        std::cout << "[SUBMIT] ✗ REJECTED: No response from server" << std::endl;
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


