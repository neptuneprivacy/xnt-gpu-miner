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
    if (template_obj.contains("metadata") && !template_obj["metadata"].is_null()) {
        json metadata = template_obj["metadata"];
        if (metadata.contains("prevBlock") && !metadata["prevBlock"].is_null()) {
            std::string prev_block = metadata.value("prevBlock", "");
        if (tip_digest != prev_block) {
            return false;
            }
        }
    }
    
    json pow_json = powToRpcFormat(pow_solution, solution_hash);
    json response = rpc_client->submitBlock(template_obj, pow_json);
    
    if (!response.empty() && response.contains("result")) {
        if (response["result"].is_boolean()) {
            return response["result"].get<bool>();
        }
        if (response["result"].contains("success")) {
            return response["result"]["success"].get<bool>();
        }
        return true;
    }
    
    if (response.contains("error") && !response["error"].is_null()) {
        json error = response["error"];
        std::string error_msg = (error.contains("message") && !error["message"].is_null())
            ? error.value("message", "Unknown error") : "Unknown error";
        LOG_DEBUG("Submission error: " << error_msg);
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


