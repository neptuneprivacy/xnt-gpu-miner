#include "rpc_client.cuh"
#include "network.cuh"
#include "pow.cuh"
#include "digest.cuh"
#include <sstream>
#include <fstream>
#include <algorithm>

static bool http_post(const std::string& url,
                     const std::string& body,
                     const std::string& auth_header,
                     std::string& response,
                     int timeout_sec) {
    std::string protocol, host, path;
    int port = 80;
    
    size_t protocol_end = url.find("://");
    if (protocol_end == std::string::npos) {
        return false;
    }
    protocol = url.substr(0, protocol_end);
    std::string rest = url.substr(protocol_end + 3);
    
    if (protocol == "https") {
        return false;
    }
    
    size_t path_start = rest.find('/');
    if (path_start != std::string::npos) {
        host = rest.substr(0, path_start);
        path = rest.substr(path_start);
    } else {
        host = rest;
        path = "/";
    }
    
    size_t port_start = host.find(':');
    if (port_start != std::string::npos) {
        try {
            port = std::stoi(host.substr(port_start + 1));
            host = host.substr(0, port_start);
        } catch (...) {
            return false;
        }
    }
    
    socket_t sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock == INVALID_SOCKET_VALUE) {
        return false;
    }
    
    struct hostent* host_entry = gethostbyname(host.c_str());
    if (!host_entry) {
        close(sock);
        return false;
    }
    
    struct sockaddr_in server_addr;
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(port);
    memcpy(&server_addr.sin_addr, host_entry->h_addr, host_entry->h_length);
    
    if (connect(sock, (struct sockaddr*)&server_addr, sizeof(server_addr)) == SOCKET_ERROR_VALUE) {
        close(sock);
        return false;
    }
    
    #ifdef _WIN32
        DWORD timeout = timeout_sec * 1000;
        setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, (const char*)&timeout, sizeof(timeout));
        setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, (const char*)&timeout, sizeof(timeout));
    #else
        struct timeval tv;
        tv.tv_sec = timeout_sec;
        tv.tv_usec = 0;
        setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
        setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
    #endif
    
    std::ostringstream request;
    request << "POST " << path << " HTTP/1.1\r\n";
    request << "Host: " << host << ":" << port << "\r\n";
    request << "Content-Type: application/json\r\n";
    request << "Content-Length: " << body.length() << "\r\n";
    if (!auth_header.empty()) {
        request << "Authorization: " << auth_header << "\r\n";
    }
    request << "Connection: close\r\n";
    request << "\r\n";
    request << body;
    
    std::string request_str = request.str();
    
    if (send(sock, request_str.c_str(), request_str.length(), 0) == SOCKET_ERROR_VALUE) {
        close(sock);
        return false;
    }
    
    response.clear();
    char buffer[4096];
    ssize_t received;
    while ((received = recv(sock, buffer, sizeof(buffer) - 1, 0)) > 0) {
        buffer[received] = '\0';
        response += buffer;
        
        // Check if we've received the complete response
        size_t header_end = response.find("\r\n\r\n");
        if (header_end != std::string::npos) {
            // Headers received, check Content-Length
            std::string headers = response.substr(0, header_end);
            size_t content_length_pos = headers.find("Content-Length:");
            if (content_length_pos != std::string::npos) {
                size_t len_start = content_length_pos + 15;
                while (len_start < headers.length() && headers[len_start] == ' ') len_start++;
                size_t len_end = len_start;
                while (len_end < headers.length() && headers[len_end] >= '0' && headers[len_end] <= '9') len_end++;
                if (len_end > len_start) {
                    try {
                        int content_length = std::stoi(headers.substr(len_start, len_end - len_start));
                        size_t body_start = header_end + 4;
                        if (response.length() - body_start >= static_cast<size_t>(content_length)) {
                            break; // Received complete response
                        }
                    } catch (...) {
                        // If parsing fails, continue reading
                    }
                }
            }
        }
    }
    
    // recv returns 0 when connection is closed (normal)
    // recv returns -1 on error (timeout or other error)
    if (received < 0) {
        close(sock);
        return false;
    }
    
    close(sock);
    
    size_t header_end = response.find("\r\n\r\n");
    if (header_end == std::string::npos) {
        return false;
    }
    
    size_t status_line_end = response.find("\r\n");
    if (status_line_end == std::string::npos) {
        return false;
    }
    
    std::string status_line = response.substr(0, status_line_end);
    
    if (status_line.find("HTTP/1.1 200") == std::string::npos &&
        status_line.find("HTTP/1.0 200") == std::string::npos) {
        return false;
    }
    
    response = response.substr(header_end + 4);
    
    return true;
}

// Base64 encode (simple implementation)
static std::string base64_encode_simple(const std::string& input) {
    const char base64_chars[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string encoded;
    int val = 0, valb = -6;
    
    for (unsigned char c : input) {
        val = (val << 8) + c;
        valb += 8;
        while (valb >= 0) {
            encoded.push_back(base64_chars[(val >> valb) & 0x3F]);
            valb -= 6;
        }
    }
    
    if (valb > -6) {
        encoded.push_back(base64_chars[((val << 8) >> (valb + 8)) & 0x3F]);
    }
    
    while (encoded.size() % 4) {
        encoded.push_back('=');
    }
    
    return encoded;
}

XntRpcClient::XntRpcClient(const RpcConfig& cfg)
    : config(cfg)
    , request_id_counter(1)
    , last_error(RpcError::None)
    , last_error_message("") {
}

XntRpcClient::XntRpcClient(const std::string& url,
                           const std::string& user,
                           const std::string& password)
    : config()
    , request_id_counter(1)
    , last_error(RpcError::None)
    , last_error_message("") {
    config.url = url;
    config.user = user;
    config.password = password;
}

bool XntRpcClient::make_rpc_request(const std::string& method,
                                   const json& params,
                                   json& response) {
    std::lock_guard<std::mutex> lock(rpc_mutex);
    
    json request;
    request["jsonrpc"] = "2.0";
    request["method"] = method;
    request["params"] = params;
    request["id"] = request_id_counter.fetch_add(1);
    
    std::string request_body = request.dump();
    std::string auth_header = build_auth_header();
    std::string http_response;
    
    if (!http_post(config.url, request_body, auth_header, http_response, config.timeout_sec)) {
        last_error = RpcError::ConnectionFailed;
        last_error_message = "HTTP request failed";
        return false;
    }
    
    try {
        response = json::parse(http_response);
        
        if (response.contains("error")) {
            last_error = parse_rpc_error(response);
            last_error_message = get_error_message(response);
            return false;
        }
        
        last_error = RpcError::None;
        last_error_message = "";
        return true;
    } catch (const std::exception& e) {
        last_error = RpcError::InvalidResponse;
        last_error_message = std::string("JSON parse error: ") + e.what();
        return false;
    }
}

std::string XntRpcClient::build_auth_header() const {
    if (!config.user.empty() && !config.password.empty()) {
        std::string credentials = config.user + ":" + config.password;
        return "Basic " + base64_encode_simple(credentials);
    }
    
    if (!config.cookie_file.empty()) {
        std::string cookie = load_cookie();
        if (!cookie.empty()) {
            return "Cookie: " + cookie;
        }
    }
    
    return "";
}

std::string XntRpcClient::load_cookie() const {
    if (config.cookie_file.empty()) {
        return "";
    }
    
    std::ifstream file(config.cookie_file);
    if (!file.is_open()) {
        return "";
    }
    
    std::string cookie;
    std::getline(file, cookie);
    return cookie;
}

RpcError XntRpcClient::parse_rpc_error(const json& response) const {
    if (!response.contains("error")) {
        return RpcError::None;
    }
    
    json error = response["error"];
    std::string message = (error.contains("message") && !error["message"].is_null()) 
        ? error.value("message", "") : "";
    
    if (message.find("InvalidBlock") != std::string::npos) {
        return RpcError::InvalidBlock;
    } else if (message.find("InsufficientWork") != std::string::npos) {
        return RpcError::InsufficientWork;
    } else if (message.find("Authentication") != std::string::npos ||
               message.find("Unauthorized") != std::string::npos) {
        return RpcError::AuthenticationFailed;
    }
    
    return RpcError::Unknown;
}

std::string XntRpcClient::get_error_message(const json& response) const {
    if (!response.contains("error")) {
        return "";
    }
    
    json error = response["error"];
    return (error.contains("message") && !error["message"].is_null())
        ? error.value("message", "Unknown error") : "Unknown error";
}

json XntRpcClient::getBlockTemplate(const std::string& guesser_address) {
    json params = json::array();
    params.push_back(guesser_address);
    
    json response;
    if (make_rpc_request("mining_getBlockTemplate", params, response)) {
        return response;
    }
    
    return json();
}

json XntRpcClient::submitBlock(const json& template_obj, const json& pow) {
    // JSON-RPC server expects params as an array: [template, pow]
    json params = json::array();
    params.push_back(template_obj);
    params.push_back(pow);
    
    json response;
    make_rpc_request("mining_submitBlock", params, response);
    
    // Return response even if it contains an error, so caller can extract error details
    return response;
}

uint64_t XntRpcClient::getChainHeight() {
    json params = json::array();
    json response;
    
    if (make_rpc_request("chain_height", params, response)) {
        if (response.contains("result") && response["result"].contains("height")) {
            return response["result"]["height"].get<uint64_t>();
        }
    }
    
    return 0;
}

std::string XntRpcClient::getTipDigest() {
    json params = json::array();
    json response;
    
    if (make_rpc_request("chain_tipDigest", params, response)) {
        if (response.contains("result") && response["result"].contains("digest")) {
            return response["result"]["digest"].get<std::string>();
        }
    }
    
    return "";
}

json XntRpcClient::getTipHeader() {
    json params = json::array();
    json response;
    
    if (make_rpc_request("chain_tipHeader", params, response)) {
        return response;
    }
    
    return json();
}

bool XntRpcClient::testConnection() {
    json params = json::array();
    json response;
    
    return make_rpc_request("chain_height", params, response);
}

// Track last logged proposal ID to avoid duplicate logs
static std::string g_last_logged_proposal_id;
static bool g_null_template_logged = false;

PowPuzzle parseRpcTemplate(const json& template_response) {
    PowPuzzle puzzle;
    
    try {
        if (!template_response.contains("result")) {
            return puzzle;
        }
        
        json result = template_response["result"];
        if (!result.contains("template") || result["template"].is_null()) {
            // Only log null template once to avoid spam
            if (!g_null_template_logged) {
                std::cout << "[RPC] " << Color::YELLOW << "Template is null (node may be syncing)" << Color::RESET << std::endl;
                g_null_template_logged = true;
            }
            return puzzle;
        }
        // Reset null template flag when we get a valid template
        if (g_null_template_logged) {
            g_null_template_logged = false;
        }
        
        json template_obj = result["template"];
        if (!template_obj.contains("metadata") || template_obj["metadata"].is_null()) {
            return puzzle;
        }
        
        // Match Rust structure: RpcBlockTemplateMetadata
        // Fields: digest, prev_block, threshold, total_guesser_reward, pow_mast_paths
        json metadata = template_obj["metadata"];
        
        // Handle both snake_case (pow_mast_paths) and camelCase (powMastPaths)
        json pow_mast_paths;
        if (metadata.contains("pow_mast_paths") && !metadata["pow_mast_paths"].is_null()) {
            pow_mast_paths = metadata["pow_mast_paths"];
        } else if (metadata.contains("powMastPaths") && !metadata["powMastPaths"].is_null()) {
            pow_mast_paths = metadata["powMastPaths"];
        } else {
            return puzzle;
        }
        
        // Extract proposal/template ID (digest field)
        // This is the unique identifier for this block template
        if (metadata.contains("digest") && !metadata["digest"].is_null()) {
            puzzle.id = metadata.value("digest", "");
        } else {
            // If digest is missing, template is invalid
            return puzzle;
        }
        // Extract threshold (target digest for PoW solution)
        if (metadata.contains("threshold") && !metadata["threshold"].is_null()) {
            puzzle.threshold = metadata.value("threshold", "");
        } else {
            // Threshold is required for mining
            return puzzle;
        }
        // Handle both snake_case (total_guesser_reward) and camelCase (totalGuesserReward)
        if (metadata.contains("total_guesser_reward") && !metadata["total_guesser_reward"].is_null()) {
            puzzle.total_guesser_reward = metadata.value("total_guesser_reward", "");
        } else if (metadata.contains("totalGuesserReward") && !metadata["totalGuesserReward"].is_null()) {
            puzzle.total_guesser_reward = metadata.value("totalGuesserReward", "");
        }
        // Handle both snake_case (prev_block) and camelCase (prevBlock)
        if (metadata.contains("prev_block") && !metadata["prev_block"].is_null()) {
            puzzle.prev_block = metadata.value("prev_block", "");
        } else if (metadata.contains("prevBlock") && !metadata["prevBlock"].is_null()) {
            puzzle.prev_block = metadata.value("prevBlock", "");
        }
        
        if (pow_mast_paths.contains("pow") && pow_mast_paths["pow"].is_array()) {
            for (const auto& path : pow_mast_paths["pow"]) {
                if (path.is_string()) {
                    puzzle.auth_paths.pow.push_back(path.get<std::string>());
                } else if (path.is_array()) {
                    std::ostringstream hex;
                    for (const auto& limb : path) {
                        if (limb.is_number()) {
                            hex << std::hex << limb.get<uint64_t>();
                        }
                    }
                    puzzle.auth_paths.pow.push_back(hex.str());
                }
            }
        }
        
        if (pow_mast_paths.contains("header") && pow_mast_paths["header"].is_array()) {
            for (const auto& path : pow_mast_paths["header"]) {
                if (path.is_string()) {
                    puzzle.auth_paths.header.push_back(path.get<std::string>());
                } else if (path.is_array()) {
                    std::ostringstream hex;
                    for (const auto& limb : path) {
                        if (limb.is_number()) {
                            hex << std::hex << limb.get<uint64_t>();
                        }
                    }
                    puzzle.auth_paths.header.push_back(hex.str());
                }
            }
        }
        
        if (pow_mast_paths.contains("kernel") && pow_mast_paths["kernel"].is_array()) {
            for (const auto& path : pow_mast_paths["kernel"]) {
                if (path.is_string()) {
                    puzzle.auth_paths.kernel.push_back(path.get<std::string>());
                } else if (path.is_array()) {
                    std::ostringstream hex;
                    for (const auto& limb : path) {
                        if (limb.is_number()) {
                            hex << std::hex << limb.get<uint64_t>();
                        }
                    }
                    puzzle.auth_paths.kernel.push_back(hex.str());
                }
            }
        }
        
        // For mainnet blocks >= 15256, we use CONSENSUS_XNT
        // Since current block height is > 15256, always use XNT consensus
        puzzle.consensus_rule_set = CONSENSUS_XNT;
        
        // Only log when we get a NEW block proposal (different proposal ID)
        // This matches the Rust RpcBlockTemplateMetadata structure
        if (puzzle.id != g_last_logged_proposal_id && !puzzle.id.empty()) {
            g_last_logged_proposal_id = puzzle.id;
            
            std::string short_id = puzzle.id.length() > 20 
                ? puzzle.id.substr(0, 12) + "..." + puzzle.id.substr(puzzle.id.length() - 8) 
                : puzzle.id;
            
            std::cout << "[RPC] " << Color::GREEN << Color::BOLD << "✓ Block proposal received" << Color::RESET << std::endl
                      << "  Proposal ID: " << Color::CYAN << short_id << Color::RESET << std::endl
                      << "  Threshold: " << puzzle.threshold.substr(0, 16) << "..." << std::endl
                      << "  Prev Block: " << (puzzle.prev_block.length() > 16 ? puzzle.prev_block.substr(0, 16) + "..." : puzzle.prev_block) << std::endl
                      << "  Reward: " << format_reward_xnt(puzzle.total_guesser_reward) << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cout << "[RPC] " << Color::RED << "✗ Failed to parse block proposal: " << e.what() << Color::RESET << std::endl;
        LOG_DEBUG("Failed to parse RPC template: " << e.what());
    }
    
    return puzzle;
}

json powToRpcFormat(const Pow& pow_solution, const Digest& solution_hash) {
    json pow_json;

    // Digests are serialized as hex strings in the JSON-RPC API
    // Use pow_solution.root (the Merkle root) not solution_hash (the final block hash)
    pow_json["root"] = digest_to_hex(pow_solution.root);

    pow_json["pathA"] = json::array();
    for (size_t i = 0; i < MERKLE_TREE_HEIGHT_; ++i) {
        pow_json["pathA"].push_back(digest_to_hex(pow_solution.path_a[i]));
    }

    pow_json["pathB"] = json::array();
    for (size_t i = 0; i < MERKLE_TREE_HEIGHT_; ++i) {
        pow_json["pathB"].push_back(digest_to_hex(pow_solution.path_b[i]));
    }

    pow_json["nonce"] = digest_to_hex(pow_solution.nonce);

    return pow_json;
}
