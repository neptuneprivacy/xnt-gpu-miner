#include "stratum_client.cuh"
#include "pow.cuh"
#include "digest.cuh"
#include <sstream>
#include <algorithm>

// Parse stratum URL into host and port
bool parse_stratum_url(const std::string& url, std::string& host, int& port) {
    std::string work_url = url;
    
    // Remove protocol prefix
    const std::vector<std::string> prefixes = {
        "stratum+tcp://", "stratum://", "tcp://"
    };
    
    for (const auto& prefix : prefixes) {
        if (work_url.find(prefix) == 0) {
            work_url = work_url.substr(prefix.length());
            break;
        }
    }
    
    // Parse host:port
    size_t port_pos = work_url.rfind(':');
    if (port_pos != std::string::npos) {
        host = work_url.substr(0, port_pos);
        try {
            port = std::stoi(work_url.substr(port_pos + 1));
        } catch (...) {
            return false;
        }
    } else {
        host = work_url;
        port = 3333;  // Default stratum port
    }
    
    return !host.empty() && port > 0 && port <= 65535;
}

StratumClient::StratumClient(const StratumConfig& cfg)
    : config(cfg)
    , sock(INVALID_SOCKET_VALUE)
    , extranonce2_size(4)
    , current_difficulty(1.0) {
    platform_socket_init();
}

StratumClient::StratumClient(const std::string& url, const std::string& username,
                             const std::string& password)
    : sock(INVALID_SOCKET_VALUE)
    , extranonce2_size(4)
    , current_difficulty(1.0) {
    platform_socket_init();
    
    if (!parse_stratum_url(url, config.host, config.port)) {
        config.host = "127.0.0.1";
        config.port = 3333;
    }
    config.username = username;
    config.password = password;
}

StratumClient::~StratumClient() {
    disconnect();
    platform_socket_cleanup();
}

bool StratumClient::connect_tcp() {
    if (sock != INVALID_SOCKET_VALUE) {
        disconnect_tcp();
    }
    
    sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock == INVALID_SOCKET_VALUE) {
        last_error = StratumError::ConnectionFailed;
        last_error_message = "Failed to create socket";
        return false;
    }
    
    // Resolve hostname
    struct hostent* host_entry = gethostbyname(config.host.c_str());
    if (!host_entry) {
        close(sock);
        sock = INVALID_SOCKET_VALUE;
        last_error = StratumError::ConnectionFailed;
        last_error_message = "Failed to resolve hostname: " + config.host;
        return false;
    }
    
    struct sockaddr_in server_addr;
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(config.port);
    memcpy(&server_addr.sin_addr, host_entry->h_addr, host_entry->h_length);
    
    // Set connection timeout
    #ifdef _WIN32
        DWORD timeout = config.connect_timeout_sec * 1000;
        setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, (const char*)&timeout, sizeof(timeout));
        setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, (const char*)&timeout, sizeof(timeout));
    #else
        struct timeval tv;
        tv.tv_sec = config.connect_timeout_sec;
        tv.tv_usec = 0;
        setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
        setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
    #endif
    
    if (::connect(sock, (struct sockaddr*)&server_addr, sizeof(server_addr)) == SOCKET_ERROR_VALUE) {
        close(sock);
        sock = INVALID_SOCKET_VALUE;
        last_error = StratumError::ConnectionFailed;
        last_error_message = "Failed to connect to " + config.host + ":" + std::to_string(config.port);
        return false;
    }
    
    // Set read timeout for normal operation
    #ifdef _WIN32
        timeout = config.read_timeout_sec * 1000;
        setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, (const char*)&timeout, sizeof(timeout));
    #else
        tv.tv_sec = config.read_timeout_sec;
        tv.tv_usec = 0;
        setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    #endif
    
    return true;
}

void StratumClient::disconnect_tcp() {
    if (sock != INVALID_SOCKET_VALUE) {
        #ifdef _WIN32
            shutdown(sock, SD_BOTH);
        #else
            shutdown(sock, SHUT_RDWR);
        #endif
        close(sock);
        sock = INVALID_SOCKET_VALUE;
    }
}

bool StratumClient::send_line(const std::string& line) {
    if (sock == INVALID_SOCKET_VALUE) {
        return false;
    }
    
    std::string msg = line;
    if (msg.empty() || msg.back() != '\n') {
        msg += '\n';
    }
    
    size_t total_sent = 0;
    while (total_sent < msg.length()) {
        ssize_t sent = send(sock, msg.c_str() + total_sent, msg.length() - total_sent, 0);
        if (sent <= 0) {
            last_error = StratumError::ConnectionLost;
            last_error_message = "Failed to send data";
            return false;
        }
        total_sent += sent;
    }
    
    return true;
}

bool StratumClient::send_json(const json& message) {
    return send_line(message.dump());
}

std::string StratumClient::read_line(int timeout_ms) {
    if (sock == INVALID_SOCKET_VALUE) {
        return "";
    }
    
    // Set timeout
    #ifdef _WIN32
        DWORD timeout = timeout_ms;
        setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, (const char*)&timeout, sizeof(timeout));
    #else
        struct timeval tv;
        tv.tv_sec = timeout_ms / 1000;
        tv.tv_usec = (timeout_ms % 1000) * 1000;
        setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    #endif
    
    std::string line;
    char c;
    
    while (true) {
        ssize_t received = recv(sock, &c, 1, 0);
        if (received <= 0) {
            if (received == 0) {
                // Connection closed
                connected = false;
            }
            break;
        }
        
        if (c == '\n') {
            break;
        }
        
        if (c != '\r') {
            line += c;
        }
    }
    
    return line;
}

bool StratumClient::connect() {
    if (connected.load()) {
        return true;
    }
    
    std::cout << "[Stratum] Connecting to " << config.host << ":" << config.port << "..." << std::endl;
    
    if (!connect_tcp()) {
        std::cout << "[Stratum] " << Color::RED << "Connection failed: " << last_error_message << Color::RESET << std::endl;
        return false;
    }
    
    connected = true;
    std::cout << "[Stratum] " << Color::GREEN << "TCP connection established" << Color::RESET << std::endl;
    
    // Perform stratum handshake
    if (!do_subscribe()) {
        std::cout << "[Stratum] " << Color::RED << "Subscribe failed" << Color::RESET << std::endl;
        disconnect();
        return false;
    }
    
    if (!do_authorize()) {
        std::cout << "[Stratum] " << Color::RED << "Authorization failed" << Color::RESET << std::endl;
        disconnect();
        return false;
    }
    
    std::cout << "[Stratum] " << Color::GREEN << "Successfully connected and authorized" << Color::RESET << std::endl;
    
    // Start receive thread
    running = true;
    receive_thread = std::thread(&StratumClient::receive_loop, this);
    
    return true;
}

void StratumClient::disconnect() {
    running = false;
    connected = false;
    authorized = false;
    
    disconnect_tcp();
    
    if (receive_thread.joinable()) {
        receive_thread.join();
    }
    
    // Clear job queue
    {
        std::lock_guard<std::mutex> lock(job_mutex);
        while (!job_queue.empty()) {
            job_queue.pop();
        }
    }
    
    // Clear pending requests
    {
        std::lock_guard<std::mutex> lock(pending_mutex);
        pending_requests.clear();
    }
}

bool StratumClient::is_connected() const {
    return connected.load() && authorized.load();
}

bool StratumClient::do_subscribe() {
    json request;
    request["id"] = request_id_counter++;
    request["method"] = "mining.subscribe";
    request["params"] = json::array({config.worker_name, "xnt-miner/1.0"});
    
    if (!send_json(request)) {
        last_error = StratumError::SubscribeFailed;
        last_error_message = "Failed to send subscribe request";
        return false;
    }
    
    // Wait for response
    std::string line = read_line(config.connect_timeout_sec * 1000);
    if (line.empty()) {
        last_error = StratumError::Timeout;
        last_error_message = "Timeout waiting for subscribe response";
        return false;
    }
    
    try {
        json response = json::parse(line);
        
        if (response.contains("error") && !response["error"].is_null()) {
            last_error = StratumError::SubscribeFailed;
            if (response["error"].is_array() && response["error"].size() > 1) {
                last_error_message = response["error"][1].get<std::string>();
            } else if (response["error"].is_string()) {
                last_error_message = response["error"].get<std::string>();
            } else {
                last_error_message = "Subscribe rejected by server";
            }
            return false;
        }
        
        if (response.contains("result") && response["result"].is_array()) {
            json result = response["result"];
            
            // Parse subscription result
            // Format: [[["mining.set_difficulty", "subscription_id"], ["mining.notify", "subscription_id"]], extranonce1, extranonce2_size]
            if (result.size() >= 2) {
                if (result[1].is_string()) {
                    extranonce1 = result[1].get<std::string>();
                }
            }
            if (result.size() >= 3) {
                if (result[2].is_number()) {
                    extranonce2_size = result[2].get<int>();
                }
            }
            
            // Extract session ID from subscriptions if available
            if (result.size() >= 1 && result[0].is_array()) {
                for (const auto& sub : result[0]) {
                    if (sub.is_array() && sub.size() >= 2) {
                        if (sub[0].get<std::string>() == "mining.notify") {
                            session_id = sub[1].get<std::string>();
                        }
                    }
                }
            }
            
            std::cout << "[Stratum] Subscribed successfully" << std::endl;
            std::cout << "  Session ID: " << (session_id.empty() ? "(none)" : session_id) << std::endl;
            std::cout << "  Extranonce1: " << extranonce1 << std::endl;
            std::cout << "  Extranonce2 size: " << extranonce2_size << std::endl;
            
            return true;
        }
        
        last_error = StratumError::InvalidResponse;
        last_error_message = "Invalid subscribe response format";
        return false;
        
    } catch (const std::exception& e) {
        last_error = StratumError::InvalidResponse;
        last_error_message = std::string("Failed to parse subscribe response: ") + e.what();
        return false;
    }
}

bool StratumClient::do_authorize() {
    json request;
    request["id"] = request_id_counter++;
    request["method"] = "mining.authorize";
    request["params"] = json::array({config.username, config.password});
    
    if (!send_json(request)) {
        last_error = StratumError::AuthenticationFailed;
        last_error_message = "Failed to send authorize request";
        return false;
    }
    
    // Wait for response
    std::string line = read_line(config.connect_timeout_sec * 1000);
    if (line.empty()) {
        last_error = StratumError::Timeout;
        last_error_message = "Timeout waiting for authorize response";
        return false;
    }
    
    try {
        json response = json::parse(line);
        
        if (response.contains("error") && !response["error"].is_null()) {
            last_error = StratumError::AuthenticationFailed;
            if (response["error"].is_array() && response["error"].size() > 1) {
                last_error_message = response["error"][1].get<std::string>();
            } else if (response["error"].is_string()) {
                last_error_message = response["error"].get<std::string>();
            } else {
                last_error_message = "Authorization rejected by server";
            }
            return false;
        }
        
        if (response.contains("result")) {
            if (response["result"].is_boolean() && response["result"].get<bool>()) {
                authorized = true;
                std::cout << "[Stratum] Authorized as: " << config.username << std::endl;
                return true;
            }
        }
        
        last_error = StratumError::AuthenticationFailed;
        last_error_message = "Authorization failed";
        return false;
        
    } catch (const std::exception& e) {
        last_error = StratumError::InvalidResponse;
        last_error_message = std::string("Failed to parse authorize response: ") + e.what();
        return false;
    }
}

void StratumClient::receive_loop() {
    while (running.load() && connected.load()) {
        std::string line = read_line(5000);  // 5 second timeout for polling
        
        if (line.empty()) {
            if (!running.load()) break;
            continue;
        }
        
        try {
            json message = json::parse(line);
            handle_message(message);
        } catch (const std::exception& e) {
            std::cerr << "[Stratum] Failed to parse message: " << e.what() << std::endl;
        }
    }
}

void StratumClient::handle_message(const json& message) {
    // Check if this is a notification (no id) or a response (has id)
    if (!message.contains("id") || message["id"].is_null()) {
        // Notification
        if (message.contains("method")) {
            std::string method = message["method"].get<std::string>();
            json params = message.value("params", json::array());
            handle_notification(method, params);
        }
    } else {
        // Response to a request
        int id = message["id"].get<int>();
        json result = message.value("result", json());
        json error = message.value("error", json());
        handle_response(id, result, error);
    }
}

void StratumClient::handle_notification(const std::string& method, const json& params) {
    if (method == "mining.notify") {
        StratumJob job = parse_job_notification(params);
        if (job.is_valid()) {
            // Update current job
            {
                std::lock_guard<std::mutex> lock(current_job_mutex);
                current_job = job;
            }
            
            // Add to queue
            {
                std::lock_guard<std::mutex> lock(job_mutex);
                if (job.clean_jobs) {
                    // Clear old jobs
                    while (!job_queue.empty()) {
                        job_queue.pop();
                    }
                }
                job_queue.push(job);
            }
            job_cv.notify_one();
            
            std::string short_id = job.job_id.length() > 20 
                ? job.job_id.substr(0, 12) + "..." + job.job_id.substr(job.job_id.length() - 8)
                : job.job_id;
            std::cout << "[Stratum] " << Color::CYAN << Color::BOLD 
                      << "New job received" << Color::RESET 
                      << " | Job ID: " << short_id 
                      << " | Clean: " << (job.clean_jobs ? "yes" : "no") << std::endl;
        }
    } else if (method == "mining.set_difficulty") {
        if (params.is_array() && !params.empty() && params[0].is_number()) {
            current_difficulty = params[0].get<double>();
            std::cout << "[Stratum] Difficulty set to: " << current_difficulty << std::endl;
        }
    } else if (method == "mining.set_extranonce") {
        if (params.is_array() && params.size() >= 2) {
            if (params[0].is_string()) {
                extranonce1 = params[0].get<std::string>();
            }
            if (params[1].is_number()) {
                extranonce2_size = params[1].get<int>();
            }
            std::cout << "[Stratum] Extranonce updated" << std::endl;
        }
    } else {
        std::cout << "[Stratum] Unknown notification: " << method << std::endl;
    }
}

void StratumClient::handle_response(int id, const json& result, const json& error) {
    std::lock_guard<std::mutex> lock(pending_mutex);
    auto it = pending_requests.find(id);
    if (it != pending_requests.end()) {
        json response;
        response["result"] = result;
        response["error"] = error;
        it->second.set_value(response);
        pending_requests.erase(it);
    }
}

StratumJob StratumClient::parse_job_notification(const json& params) {
    StratumJob job;
    
    // XNT Stratum job notification format:
    // [job_id, prev_block_hash, threshold, total_guesser_reward, pow_mast_paths, header_mast_paths, kernel_mast_paths, clean_jobs]
    // 
    // This format is designed to match the RPC getBlockTemplate response structure
    
    try {
        if (!params.is_array() || params.size() < 7) {
            return job;
        }
        
        // Job ID (unique identifier for this job)
        if (params[0].is_string()) {
            job.job_id = params[0].get<std::string>();
        }
        
        // Previous block hash
        if (params[1].is_string()) {
            job.prev_block_hash = params[1].get<std::string>();
        }
        
        // Threshold (target difficulty)
        if (params[2].is_string()) {
            job.threshold = params[2].get<std::string>();
        }
        
        // Total guesser reward
        if (params[3].is_string()) {
            job.total_guesser_reward = params[3].get<std::string>();
        }
        
        // PoW MAST paths (array of 3 hex strings)
        if (params[4].is_array()) {
            for (const auto& path : params[4]) {
                if (path.is_string()) {
                    job.auth_paths.pow.push_back(path.get<std::string>());
                }
            }
        }
        
        // Header MAST paths (array of 2 hex strings)
        if (params[5].is_array()) {
            for (const auto& path : params[5]) {
                if (path.is_string()) {
                    job.auth_paths.header.push_back(path.get<std::string>());
                }
            }
        }
        
        // Kernel MAST paths (array of 1 hex string)
        if (params[6].is_array()) {
            for (const auto& path : params[6]) {
                if (path.is_string()) {
                    job.auth_paths.kernel.push_back(path.get<std::string>());
                }
            }
        }
        
        // Clean jobs flag (optional)
        if (params.size() > 7 && params[7].is_boolean()) {
            job.clean_jobs = params[7].get<bool>();
        }
        
        // Block height (optional)
        if (params.size() > 8 && params[8].is_number()) {
            job.height = params[8].get<uint64_t>();
        }
        
        job.consensus_rule_set = CONSENSUS_XNT;
        
    } catch (const std::exception& e) {
        std::cerr << "[Stratum] Failed to parse job notification: " << e.what() << std::endl;
        return StratumJob();
    }
    
    return job;
}

json StratumClient::getBlockTemplate() {
    // Return the current job as a JSON template compatible with solo mining format
    std::lock_guard<std::mutex> lock(current_job_mutex);
    
    if (!current_job.is_valid()) {
        return json();
    }
    
    // Build a template response that matches the solo mining format
    json template_obj;
    json metadata;
    
    metadata["digest"] = current_job.job_id;
    metadata["prev_block"] = current_job.prev_block_hash;
    metadata["threshold"] = current_job.threshold;
    metadata["total_guesser_reward"] = current_job.total_guesser_reward;
    
    json pow_mast_paths;
    pow_mast_paths["pow"] = current_job.auth_paths.pow;
    pow_mast_paths["header"] = current_job.auth_paths.header;
    pow_mast_paths["kernel"] = current_job.auth_paths.kernel;
    metadata["pow_mast_paths"] = pow_mast_paths;
    
    template_obj["metadata"] = metadata;
    
    // Stratum doesn't provide the full block structure, so we leave it minimal
    template_obj["block"] = json::object();
    
    json result;
    result["template"] = template_obj;
    
    json response;
    response["result"] = result;
    
    return response;
}

bool StratumClient::wait_for_job(json& job, int timeout_ms) {
    std::unique_lock<std::mutex> lock(job_mutex);
    
    if (job_cv.wait_for(lock, std::chrono::milliseconds(timeout_ms), [this] {
        return !job_queue.empty() || !running.load();
    })) {
        if (!job_queue.empty()) {
            StratumJob stratum_job = job_queue.front();
            job_queue.pop();
            
            // Convert to JSON format compatible with mining controller
            PowPuzzle puzzle = stratum_job.to_pow_puzzle();
            
            // Build response matching parseRpcTemplate expected format
            json template_obj;
            json metadata;
            
            metadata["digest"] = puzzle.id;
            metadata["prev_block"] = puzzle.prev_block;
            metadata["threshold"] = puzzle.threshold;
            metadata["total_guesser_reward"] = puzzle.total_guesser_reward;
            
            json pow_mast_paths;
            pow_mast_paths["pow"] = puzzle.auth_paths.pow;
            pow_mast_paths["header"] = puzzle.auth_paths.header;
            pow_mast_paths["kernel"] = puzzle.auth_paths.kernel;
            metadata["pow_mast_paths"] = pow_mast_paths;
            
            template_obj["metadata"] = metadata;
            template_obj["block"] = json::object();
            
            json result;
            result["template"] = template_obj;
            
            job["result"] = result;
            
            return true;
        }
    }
    
    return false;
}

bool StratumClient::submit_solution(
    const std::string& proposal_id,
    const Pow& pow_solution,
    const Digest& solution_hash,
    const json& template_obj) {
    
    (void)template_obj;  // Not used in stratum submission
    
    if (!is_connected()) {
        std::cout << "[Stratum] " << Color::RED << "Cannot submit - not connected" << Color::RESET << std::endl;
        return false;
    }
    
    shares_submitted++;
    
    // Build stratum submit request
    // Format: mining.submit(username, job_id, extranonce2, ntime, nonce, pow_data)
    // 
    // For XNT, we need to submit the full PoW proof:
    // [username, job_id, merkle_root, path_a[], path_b[], nonce]
    
    json params = json::array();
    params.push_back(config.username);
    params.push_back(proposal_id);
    
    // Serialize PoW solution
    params.push_back(digest_to_hex(pow_solution.root));
    
    // Path A
    json path_a = json::array();
    for (size_t i = 0; i < MERKLE_TREE_HEIGHT_; ++i) {
        path_a.push_back(digest_to_hex(pow_solution.path_a[i]));
    }
    params.push_back(path_a);
    
    // Path B
    json path_b = json::array();
    for (size_t i = 0; i < MERKLE_TREE_HEIGHT_; ++i) {
        path_b.push_back(digest_to_hex(pow_solution.path_b[i]));
    }
    params.push_back(path_b);
    
    // Nonce
    params.push_back(digest_to_hex(pow_solution.nonce));
    
    // Solution hash (the final block hash)
    params.push_back(digest_to_hex(solution_hash));
    
    json request;
    int req_id = request_id_counter++;
    request["id"] = req_id;
    request["method"] = "mining.submit";
    request["params"] = params;
    
    // Create promise for response
    std::promise<json> response_promise;
    std::future<json> response_future = response_promise.get_future();
    
    {
        std::lock_guard<std::mutex> lock(pending_mutex);
        pending_requests[req_id] = std::move(response_promise);
    }
    
    if (!send_json(request)) {
        std::lock_guard<std::mutex> lock(pending_mutex);
        pending_requests.erase(req_id);
        std::cout << "[Stratum] " << Color::RED << "Failed to send submit request" << Color::RESET << std::endl;
        shares_rejected++;
        return false;
    }
    
    // Wait for response with timeout
    if (response_future.wait_for(std::chrono::seconds(30)) != std::future_status::ready) {
        std::lock_guard<std::mutex> lock(pending_mutex);
        pending_requests.erase(req_id);
        std::cout << "[Stratum] " << Color::RED << "Timeout waiting for submit response" << Color::RESET << std::endl;
        shares_rejected++;
        return false;
    }
    
    json response = response_future.get();
    
    if (response.contains("error") && !response["error"].is_null()) {
        shares_rejected++;
        std::string error_msg = "Unknown error";
        if (response["error"].is_array() && response["error"].size() > 1) {
            error_msg = response["error"][1].get<std::string>();
        } else if (response["error"].is_string()) {
            error_msg = response["error"].get<std::string>();
        }
        std::cout << "[Stratum] " << Color::RED << "Share rejected: " << error_msg << Color::RESET << std::endl;
        return false;
    }
    
    if (response.contains("result") && response["result"].is_boolean() && response["result"].get<bool>()) {
        shares_accepted++;
        std::cout << "[Stratum] " << Color::GREEN << Color::BOLD 
                  << "Share accepted!" << Color::RESET 
                  << " (" << shares_accepted.load() << "/" << shares_submitted.load() << ")" << std::endl;
        return true;
    }
    
    shares_rejected++;
    std::cout << "[Stratum] " << Color::RED << "Share rejected (unknown reason)" << Color::RESET << std::endl;
    return false;
}
