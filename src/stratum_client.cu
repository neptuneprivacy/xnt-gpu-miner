#include "stratum_client.cuh"
#include "pow.cuh"
#include "digest.cuh"
#include <sstream>
#include <iomanip>
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
    , sock(INVALID_SOCKET_VALUE) {
    platform_socket_init();
}

StratumClient::StratumClient(const std::string& url, const std::string& address,
                             const std::string& worker_name,
                             const std::string& password)
    : sock(INVALID_SOCKET_VALUE) {
    platform_socket_init();
    
    if (!parse_stratum_url(url, config.host, config.port)) {
        config.host = "127.0.0.1";
        config.port = 3333;
    }
    config.address = address;
    config.name = worker_name;
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
    
    std::cout << "[Pool] Connecting to " << config.host << ":" << config.port << "..." << std::endl;
    
    if (!connect_tcp()) {
        std::cout << "[Pool] " << Color::RED << "Connection failed: " << last_error_message << Color::RESET << std::endl;
        return false;
    }
    
    connected = true;
    std::cout << "[Pool] " << Color::GREEN << "TCP connection established" << Color::RESET << std::endl;
    
    // Perform login
    if (!do_login()) {
        std::cout << "[Pool] " << Color::RED << "Login failed: " << last_error_message << Color::RESET << std::endl;
        disconnect();
        return false;
    }
    
    std::cout << "[Pool] " << Color::GREEN << "Successfully logged in as worker " << worker_id << Color::RESET << std::endl;
    
    // Start receive thread
    running = true;
    receive_thread = std::thread(&StratumClient::receive_loop, this);
    
    // Start keepalive thread
    keepalive_thread = std::thread(&StratumClient::keepalive_loop, this);
    
    return true;
}

void StratumClient::disconnect() {
    running = false;
    connected = false;
    logged_in = false;
    
    disconnect_tcp();
    
    // Wake up any waiting threads
    job_cv.notify_all();
    
    if (receive_thread.joinable()) {
        receive_thread.join();
    }
    
    if (keepalive_thread.joinable()) {
        keepalive_thread.join();
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
    
    worker_id = 0;
}

bool StratumClient::is_connected() const {
    return connected.load() && logged_in.load();
}

bool StratumClient::do_login() {
    // Build login request matching pool schema:
    // { "id": N, "method": "login", "params": { "name": "...", "address": "...", "password": "...", "agent": "..." } }
    json params;
    params["name"] = config.name;
    params["address"] = config.address;
    if (!config.password.empty()) {
        params["password"] = config.password;
    }
    params["agent"] = config.agent;
    
    json request;
    request["id"] = request_id_counter++;
    request["method"] = "login";
    request["params"] = params;
    
    if (!send_json(request)) {
        last_error = StratumError::AuthenticationFailed;
        last_error_message = "Failed to send login request";
        return false;
    }
    
    // Wait for response
    std::string line = read_line(config.connect_timeout_sec * 1000);
    if (line.empty()) {
        last_error = StratumError::Timeout;
        last_error_message = "Timeout waiting for login response";
        return false;
    }
    
    try {
        json response = json::parse(line);
        
        // Check for JSON-RPC error
        if (response.contains("error") && !response["error"].is_null()) {
            last_error = StratumError::AuthenticationFailed;
            auto& err = response["error"];
            if (err.is_object() && err.contains("message")) {
                last_error_message = err["message"].get<std::string>();
            } else if (err.is_string()) {
                last_error_message = err.get<std::string>();
            } else {
                last_error_message = "Login rejected by pool";
            }
            return false;
        }
        
        // Parse login response: { "id": N, "jsonrpc": "2.0", "result": { "id": worker_id, "job": {...} } }
        if (response.contains("result") && response["result"].is_object()) {
            auto& result = response["result"];
            
            // Get worker ID
            if (result.contains("id") && result["id"].is_number()) {
                worker_id = result["id"].get<size_t>();
            } else {
                last_error = StratumError::InvalidResponse;
                last_error_message = "Login response missing worker id";
                return false;
            }
            
            logged_in = true;
            std::cout << "[Pool] Logged in as: " << config.name << " (address: " << config.address << ")" << std::endl;
            std::cout << "[Pool] Worker ID: " << worker_id << std::endl;
            
            // Check for initial job in login response
            if (result.contains("job") && !result["job"].is_null()) {
                StratumJob job = parse_job_notification(result["job"]);
                if (job.is_valid()) {
                    std::lock_guard<std::mutex> lock(current_job_mutex);
                    current_job = job;
                    
                    {
                        std::lock_guard<std::mutex> job_lock(job_mutex);
                        job_queue.push(job);
                    }
                    job_cv.notify_one();
                    
                    std::cout << "[Pool] Received initial job: " << job.job_id.substr(0, 16) << "..." << std::endl;
                }
            }
            
            return true;
        }
        
        last_error = StratumError::InvalidResponse;
        last_error_message = "Invalid login response format";
        return false;
        
    } catch (const std::exception& e) {
        last_error = StratumError::InvalidResponse;
        last_error_message = std::string("Failed to parse login response: ") + e.what();
        return false;
    }
}

bool StratumClient::send_keepalive() {
    // Build keepalive request: { "method": "keepalived", "params": {} }
    json request;
    request["method"] = "keepalived";
    request["params"] = json::object();
    // Note: keepalived is a notification, no id needed
    
    return send_json(request);
}

void StratumClient::keepalive_loop() {
    while (running.load() && connected.load()) {
        // Sleep for keepalive interval
        for (int i = 0; i < config.keepalive_interval_sec && running.load(); ++i) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        
        if (!running.load() || !connected.load()) {
            break;
        }
        
        if (!send_keepalive()) {
            std::cerr << "[Pool] Failed to send keepalive" << std::endl;
        }
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
            std::cerr << "[Pool] Failed to parse message: " << e.what() << std::endl;
        }
    }
}

void StratumClient::handle_message(const json& message) {
    // Check if this is a notification (no id) or a response (has id)
    if (!message.contains("id") || message["id"].is_null()) {
        // Notification from pool (job, pause, etc.)
        if (message.contains("method")) {
            std::string method = message["method"].get<std::string>();
            json params = message.value("params", json::object());
            handle_notification(method, params);
        }
    } else {
        // Response to a request
        uint64_t id = message["id"].get<uint64_t>();
        json result = message.value("result", json());
        json error = message.value("error", json());
        handle_response(id, result, error);
    }
}

void StratumClient::handle_notification(const std::string& method, const json& params) {
    // Pool schema notifications: job, pause
    if (method == "job") {
        // Job notification: { "method": "job", "params": { "id": "...", "paths": {...}, "difficulty": "..." } }
        StratumJob job = parse_job_notification(params);
        if (job.is_valid()) {
            // Update current job
            {
                std::lock_guard<std::mutex> lock(current_job_mutex);
                current_job = job;
            }
            
            // Add to queue (new jobs always replace old ones)
            {
                std::lock_guard<std::mutex> lock(job_mutex);
                // Clear old jobs on new job
                while (!job_queue.empty()) {
                    job_queue.pop();
                }
                job_queue.push(job);
            }
            job_cv.notify_one();
            
            std::string short_id = job.job_id.length() > 20 
                ? job.job_id.substr(0, 12) + "..." + job.job_id.substr(job.job_id.length() - 8)
                : job.job_id;
            std::cout << "[Pool] " << Color::CYAN << Color::BOLD 
                      << "New job received" << Color::RESET 
                      << " | Job ID: " << short_id 
                      << " | Difficulty: " << job.difficulty << std::endl;
        }
    } else if (method == "pause") {
        // Pause notification: stop mining temporarily
        std::cout << "[Pool] " << Color::YELLOW << "Mining paused by pool" << Color::RESET << std::endl;
        // Clear job queue to stop mining
        {
            std::lock_guard<std::mutex> lock(job_mutex);
            while (!job_queue.empty()) {
                job_queue.pop();
            }
        }
        {
            std::lock_guard<std::mutex> lock(current_job_mutex);
            current_job = StratumJob();  // Invalidate current job
        }
    } else {
        std::cout << "[Pool] Unknown notification: " << method << std::endl;
    }
}

void StratumClient::handle_response(uint64_t id, const json& result, const json& error) {
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
    
    // Pool schema Job format (from schema.rs):
    // {
    //   "id": "digest_hex_string",
    //   "paths": {
    //     "pow": { "kernel_body": [...], "type_scripts": [...], "kernel": [...] },
    //     "header": { "body": [...], "appendix": [...] },
    //     "kernel": [...]
    //   },
    //   "difficulty": "string"
    // }
    
    try {
        if (!params.is_object()) {
            std::cerr << "[Pool] Job params is not an object" << std::endl;
            return job;
        }
        
        // Job ID (digest as hex string)
        if (params.contains("id")) {
            if (params["id"].is_string()) {
                job.job_id = params["id"].get<std::string>();
            } else if (params["id"].is_array()) {
                // Handle Digest as array of u64 values - convert to hex
                std::stringstream ss;
                for (const auto& val : params["id"]) {
                    if (val.is_number()) {
                        ss << std::hex << std::setfill('0') << std::setw(16) << val.get<uint64_t>();
                    }
                }
                job.job_id = ss.str();
            }
        }
        
        // Difficulty
        if (params.contains("difficulty") && params["difficulty"].is_string()) {
            job.difficulty = params["difficulty"].get<std::string>();
        }
        
        // PowMastPaths structure
        if (params.contains("paths") && params["paths"].is_object()) {
            const auto& paths = params["paths"];
            
            // Parse pow paths: { "kernel_body": [...], "type_scripts": [...], "kernel": [...] }
            if (paths.contains("pow") && paths["pow"].is_object()) {
                const auto& pow = paths["pow"];
                
                if (pow.contains("kernel_body") && pow["kernel_body"].is_array()) {
                    for (const auto& item : pow["kernel_body"]) {
                        if (item.is_string()) {
                            job.paths.pow_kernel_body.push_back(item.get<std::string>());
                        }
                    }
                }
                if (pow.contains("type_scripts") && pow["type_scripts"].is_array()) {
                    for (const auto& item : pow["type_scripts"]) {
                        if (item.is_string()) {
                            job.paths.pow_type_scripts.push_back(item.get<std::string>());
                        }
                    }
                }
                if (pow.contains("kernel") && pow["kernel"].is_array()) {
                    for (const auto& item : pow["kernel"]) {
                        if (item.is_string()) {
                            job.paths.pow_kernel.push_back(item.get<std::string>());
                        }
                    }
                }
            }
            
            // Parse header paths: { "body": [...], "appendix": [...] }
            if (paths.contains("header") && paths["header"].is_object()) {
                const auto& header = paths["header"];
                
                if (header.contains("body") && header["body"].is_array()) {
                    for (const auto& item : header["body"]) {
                        if (item.is_string()) {
                            job.paths.header_body.push_back(item.get<std::string>());
                        }
                    }
                }
                if (header.contains("appendix") && header["appendix"].is_array()) {
                    for (const auto& item : header["appendix"]) {
                        if (item.is_string()) {
                            job.paths.header_appendix.push_back(item.get<std::string>());
                        }
                    }
                }
            }
            
            // Parse kernel paths (direct array)
            if (paths.contains("kernel") && paths["kernel"].is_array()) {
                for (const auto& item : paths["kernel"]) {
                    if (item.is_string()) {
                        job.paths.kernel.push_back(item.get<std::string>());
                    }
                }
            }
        }
        
    } catch (const std::exception& e) {
        std::cerr << "[Pool] Failed to parse job notification: " << e.what() << std::endl;
        return StratumJob();
    }
    
    return job;
}

json StratumClient::getBlockTemplate() {
    // Return the current job as a JSON template compatible with mining code
    std::lock_guard<std::mutex> lock(current_job_mutex);
    
    if (!current_job.is_valid()) {
        return json();
    }
    
    // Build a template response that matches the expected format
    json template_obj;
    json metadata;
    
    metadata["digest"] = current_job.job_id;
    metadata["threshold"] = current_job.difficulty;
    
    // Build pow_mast_paths from pool paths structure
    json pow_mast_paths;
    
    // Combine pow paths into single array for legacy format
    json pow_paths = json::array();
    for (const auto& p : current_job.paths.pow_kernel_body) pow_paths.push_back(p);
    for (const auto& p : current_job.paths.pow_type_scripts) pow_paths.push_back(p);
    for (const auto& p : current_job.paths.pow_kernel) pow_paths.push_back(p);
    pow_mast_paths["pow"] = pow_paths;
    
    // Combine header paths
    json header_paths = json::array();
    for (const auto& p : current_job.paths.header_body) header_paths.push_back(p);
    for (const auto& p : current_job.paths.header_appendix) header_paths.push_back(p);
    pow_mast_paths["header"] = header_paths;
    
    // Kernel paths
    pow_mast_paths["kernel"] = current_job.paths.kernel;
    
    metadata["pow_mast_paths"] = pow_mast_paths;
    
    template_obj["metadata"] = metadata;
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
            
            // Build response matching expected format
            json template_obj;
            json metadata;
            
            metadata["digest"] = puzzle.id;
            metadata["threshold"] = puzzle.threshold;
            
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
    
    (void)template_obj;  // Not used in pool submission
    (void)solution_hash; // Not needed in pool schema
    
    if (!is_connected()) {
        std::cout << "[Pool] " << Color::RED << "Cannot submit - not connected" << Color::RESET << std::endl;
        return false;
    }
    
    shares_submitted++;
    
    // Build submit request matching pool schema:
    // { "id": N, "method": "submit", "params": { "worker": worker_id, "id": job_id, "pow": BlockPow } }
    //
    // BlockPow structure (from neptune-cash):
    // {
    //   "nonce": [u64; 5],           // Digest as array
    //   "root": [u64; 5],            // Digest as array  
    //   "authentication_path_a": [[u64; 5]; 25],  // Array of Digests
    //   "authentication_path_b": [[u64; 5]; 25]   // Array of Digests
    // }
    
    // Build BlockPow object
    json pow_obj;
    
    // Nonce as array of 5 u64 values
    json nonce_arr = json::array();
    for (int i = 0; i < 5; ++i) {
        nonce_arr.push_back(pow_solution.nonce.values[i]);
    }
    pow_obj["nonce"] = nonce_arr;
    
    // Root as array of 5 u64 values
    json root_arr = json::array();
    for (int i = 0; i < 5; ++i) {
        root_arr.push_back(pow_solution.root.values[i]);
    }
    pow_obj["root"] = root_arr;
    
    // Authentication path A - array of MERKLE_TREE_HEIGHT_ digests
    json path_a = json::array();
    for (size_t i = 0; i < MERKLE_TREE_HEIGHT_; ++i) {
        json digest_arr = json::array();
        for (int j = 0; j < 5; ++j) {
            digest_arr.push_back(pow_solution.path_a[i].values[j]);
        }
        path_a.push_back(digest_arr);
    }
    pow_obj["authentication_path_a"] = path_a;
    
    // Authentication path B - array of MERKLE_TREE_HEIGHT_ digests
    json path_b = json::array();
    for (size_t i = 0; i < MERKLE_TREE_HEIGHT_; ++i) {
        json digest_arr = json::array();
        for (int j = 0; j < 5; ++j) {
            digest_arr.push_back(pow_solution.path_b[i].values[j]);
        }
        path_b.push_back(digest_arr);
    }
    pow_obj["authentication_path_b"] = path_b;
    
    // Build params object
    json params;
    params["worker"] = worker_id;
    params["id"] = proposal_id;
    params["pow"] = pow_obj;
    
    json request;
    uint64_t req_id = request_id_counter++;
    request["id"] = req_id;
    request["method"] = "submit";
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
        std::cout << "[Pool] " << Color::RED << "Failed to send submit request" << Color::RESET << std::endl;
        shares_rejected++;
        return false;
    }
    
    // Wait for response with timeout
    if (response_future.wait_for(std::chrono::seconds(30)) != std::future_status::ready) {
        std::lock_guard<std::mutex> lock(pending_mutex);
        pending_requests.erase(req_id);
        std::cout << "[Pool] " << Color::RED << "Timeout waiting for submit response" << Color::RESET << std::endl;
        shares_rejected++;
        return false;
    }
    
    json response = response_future.get();
    
    // Check for JSON-RPC error
    if (response.contains("error") && !response["error"].is_null()) {
        shares_rejected++;
        std::string error_msg = "Unknown error";
        auto& err = response["error"];
        if (err.is_object() && err.contains("message")) {
            error_msg = err["message"].get<std::string>();
        } else if (err.is_string()) {
            error_msg = err.get<std::string>();
        }
        std::cout << "[Pool] " << Color::RED << "Share rejected: " << error_msg << Color::RESET << std::endl;
        return false;
    }
    
    // Pool schema submit response: { "result": { "success": true } }
    if (response.contains("result") && response["result"].is_object()) {
        auto& result = response["result"];
        if (result.contains("success") && result["success"].is_boolean() && result["success"].get<bool>()) {
            shares_accepted++;
            std::cout << "[Pool] " << Color::GREEN << Color::BOLD 
                      << "Share accepted!" << Color::RESET 
                      << " (" << shares_accepted.load() << "/" << shares_submitted.load() << ")" << std::endl;
            return true;
        }
    }
    
    shares_rejected++;
    std::cout << "[Pool] " << Color::RED << "Share rejected (unknown reason)" << Color::RESET << std::endl;
    return false;
}
