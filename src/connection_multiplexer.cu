#include "connection_multiplexer.cuh"
#include "rpc_client.cuh"
#include "stratum_client.cuh"
#include "network.cuh"
#include "mining.cuh"
#include "pow.cuh"
#include "digest.cuh"
#include "common.cuh"

// ============================================================================
// Singleton Instance
// ============================================================================

std::unique_ptr<ConnectionMultiplexer> ConnectionMultiplexer::instance = nullptr;
std::mutex ConnectionMultiplexer::instance_mutex;

ConnectionMultiplexer& ConnectionMultiplexer::getInstance() {
    std::lock_guard<std::mutex> lock(instance_mutex);
    if (!instance) {
        instance = std::unique_ptr<ConnectionMultiplexer>(new ConnectionMultiplexer());
    }
    return *instance;
}

void ConnectionMultiplexer::destroyInstance() {
    std::lock_guard<std::mutex> lock(instance_mutex);
    if (instance) {
        instance->shutdown();
        instance.reset();
    }
}

// ============================================================================
// SoloMiningClient Implementation
// ============================================================================

SoloMiningClient::SoloMiningClient(const std::string& url)
    : rpc_url(url) {
    RpcConfig config;
    config.url = url;
    config.timeout_sec = 30;
    config.poll_interval_sec = g_fetch_interval_sec;
    rpc_client = std::make_unique<XntRpcClient>(config);
}

SoloMiningClient::~SoloMiningClient() {
    disconnect();
}

bool SoloMiningClient::connect() {
    std::lock_guard<std::mutex> lock(client_mutex);
    if (!rpc_client) {
        return false;
    }
    bool result = rpc_client->testConnection();
    connected.store(result);
    return result;
}

void SoloMiningClient::disconnect() {
    connected.store(false);
}

bool SoloMiningClient::is_connected() const {
    return connected.load();
}

// reconnect() is now implemented inline in the header

json SoloMiningClient::getBlockTemplate() {
    std::lock_guard<std::mutex> lock(client_mutex);
    if (!rpc_client) {
        return json();
    }
    return rpc_client->getBlockTemplate("");
}

json SoloMiningClient::getBlockTemplate(const std::string& wallet_address) {
    std::lock_guard<std::mutex> lock(client_mutex);
    if (!rpc_client || wallet_address.empty()) {
        return json();
    }
    return rpc_client->getBlockTemplate(wallet_address);
}

bool SoloMiningClient::submitSolution(
    const std::string& proposal_id,
    const Pow& pow_solution,
    const Digest& solution_hash,
    const json& template_obj) {
    
    std::lock_guard<std::mutex> lock(client_mutex);
    
    if (!rpc_client) {
        return false;
    }
    
    if (template_obj.is_null() || template_obj.empty()) {
        std::cout << "[SUBMIT] ERROR: No template provided for proposal " << proposal_id << std::endl;
        return false;
    }
    
    // Check if template is stale
    std::string tip_digest = rpc_client->getTipDigest();
    
    std::string prev_block;
    if (template_obj.contains("metadata") && !template_obj["metadata"].is_null()) {
        json metadata = template_obj["metadata"];
        if (metadata.contains("prevBlock") && !metadata["prevBlock"].is_null()) {
            prev_block = metadata.value("prevBlock", "");
        } else if (metadata.contains("prev_block") && !metadata["prev_block"].is_null()) {
            prev_block = metadata.value("prev_block", "");
        }
    }
    
    // Debug: Always show what we're comparing
    std::string short_prev = prev_block.length() > 20 ? prev_block.substr(0, 12) + "..." + prev_block.substr(prev_block.length() - 8) : prev_block;
    std::string short_tip = tip_digest.length() > 20 ? tip_digest.substr(0, 12) + "..." + tip_digest.substr(tip_digest.length() - 8) : tip_digest;
    std::cout << "[SUBMIT DEBUG] prev_block=" << (prev_block.empty() ? "(empty)" : short_prev) 
              << " tip=" << short_tip << std::endl;

    if (!prev_block.empty() && tip_digest != prev_block) {
        std::cout << "[SUBMIT] " << Color::RED << "STALE: Template stale" << Color::RESET << std::endl;
        return false;
    }
    
    // Extract the block from template
    if (!template_obj.contains("block") || template_obj["block"].is_null()) {
        std::cout << "[SUBMIT] ERROR: Template missing 'block' field" << std::endl;
        return false;
    }
    
    json block_obj = template_obj["block"];
    
    if (!block_obj.contains("kernel") || block_obj["kernel"].is_null()) {
        std::cout << "[SUBMIT] ERROR: Block missing 'kernel' field" << std::endl;
        return false;
    }
    
    // Debug: Log appendix structure being sent
    json kernel_obj = block_obj["kernel"];
    if (kernel_obj.contains("appendix") && kernel_obj["appendix"].is_array()) {
        std::cout << "[SUBMIT DEBUG] Appendix has " << kernel_obj["appendix"].size() << " claims" << std::endl;
        for (size_t i = 0; i < kernel_obj["appendix"].size(); ++i) {
            json claim = kernel_obj["appendix"][i];
            std::cout << "[SUBMIT DEBUG] Claim " << i << ": " << claim.dump() << std::endl;
        }
    } else {
        std::cout << "[SUBMIT DEBUG] " << Color::RED << "ERROR: No appendix or appendix not array!" << Color::RESET << std::endl;
        std::cout << "[SUBMIT DEBUG] kernel keys: ";
        for (auto it = kernel_obj.begin(); it != kernel_obj.end(); ++it) {
            std::cout << it.key() << " ";
        }
        std::cout << std::endl;
    }
    
    json pow_json = powToRpcFormat(pow_solution, solution_hash);
    
    json response = rpc_client->submitBlock(block_obj, pow_json);
    
    if (!response.empty() && response.contains("result")) {
        if (response["result"].is_boolean()) {
            bool success = response["result"].get<bool>();
            if (!success) {
                std::cout << "[SUBMIT] " << Color::RED << "REJECTED: Block rejected" << Color::RESET << std::endl;
            }
            return success;
        }
        if (response["result"].contains("success")) {
            bool success = response["result"]["success"].get<bool>();
            if (!success) {
                std::cout << "[SUBMIT] " << Color::RED << "REJECTED: Block rejected" << Color::RESET << std::endl;
            }
            return success;
        }
        return true;
    }
    
    if (response.contains("error") && !response["error"].is_null()) {
        json error = response["error"];
        std::string error_reason = "Unknown error";
        
        if (error.contains("data") && !error["data"].is_null()) {
            json error_data = error["data"];
            if (error_data.is_object() && !error_data.empty()) {
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
        
        std::cout << "[SUBMIT] " << Color::RED << "REJECTED: " << error_reason << Color::RESET << std::endl;
    } else {
        std::cout << "[SUBMIT] " << Color::RED << "REJECTED: No response from server" << Color::RESET << std::endl;
    }
    
    return false;
}

std::string SoloMiningClient::getTipDigest() {
    std::lock_guard<std::mutex> lock(client_mutex);
    if (!rpc_client) {
        return "";
    }
    return rpc_client->getTipDigest();
}

uint64_t SoloMiningClient::getChainHeight() {
    std::lock_guard<std::mutex> lock(client_mutex);
    if (!rpc_client) {
        return 0;
    }
    return rpc_client->getChainHeight();
}

// ============================================================================
// ConnectionMultiplexer Implementation
// ============================================================================

ConnectionMultiplexer::ConnectionMultiplexer() {
    stats.last_job_time = std::chrono::steady_clock::now();
    stats.last_submission_time = std::chrono::steady_clock::now();
}

ConnectionMultiplexer::~ConnectionMultiplexer() {
    shutdown();
}

bool ConnectionMultiplexer::initialize(const std::string& ep, const std::string& wallet, 
                                       const std::string& stratum_password) {
    if (initialized.load()) {
        return true;
    }
    
    endpoint = ep;
    wallet_address = wallet;
    
    // Detect mining mode from endpoint URL
    MiningMode mode = detect_mining_mode(endpoint);
    mining_mode = mode;
    
    // Create the appropriate mining client based on mode
    if (mode == MiningMode::Stratum) {
        std::string host;
        int port;
        bool use_ssl = false;
        if (parse_stratum_url(endpoint, host, port, use_ssl)) {
            StratumConfig config;
            config.host = host;
            config.port = port;
            config.use_ssl = use_ssl;
            config.address = wallet_address;
            config.name = g_miner_worker_name.empty() ? "xnt-miner" : g_miner_worker_name;
            config.password = stratum_password;
            config.agent = "xnt-gpu-miner/1.0";
            client = std::make_unique<StratumClient>(config);
            std::cout << "Attempting to connect to stratum server at " << endpoint << "..." << std::endl;
        } else {
            std::cerr << Color::RED << "Invalid stratum URL: " << endpoint << Color::RESET << std::endl;
            return false;
        }
    } else {
        client = std::make_unique<SoloMiningClient>(endpoint);
        std::cout << "Attempting to connect to RPC server at " << endpoint << "..." << std::endl;
    }
    
    // Initial connection attempt
    int retry_count = 0;
    bool conn_success = false;
    
    while (!conn_success && !stop_mining && retry_count < 10) {
        if (client->connect()) {
            conn_success = true;
            const char* server_type = (mining_mode == MiningMode::Stratum) ? "stratum server" : "RPC server";
            std::cout << Color::GREEN << "Successfully connected to " << server_type << "!" << Color::RESET << std::endl;
            break;
        }
        
        retry_count++;
        std::cout << "Connection attempt " << retry_count << " failed, retrying in 10 seconds..." << std::endl;
        
        for (int i = 0; i < 10 && !stop_mining; ++i) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }
    
    if (!conn_success) {
        std::cout << Color::RED << "Failed to connect to RPC server after " << retry_count << " attempts" << Color::RESET << std::endl;
        return false;
    }
    
    connected.store(true);
    running.store(true);
    initialized.store(true);
    
    // Start worker threads
    job_broadcaster_thread = std::thread(&ConnectionMultiplexer::jobBroadcasterLoop, this);
    solution_submitter_thread = std::thread(&ConnectionMultiplexer::solutionSubmitterLoop, this);
    health_monitor_thread = std::thread(&ConnectionMultiplexer::healthMonitorLoop, this);
    
    // Start tip monitor for solo mode (fast stale detection)
    if (mining_mode == MiningMode::Solo) {
        tip_monitor_thread = std::thread(&ConnectionMultiplexer::tipMonitorLoop, this);
    }
    
    std::cout << Color::GREEN << "Connection multiplexer started" << Color::RESET << std::endl;
    
    return true;
}

void ConnectionMultiplexer::shutdown() {
    if (!initialized.load()) {
        return;
    }
    
    running.store(false);
    connected.store(false);
    
    // Wake up the solution submitter
    submission_cv.notify_all();
    
    // Wait for threads to finish
    if (job_broadcaster_thread.joinable()) {
        job_broadcaster_thread.join();
    }
    if (solution_submitter_thread.joinable()) {
        solution_submitter_thread.join();
    }
    if (health_monitor_thread.joinable()) {
        health_monitor_thread.join();
    }
    if (tip_monitor_thread.joinable()) {
        tip_monitor_thread.join();
    }
    
    // Clear workers
    {
        std::unique_lock<std::shared_mutex> lock(workers_mutex);
        workers.clear();
    }
    
    // Clear submission queue
    {
        std::lock_guard<std::mutex> lock(submission_mutex);
        while (!submission_queue.empty()) {
            auto& submission = submission_queue.front();
            submission->result_promise->set_value(false);
            submission_queue.pop();
        }
    }
    
    client.reset();
    initialized.store(false);
    
    std::cout << "[Multiplexer] Shutdown complete" << std::endl;
}

GpuWorkerHandle* ConnectionMultiplexer::registerWorker(int gpu_id, GpuResources* resources) {
    std::unique_lock<std::shared_mutex> lock(workers_mutex);
    
    if (!resources->event_handler) {
        resources->event_handler = std::make_unique<EventHandler>();
    }
    auto handle = std::make_unique<GpuWorkerHandle>(gpu_id, resources);
    GpuWorkerHandle* ptr = handle.get();
    workers.push_back(std::move(handle));
    
    std::cout << "[Multiplexer] Registered GPU " << gpu_id << " worker" << std::endl;
    
    return ptr;
}

void ConnectionMultiplexer::unregisterWorker(GpuWorkerHandle* handle) {
    if (!handle) return;
    
    // Save GPU ID before erasing (handle will be deleted by unique_ptr)
    int gpu_id = handle->gpu_id;
    
    std::unique_lock<std::shared_mutex> lock(workers_mutex);
    
    handle->active = false;
    
    workers.erase(
        std::remove_if(workers.begin(), workers.end(),
            [handle](const std::unique_ptr<GpuWorkerHandle>& h) {
                return h.get() == handle;
            }),
        workers.end());
    
    std::cout << "[Multiplexer] Unregistered GPU " << gpu_id << " worker" << std::endl;
}

size_t ConnectionMultiplexer::getActiveWorkerCount() const {
    std::shared_lock<std::shared_mutex> lock(workers_mutex);
    size_t count = 0;
    for (const auto& w : workers) {
        if (w && w->active) {
            count++;
        }
    }
    return count;
}

std::future<bool> ConnectionMultiplexer::submitSolution(
    int gpu_id,
    const std::string& proposal_id,
    const Pow& pow_solution,
    const Digest& solution_hash,
    const json& template_obj) {
    
    auto submission = std::make_unique<SolutionSubmission>(
        gpu_id, proposal_id, pow_solution, solution_hash, template_obj);
    
    auto future = submission->result_promise->get_future();
    
    {
        std::lock_guard<std::mutex> lock(submission_mutex);
        submission_queue.push(std::move(submission));
    }
    submission_cv.notify_one();
    
    // Update worker stats
    {
        std::shared_lock<std::shared_mutex> lock(workers_mutex);
        for (const auto& worker : workers) {
            if (worker && worker->gpu_id == gpu_id) {
                worker->submissions_queued++;
                break;
            }
        }
    }
    
    return future;
}

bool ConnectionMultiplexer::isConnected() const {
    return connected.load() && client && client->is_connected();
}

void ConnectionMultiplexer::updateAllWorkersConnectionState(bool is_connected) {
    std::shared_lock<std::shared_mutex> lock(workers_mutex);
    for (const auto& worker : workers) {
        if (worker && worker->resources) {
            worker->resources->gpu_node_connected = is_connected;
        }
    }
}

// ============================================================================
// Thread Loops
// ============================================================================

void ConnectionMultiplexer::jobBroadcasterLoop() {
    try {
        LOG_DEBUG("[JobBroadcaster] Started");

        auto last_poll_time = std::chrono::steady_clock::now();

        // Wait for at least one worker to register
        while (running.load()) {
            if (getActiveWorkerCount() > 0) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        std::cout << "[JobBroadcaster] Block proposal fetcher started" << std::endl;

        while (running.load() && !stop_mining) {
            // Check connection
            if (!client || !client->is_connected()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
                continue;
            }

            auto now = std::chrono::steady_clock::now();
            auto time_since_poll = std::chrono::duration_cast<std::chrono::seconds>(
                now - last_poll_time).count();

            bool should_poll = false;

            // Check if it's time to poll
            if (time_since_poll >= g_fetch_interval_sec) {
                should_poll = true;
            }

            // Also poll if we have no template yet
            {
                std::lock_guard<std::mutex> lock(job_mutex);
                if (last_template_id.empty()) {
                    should_poll = true;
                }
            }

            if (should_poll) {
                json template_response = client->getBlockTemplate(wallet_address);
                last_poll_time = std::chrono::steady_clock::now();

                if (!template_response.empty() && template_response.contains("result")) {
                    json result = template_response["result"];

                    if (!result.contains("template") || result["template"].is_null()) {
                        // Template null, node may be syncing
                        std::this_thread::sleep_for(std::chrono::milliseconds(500));
                        continue;
                    }

                    json template_obj = result["template"];
                    if (!template_obj.contains("metadata") || template_obj["metadata"].is_null()) {
                        std::this_thread::sleep_for(std::chrono::milliseconds(500));
                        continue;
                    }

                    json metadata = template_obj["metadata"];
                    std::string template_id = metadata.contains("digest")
                        ? metadata.value("digest", "") : "";

                    bool is_new_template = false;
                    {
                        std::lock_guard<std::mutex> lock(job_mutex);
                        if (!template_id.empty() && template_id != last_template_id) {
                            // Verify prev_block matches tip
                            std::string prev_block;
                            if (metadata.contains("prevBlock")) {
                                prev_block = metadata.value("prevBlock", "");
                            } else if (metadata.contains("prev_block")) {
                                prev_block = metadata.value("prev_block", "");
                            }

                            std::string tip_digest = client->getTipDigest();

                            if (prev_block.empty() || prev_block == tip_digest) {
                                last_template_id = template_id;
                                current_tip_digest = tip_digest;  // Track current tip for stale detection
                                is_new_template = true;

                                stats.total_jobs_fetched++;
                                {
                                    std::lock_guard<std::mutex> time_lock(stats.time_mutex);
                                    stats.last_job_time = std::chrono::steady_clock::now();
                                }
                            }
                        }
                    }

                    if (is_new_template) {
                        broadcastJobToWorkers(template_response);
                    }
                }
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }

        LOG_DEBUG("[JobBroadcaster] Stopped");
    } catch (const std::exception& e) {
        std::cerr << "[JobBroadcaster] Exception: " << e.what() << std::endl;
        running.store(false);
        connected.store(false);
    } catch (...) {
        std::cerr << "[JobBroadcaster] Unknown exception" << std::endl;
        running.store(false);
        connected.store(false);
    }
}

void ConnectionMultiplexer::broadcastJobToWorkers(const json& job) {
    std::shared_lock<std::shared_mutex> lock(workers_mutex);
    
    std::string job_data = job.dump();
    
    for (const auto& worker : workers) {
        if (worker && worker->active && worker->resources && worker->resources->event_handler) {
            worker->resources->event_handler->postEvent(
                EventType::NEW_PUZZLE, "", job_data);
            worker->jobs_received++;
        }
    }
    
    stats.total_jobs_broadcast++;
}

void ConnectionMultiplexer::solutionSubmitterLoop() {
    try {
        LOG_DEBUG("[SolutionSubmitter] Started");

        while (running.load() && !stop_mining) {
            std::unique_ptr<SolutionSubmission> submission;

            {
                std::unique_lock<std::mutex> lock(submission_mutex);
                submission_cv.wait_for(lock, std::chrono::milliseconds(100), [this] {
                    return !submission_queue.empty() || !running.load() || stop_mining;
                });

                if ((!running.load() || stop_mining) && submission_queue.empty()) {
                    break;
                }

                if (!submission_queue.empty()) {
                    submission = std::move(submission_queue.front());
                    submission_queue.pop();
                }
            }

            if (submission) {
                bool result = processSubmission(*submission);
                submission->result_promise->set_value(result);
            }
        }

        LOG_DEBUG("[SolutionSubmitter] Stopped");
    } catch (const std::exception& e) {
        std::cerr << "[SolutionSubmitter] Exception: " << e.what() << std::endl;
        running.store(false);
        connected.store(false);
    } catch (...) {
        std::cerr << "[SolutionSubmitter] Unknown exception" << std::endl;
        running.store(false);
        connected.store(false);
    }
}

bool ConnectionMultiplexer::processSubmission(SolutionSubmission& submission) {
    if (!client || !client->is_connected()) {
        std::cout << "[GPU " << submission.gpu_id << "] " << Color::RED 
                  << "Cannot submit - not connected to node" << Color::RESET << std::endl;
        return false;
    }
    
    stats.total_solutions_submitted++;
    bool accepted = false;
    try {
        accepted = client->submitSolution(
            submission.proposal_id,
            submission.pow_solution,
            submission.solution_hash,
            submission.template_obj);
    } catch (const std::exception& e) {
        std::cerr << "[Submit] Exception: " << e.what() << std::endl;
        accepted = false;
    } catch (...) {
        std::cerr << "[Submit] Unknown exception" << std::endl;
        accepted = false;
    }
    
    if (accepted) {
        stats.total_solutions_accepted++;
    } else {
        stats.total_solutions_rejected++;
    }
    
    {
        std::lock_guard<std::mutex> time_lock(stats.time_mutex);
        stats.last_submission_time = std::chrono::steady_clock::now();
    }
    
    return accepted;
}

void ConnectionMultiplexer::healthMonitorLoop() {
    try {
        LOG_DEBUG("[HealthMonitor] Started");

        while (running.load() && !stop_mining) {
            std::this_thread::sleep_for(std::chrono::seconds(5));

            if (!running.load() || stop_mining) break;

            // Check connection health
            if (client) {
                bool was_connected = connected.load();
                bool is_now_connected = client->is_connected();

                if (was_connected && !is_now_connected) {
                    // Lost connection
                    std::cout << "[HealthMonitor] " << Color::YELLOW
                              << "Connection lost, attempting reconnection..." << Color::RESET << std::endl;

                    connected.store(false);
                    updateAllWorkersConnectionState(false);

                    if (attemptReconnection()) {
                        connected.store(true);
                        updateAllWorkersConnectionState(true);
                    }
                } else if (!was_connected && is_now_connected) {
                    // Connection restored (shouldn't happen normally)
                    connected.store(true);
                    updateAllWorkersConnectionState(true);
                }
            }

            // Check for stale jobs
            {
                std::lock_guard<std::mutex> time_lock(stats.time_mutex);
                auto now = std::chrono::steady_clock::now();
                auto time_since_job = std::chrono::duration_cast<std::chrono::seconds>(
                    now - stats.last_job_time).count();

                if (time_since_job > JOB_STALENESS_THRESHOLD_SEC && connected.load()) {
                    LOG_DEBUG("[HealthMonitor] Jobs stale (" << time_since_job << "s), will refresh on next poll");
                }
            }
        }

        LOG_DEBUG("[HealthMonitor] Stopped");
    } catch (const std::exception& e) {
        std::cerr << "[HealthMonitor] Exception: " << e.what() << std::endl;
        running.store(false);
        connected.store(false);
    } catch (...) {
        std::cerr << "[HealthMonitor] Unknown exception" << std::endl;
        running.store(false);
        connected.store(false);
    }
}

bool ConnectionMultiplexer::attemptReconnection() {
    int attempt = 0;
    
    while (running.load() && !stop_mining && attempt < MAX_CONSECUTIVE_RECONNECTIONS) {
        attempt++;
        
        int delay = std::min(MIN_RECONNECT_DELAY_SEC * (1 << (attempt - 1)), 
                            MAX_RECONNECT_DELAY_SEC);
        
        std::cout << "[Multiplexer] Reconnection attempt " << attempt 
                  << " in " << delay << "s..." << std::endl;
        
        for (int i = 0; i < delay && running.load() && !stop_mining; ++i) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        
        if (!running.load() || stop_mining) break;
        
        if (client && client->reconnect()) {
            std::cout << Color::GREEN << "[Multiplexer] Reconnected successfully!" 
                      << Color::RESET << std::endl;
            
            stats.reconnection_count++;
            return true;
        }
    }
    
    std::cout << Color::RED << "[Multiplexer] Failed to reconnect after " 
              << attempt << " attempts" << Color::RESET << std::endl;
    return false;
}

void ConnectionMultiplexer::tipMonitorLoop() {
    try {
        LOG_DEBUG("[TipMonitor] Started");
        
        // Check tip every 500ms for fast stale detection
        const auto TIP_CHECK_INTERVAL = std::chrono::milliseconds(500);
        
        while (running.load() && !stop_mining) {
            std::this_thread::sleep_for(TIP_CHECK_INTERVAL);
            
            if (!client || !client->is_connected()) {
                continue;
            }
            
            // Get current tip from node
            std::string new_tip = client->getTipDigest();
            if (new_tip.empty()) {
                continue;
            }
            
            bool tip_changed = false;
            {
                std::lock_guard<std::mutex> lock(job_mutex);
                if (!current_tip_digest.empty() && current_tip_digest != new_tip) {
                    tip_changed = true;
                    std::string short_old = current_tip_digest.length() > 16 ? 
                        current_tip_digest.substr(0, 8) + "..." + current_tip_digest.substr(current_tip_digest.length() - 8) : 
                        current_tip_digest;
                    std::string short_new = new_tip.length() > 16 ? 
                        new_tip.substr(0, 8) + "..." + new_tip.substr(new_tip.length() - 8) : 
                        new_tip;
                    std::cout << "[TipMonitor] " << Color::YELLOW << "New block detected!" << Color::RESET
                              << " Tip: " << short_old << " -> " << short_new << std::endl;
                }
                current_tip_digest = new_tip;
            }
            
            if (tip_changed) {
                // Immediately pause all workers - their templates are now stale
                {
                    std::shared_lock<std::shared_mutex> lock(workers_mutex);
                    for (const auto& worker : workers) {
                        if (worker && worker->active && worker->resources) {
                            worker->resources->set_paused(true);
                            if (worker->resources->event_handler) {
                                worker->resources->event_handler->postEvent(EventType::TIP_CHANGED);
                            }
                        }
                    }
                }
                
                // Clear the last template ID to force immediate fetch
                {
                    std::lock_guard<std::mutex> lock(job_mutex);
                    last_template_id.clear();
                }
                
                // Immediately fetch new template
                json template_response = client->getBlockTemplate(wallet_address);
                
                if (!template_response.empty() && template_response.contains("result")) {
                    json result = template_response["result"];
                    
                    if (result.contains("template") && !result["template"].is_null()) {
                        json template_obj = result["template"];
                        
                        if (template_obj.contains("metadata") && !template_obj["metadata"].is_null()) {
                            json metadata = template_obj["metadata"];
                            std::string template_id = metadata.contains("digest") ? 
                                metadata.value("digest", "") : "";
                            
                            {
                                std::lock_guard<std::mutex> lock(job_mutex);
                                last_template_id = template_id;
                            }
                            
                            stats.total_jobs_fetched++;
                            {
                                std::lock_guard<std::mutex> time_lock(stats.time_mutex);
                                stats.last_job_time = std::chrono::steady_clock::now();
                            }
                            
                            std::cout << "[TipMonitor] " << Color::GREEN << "New template fetched" << Color::RESET << std::endl;
                            broadcastJobToWorkers(template_response);
                        }
                    }
                }
            }
        }
        
        LOG_DEBUG("[TipMonitor] Stopped");
    } catch (const std::exception& e) {
        std::cerr << "[TipMonitor] Exception: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "[TipMonitor] Unknown exception" << std::endl;
    }
}
