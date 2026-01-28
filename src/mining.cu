#include "mining.cuh"
#include "rpc_client.cuh"
#include "stratum_client.cuh"
#include "common.cuh"

UnifiedMiningController* g_mining_controller = nullptr;

UnifiedMiningController::UnifiedMiningController(
    int gpu_id,
    GpuResources* resources,
    const std::string& endpoint,
    MiningMode mode,
    const std::string& stratum_pass)
    : endpoint(endpoint)
    , gpu_id(gpu_id)
    , gpu_resources(resources)
    , mining_mode(mode)
    , stratum_password(stratum_pass) {
}

UnifiedMiningController::~UnifiedMiningController() {
    stop();
}

void UnifiedMiningController::start() {
    gpu_resources->event_handler = std::make_unique<EventHandler>();
    gpu_resources->mining_mode = mining_mode;
    
    // Create appropriate client based on mining mode
    MiningClient* new_client = nullptr;
    if (mining_mode == MiningMode::Stratum) {
        std::string host;
        int port;
        if (parse_stratum_url(endpoint, host, port)) {
            StratumConfig config;
            config.host = host;
            config.port = port;
            config.username = g_miner_wallet_address;
            config.password = stratum_password;
            config.worker_name = "xnt-miner-gpu" + std::to_string(gpu_id);
            new_client = new StratumClient(config);
        } else {
            std::cerr << Color::RED << "Invalid stratum URL: " << endpoint << Color::RESET << std::endl;
            return;
        }
    } else {
        new_client = new NeptuneCudaMinerClient(endpoint, g_miner_wallet_address);
    }
    
    {
        std::lock_guard<std::mutex> lock(gpu_resources->state_mutex);
        gpu_resources->client = new_client;
    }
    
    int retry_count = 0;
    bool connected = false;
    
    while (!connected && !stop_mining && !gpu_resources->gpu_stop_flag) {
        MiningClient* client_ptr = nullptr;
        {
            std::lock_guard<std::mutex> lock(gpu_resources->state_mutex);
            client_ptr = gpu_resources->client;
        }
        
        if (!client_ptr) break;
        
        if (gpu_id == 0) {
            const char* mode_str = mining_mode == MiningMode::Stratum ? "stratum server" : "RPC server";
            std::cout << "Attempting to connect to " << mode_str << " at " << endpoint << "..." << std::endl;
        }
        if (client_ptr->connect()) {
            connected = true;
            if (gpu_id == 0) {
                const char* mode_str = mining_mode == MiningMode::Stratum ? "stratum server" : "RPC server";
                std::cout << "Successfully connected to " << mode_str << "!" << std::endl;
            }
            break;
        }
        
        retry_count++;
        
        if (gpu_id == 0) {
            std::cout << "Connection attempt " << retry_count << " failed, retrying in 10 seconds..." << std::endl;
        }
        
        for (int i = 0; i < 10 && !stop_mining && !gpu_resources->gpu_stop_flag; ++i) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }
    
    if (!connected) {
        if (gpu_id == 0) {
            std::cout << "Connection aborted after " << retry_count << " attempts" << std::endl;
        }
        
        std::lock_guard<std::mutex> lock(gpu_resources->state_mutex);
        delete gpu_resources->client;
        gpu_resources->client = nullptr;
        gpu_resources->gpu_node_connected = false;
        return;
    }
    
    gpu_resources->gpu_node_connected = true;
    
    if (gpu_id == 0) {
        std::cout << "Initializing CUDA..." << std::endl;
    }
    
    if (!initializeCuda()) {
        if (gpu_id == 0) {
            std::cout << Color::RED << "Failed to initialize CUDA for GPU " << gpu_id << Color::RESET << std::endl;
        }
        LOG_DEBUG("Failed to initialize CUDA for GPU " << gpu_id);
        return;
    }
    
    if (gpu_id == 0) {
        std::cout << Color::GREEN << "CUDA initialized successfully" << Color::RESET << std::endl;
        std::cout << "Starting mining controller..." << std::endl;
    }
    
    controller_running = true;
    
    // Use appropriate fetcher based on mining mode
    if (mining_mode == MiningMode::Stratum) {
        fetcher_thread = std::thread(&UnifiedMiningController::stratumFetcherLoop, this);
    } else {
        fetcher_thread = std::thread(&UnifiedMiningController::fetcherLoop, this);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    if (gpu_id == 0) {
        std::cout << "Mining loop started. Waiting for block proposals..." << std::endl;
    }
    
    miningLoop();
    
    if (fetcher_thread.joinable()) {
        fetcher_thread.join();
    }
    {
        std::lock_guard<std::mutex> lock(gpu_resources->state_mutex);
        if (gpu_resources->client) {
            delete gpu_resources->client;
            gpu_resources->client = nullptr;
        }
    }
}

void UnifiedMiningController::stop() {
    controller_running = false;
    gpu_resources->gpu_stop_flag = true;
    
    if (gpu_resources->event_handler) {
        gpu_resources->event_handler->shutdown();
    }
}

bool UnifiedMiningController::initializeCuda() {
    cudaError_t err = cudaSetDevice(gpu_id);
    if (err != cudaSuccess) {
        LOG_ERROR("cudaSetDevice", err);
        return false;
    }
    
    // Get device properties
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, gpu_id);
    if (err != cudaSuccess) {
        LOG_ERROR("cudaGetDeviceProperties", err);
        return false;
    }
    
    // Store GPU info
    gpu_resources->gpu_name = prop.name;
    gpu_resources->gpu_vram_total = prop.totalGlobalMem;
    gpu_resources->gpu_uuid = get_gpu_uuid(gpu_id);
    
    // Use default values (database removed)
    gpu_resources->optimal_max_nonces = 1000000ULL;
    
    return true;
}

bool UnifiedMiningController::connectToNode() {
    MiningClient* client_ptr = nullptr;
    {
        std::lock_guard<std::mutex> lock(gpu_resources->state_mutex);
        client_ptr = gpu_resources->client;
    }
    
    if (!client_ptr) return false;
    return client_ptr->connect();
}

bool UnifiedMiningController::reconnect_to_node() {
    // Check if reconnection already in progress
    bool expected = false;
    if (!gpu_resources->gpu_reconnection_in_progress.compare_exchange_strong(expected, true)) {
        // Wait for existing reconnection
        int wait_count = 0;
        while (gpu_resources->gpu_reconnection_in_progress && !stop_mining && wait_count < 120) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            wait_count++;
        }
        return gpu_resources->gpu_node_connected;
    }
    
    // Pause mining during reconnection
    gpu_resources->set_paused(true);
    
    // RPC doesn't require explicit disconnection
    
    gpu_resources->gpu_node_connected = false;
    
    // Attempt reconnection with backoff
    int attempt = 0;
    bool connected = false;
    
    const int MAX_RECONNECT_ATTEMPTS = 10;
    while (!connected && !stop_mining && !gpu_resources->gpu_stop_flag && attempt < MAX_RECONNECT_ATTEMPTS) {
        attempt++;
        
        // Calculate backoff delay
        int delay = std::min(MIN_RECONNECT_DELAY_SEC * (1 << (attempt - 1)), MAX_RECONNECT_DELAY_SEC);
        
        LOG_DEBUG("[GPU " << gpu_id << "] Reconnection attempt " << attempt 
                  << " in " << delay << "s...");
        
        // Wait with early exit on stop signal
        for (int i = 0; i < delay && !stop_mining && !gpu_resources->gpu_stop_flag; ++i) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        
        if (stop_mining || gpu_resources->gpu_stop_flag) break;
        
        // Try to connect
        MiningClient* client_ptr = nullptr;
        {
            std::lock_guard<std::mutex> lock(gpu_resources->state_mutex);
            client_ptr = gpu_resources->client;
        }
        
        if (client_ptr && client_ptr->connect()) {
            connected = true;
            gpu_resources->gpu_node_connected = true;
            LOG_DEBUG("[GPU " << gpu_id << "] Reconnected successfully");
        }
    }
    
    gpu_resources->gpu_reconnection_in_progress = false;
    
    if (connected) {
        gpu_resources->set_paused(false);
    }
    
    return connected;
}

void UnifiedMiningController::fetcherLoop() {
    puzzleFetcher(gpu_resources, this);
}

void UnifiedMiningController::stratumFetcherLoop() {
    if (!gpu_resources) return;
    
    if (gpu_id == 0) {
        std::cout << "[Stratum] Job fetcher started (push-based)" << std::endl;
    }
    
    while (!stop_mining && !gpu_resources->gpu_stop_flag && controller_running) {
        MiningClient* client_ptr = nullptr;
        {
            std::lock_guard<std::mutex> lock(gpu_resources->state_mutex);
            client_ptr = gpu_resources->client;
        }
        
        if (!client_ptr || !client_ptr->is_connected()) {
            if (!gpu_resources->gpu_reconnection_in_progress) {
                if (gpu_id == 0) {
                    static auto last_reconnect_log = std::chrono::steady_clock::now();
                    auto now = std::chrono::steady_clock::now();
                    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - last_reconnect_log).count();
                    if (elapsed >= 10) {
                        std::cout << "[Stratum] " << Color::YELLOW << "Disconnected, attempting reconnect..." << Color::RESET << std::endl;
                        last_reconnect_log = now;
                    }
                }
                gpu_resources->event_handler->postEvent(EventType::RECONNECT);
            }
            std::this_thread::sleep_for(std::chrono::seconds(1));
            continue;
        }
        
        // Wait for job notification from stratum server
        json job_response;
        if (client_ptr->wait_for_job(job_response, 5000)) {
            // Got a new job
            if (!job_response.empty()) {
                std::string job_data = job_response.dump();
                gpu_resources->event_handler->postEvent(EventType::NEW_PUZZLE, "", job_data);
                gpu_resources->update_job_received();
            }
        }
        
        // Check for job staleness
        if (gpu_resources->get_time_since_last_job() > JOB_STALENESS_THRESHOLD_SEC * 2) {
            if (gpu_id == 0) {
                std::cout << "[Stratum] " << Color::YELLOW << "No jobs received for a while, connection may be stale" << Color::RESET << std::endl;
            }
        }
    }
    
    if (gpu_id == 0) {
        std::cout << "[Stratum] Job fetcher stopped" << std::endl;
    }
}

void UnifiedMiningController::miningLoop() {
    while (controller_running && !stop_mining && !gpu_resources->gpu_stop_flag) {
        MiningEvent event;
        bool has_event = gpu_resources->event_handler->waitForEvent(
            event, std::chrono::milliseconds(1000));
        
        if (!has_event) {
            if (!gpu_resources->gpu_node_connected && !gpu_resources->gpu_reconnection_in_progress) {
                reconnect_to_node();
            }
            continue;
        }
        
        switch (event.type) {
            case EventType::NEW_PUZZLE:
                handleNewPuzzle(event);
                break;
                
            case EventType::SOLUTION_FOUND:
                handleSolutionFound(event);
                break;
                
            case EventType::STOP_MINING:
                controller_running = false;
                break;
                
            case EventType::ERROR_EVENT:
                handleError(event);
                break;
                
            case EventType::RECONNECT:
                reconnect_to_node();
                break;
                
            default:
                break;
        }
    }
}

void UnifiedMiningController::handleNewPuzzle(const MiningEvent& event) {
    if (event.data.empty()) {
        std::cout << "[GPU " << gpu_id << "] " << Color::RED << "Empty puzzle data" << Color::RESET << std::endl;
        return;
    }
    
    PowPuzzle puzzle = parsePowPuzzle(event.data);
    if (!puzzle.is_valid()) {
        std::cout << "[GPU " << gpu_id << "] " << Color::RED << "Invalid puzzle from parsePowPuzzle" << Color::RESET << std::endl;
        return;
    }
    
    // Parse and store the template for this proposal
    json template_response = json::parse(event.data);
    json template_obj;
    if (template_response.contains("result") && template_response["result"].contains("template")) {
        template_obj = template_response["result"]["template"];
    } else if (template_response.contains("template")) {
        template_obj = template_response["template"];
    }
    
    {
        std::lock_guard<std::mutex> lock(gpu_resources->state_mutex);
        if (puzzle.id == gpu_resources->current_proposal_id) {
            return;  // Same puzzle, skip
        }
        gpu_resources->current_proposal_id = puzzle.id;
        gpu_resources->current_template = template_obj;  // Store template for this proposal
        Digest original_target = hex_to_digest(puzzle.threshold);
        gpu_resources->current_real_target = original_target;
        if (g_test_mode) {
            // Make target 100000x easier for testing
            gpu_resources->current_target = make_target_easier(original_target, 100000);
            std::cout << "[GPU " << gpu_id << "] " << Color::YELLOW 
                      << "TEST MODE: Target made 100000x easier" << Color::RESET << std::endl;
        } else {
            gpu_resources->current_target = original_target;
        }
    }
    
    resetNonceCounter(gpu_resources, puzzle.id);
    
    Digest prev_block = hex_to_digest(puzzle.prev_block);
    PowMastPaths mast_paths = convertToPowMastPaths(puzzle.auth_paths);
    
    // Check if we can reuse existing buffer (same prev_block and MAST paths)
    bool need_preprocess = true;
    {
        std::lock_guard<std::mutex> lock(gpu_resources->state_mutex);
        if (gpu_resources->buffer && gpu_resources->buffer->is_valid()) {
            // Check if prev_block matches
            bool prev_block_match = true;
            for (int i = 0; i < DIGEST_LEN; ++i) {
                if (gpu_resources->cached_prev_block.values[i] != prev_block.values[i]) {
                    prev_block_match = false;
                    break;
                }
            }
            
            // Check if MAST paths match (for XNT consensus, commitment depends on MAST paths)
            bool mast_paths_match = true;
            if (prev_block_match) {
                for (int i = 0; i < 3; ++i) {
                    for (int j = 0; j < DIGEST_LEN; ++j) {
                        if (gpu_resources->cached_mast_paths.pow[i].values[j] != mast_paths.pow[i].values[j]) {
                            mast_paths_match = false;
                            break;
                        }
                    }
                    if (!mast_paths_match) break;
                }
                if (mast_paths_match) {
                    for (int i = 0; i < 2; ++i) {
                        for (int j = 0; j < DIGEST_LEN; ++j) {
                            if (gpu_resources->cached_mast_paths.header[i].values[j] != mast_paths.header[i].values[j]) {
                                mast_paths_match = false;
                                break;
                            }
                        }
                        if (!mast_paths_match) break;
                    }
                }
                if (mast_paths_match) {
                    for (int j = 0; j < DIGEST_LEN; ++j) {
                        if (gpu_resources->cached_mast_paths.kernel[0].values[j] != mast_paths.kernel[0].values[j]) {
                            mast_paths_match = false;
                            break;
                        }
                    }
                }
            }
            
            if (prev_block_match && mast_paths_match) {
                need_preprocess = false;
                // Update MAST paths in buffer and cache (for consistency)
                gpu_resources->buffer->mast_paths = mast_paths;
                gpu_resources->cached_mast_paths = mast_paths;
                std::cout << "[GPU " << gpu_id << "] " << Color::GREEN 
                          << "Reusing cached buffer (same prev_block and MAST paths)" << Color::RESET << std::endl;
            }
        }
    }
    
    if (need_preprocess) {
        auto preprocess_start = std::chrono::steady_clock::now();
        
        if (!preprocessPuzzle(puzzle, gpu_resources)) {
            std::cout << "[GPU " << gpu_id << "] " << Color::RED << "Preprocessing failed" << Color::RESET << std::endl;
            return;
        }
        
        auto preprocess_end = std::chrono::steady_clock::now();
        auto preprocess_duration = std::chrono::duration_cast<std::chrono::milliseconds>(preprocess_end - preprocess_start).count();
        double preprocess_seconds = preprocess_duration / 1000.0;
        
        std::cout << "[GPU " << gpu_id << "] " << Color::GREEN << "Finished preprocessing" << Color::RESET 
                  << " (" << std::fixed << std::setprecision(2) << preprocess_seconds << "s)" << std::endl;
        
        // Update cache
        {
            std::lock_guard<std::mutex> lock(gpu_resources->state_mutex);
            gpu_resources->cached_prev_block = prev_block;
            gpu_resources->cached_mast_paths = mast_paths;
        }
    }
    
    gpu_resources->update_job_received();
    gpu_resources->set_paused(false);
    
    // Log new job received
    std::string short_id = puzzle.id.length() > 20 ? puzzle.id.substr(0, 12) + "..." + puzzle.id.substr(puzzle.id.length() - 8) : puzzle.id;
    std::cout << "[GPU " << gpu_id << "] " << Color::CYAN << Color::BOLD 
              << "✓ New job received" << Color::RESET 
              << " | Template ID: " << Color::CYAN << short_id << Color::RESET << std::endl;
    
    continuousMiningLoop(gpu_resources, this);
}

void UnifiedMiningController::handleSolutionFound(const MiningEvent& event) {
    LOG_DEBUG("[GPU " << gpu_id << "] Solution found event received");
}

void UnifiedMiningController::handleError(const MiningEvent& event) {
    LOG_DEBUG("[GPU " << gpu_id << "] Error: " << event.data);
    
    if (event.data.find("connection") != std::string::npos ||
        event.data.find("disconnect") != std::string::npos) {
        reconnect_to_node();
    }
}

json UnifiedMiningController::requestJob() {
    MiningClient* client_ptr = nullptr;
    {
        std::lock_guard<std::mutex> lock(gpu_resources->state_mutex);
        client_ptr = gpu_resources->client;
    }
    
    if (!client_ptr || !client_ptr->is_connected()) {
        return json();
    }
    
    return client_ptr->getBlockTemplate();
}

bool UnifiedMiningController::processJobResponse(const json& job_response) {
    if (job_response.empty()) return false;
    
    if (job_response.contains("error")) {
        LOG_DEBUG("[GPU " << gpu_id << "] Job error: " << job_response["error"].get<std::string>());
        return false;
    }
    std::string job_data = job_response.dump();
    gpu_resources->event_handler->postEvent(EventType::NEW_PUZZLE, "", job_data);
    
    return true;
}

// ===== MULTI-GPU MANAGER IMPLEMENTATION =====

MultiGpuManager::MultiGpuManager(
    const std::string& endpoint,
    int specific_gpu,
    MiningMode mode,
    const std::string& stratum_pass)
    : endpoint(endpoint)
    , single_gpu_id(specific_gpu)
    , mining_mode(mode)
    , stratum_password(stratum_pass) {
}

MultiGpuManager::~MultiGpuManager() {
    stopAll();
}

bool MultiGpuManager::detectAndInitGpus() {
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    
    if (error != cudaSuccess || device_count == 0) {
        std::cerr << "No CUDA devices found" << std::endl;
        return false;
    }
    
    // Determine which GPUs to use
    if (single_gpu_id >= 0) {
        // Single GPU mode
        if (single_gpu_id >= device_count) {
            std::cerr << "GPU " << single_gpu_id << " not found (max: " << device_count - 1 << ")" << std::endl;
            return false;
        }
        gpu_ids.push_back(single_gpu_id);
    } else {
        // All GPUs mode
        for (int i = 0; i < device_count; ++i) {
            gpu_ids.push_back(i);
        }
    }
    
    for (int gpu_id : gpu_ids) {
        if (!initializeGpu(gpu_id)) {
            std::cerr << "Failed to initialize GPU " << gpu_id << std::endl;
            continue;
        }
    }
    
    g_total_gpu_count = static_cast<int>(gpu_resources.size());
    
    {
        std::lock_guard<std::mutex> lock(g_all_gpu_resources_mutex);
        for (auto& res : gpu_resources) {
            g_all_gpu_resources.push_back(res.get());
        }
    }
    
    return !gpu_resources.empty();
}

bool MultiGpuManager::initializeGpu(int device_id) {
    auto gpu_res = std::make_unique<GpuResources>(device_id);
    
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device_id);
    if (err != cudaSuccess) {
        return false;
    }
    
    size_t vram_gb = prop.totalGlobalMem / (1024ULL * 1024ULL * 1024ULL);
    if (vram_gb < 8) {
        LOG_DEBUG("GPU " << device_id << " has insufficient VRAM (" << vram_gb << " GB)");
        return false;
    }
    
    gpu_res->gpu_name = prop.name;
    gpu_res->gpu_vram_total = prop.totalGlobalMem;
    gpu_res->gpu_uuid = get_gpu_uuid(device_id);
    gpu_res->optimal_max_nonces = 1000000ULL;
    gpu_res->mining_mode = mining_mode;
    
    auto controller = std::make_unique<UnifiedMiningController>(
        device_id, gpu_res.get(), endpoint, mining_mode, stratum_password
    );
    
    gpu_resources.push_back(std::move(gpu_res));
    controllers.push_back(std::move(controller));
    
    std::cout << "Initialized GPU " << device_id << ": " << prop.name 
              << " (" << vram_gb << " GB)" << std::endl;
    
    return true;
}

void MultiGpuManager::startAll() {
    if (gpu_resources.empty()) {
        std::cerr << "No GPUs initialized" << std::endl;
        return;
    }
    
    for (size_t i = 0; i < controllers.size(); ++i) {
        gpu_threads.emplace_back(&MultiGpuManager::gpuMiningThread, this, i);
        if (i < controllers.size() - 1) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
    }
    
    for (auto& thread : gpu_threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

void MultiGpuManager::stopAll() {
    stop_mining = true;
    
    for (auto& controller : controllers) {
        controller->stop();
    }
    
    // Wait for threads
    for (auto& thread : gpu_threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    
    for (auto& res : gpu_resources) {
        if (res->buffer) {
            res->buffer->cleanup();
            res->buffer.reset();
        }
    }
    
    {
        std::lock_guard<std::mutex> lock(g_all_gpu_resources_mutex);
        g_all_gpu_resources.clear();
    }
}

GpuResources* MultiGpuManager::getGpuResources(size_t index) {
    if (index >= gpu_resources.size()) return nullptr;
    return gpu_resources[index].get();
}

void MultiGpuManager::gpuMiningThread(size_t index) {
    if (index >= controllers.size()) return;
    controllers[index]->start();
}

void startUnifiedMining(
    const std::string& endpoint,
    int specific_gpu,
    MiningMode mode,
    const std::string& stratum_pass) {
    
    MultiGpuManager manager(endpoint, specific_gpu, mode, stratum_pass);
    
    if (!manager.detectAndInitGpus()) {
        std::cerr << "Failed to initialize GPUs" << std::endl;
        return;
    }
    
    try {
        manager.startAll();
    } catch (const std::exception& e) {
        std::cerr << "Mining error: " << e.what() << std::endl;
    }
    
    cleanup_gpu_memory();
}

bool preprocessPuzzle(const PowPuzzle& puzzle, GpuResources* gpu_res) {
    if (!gpu_res) return false;
    
    cudaError_t err = cudaSetDevice(gpu_res->gpu_id);
    if (err != cudaSuccess) {
        LOG_ERROR("cudaSetDevice in preprocess", err);
        return false;
    }
    
    PowMastPaths mast_paths = convertToPowMastPaths(puzzle.auth_paths);
    Digest prev_block = hex_to_digest(puzzle.prev_block);
    auto buffer = Pow::preprocess(mast_paths, prev_block, puzzle.consensus_rule_set, nullptr);
    
    if (!buffer.is_valid()) {
        LOG_DEBUG("[GPU " << gpu_res->gpu_id << "] Preprocessing failed");
        return false;
    }
    
    gpu_res->buffer = std::make_unique<GuesserBuffer>(std::move(buffer));
    return true;
}

bool minePuzzleWithCuda(const PowPuzzle& puzzle, GpuResources* gpu_res) {
    if (!gpu_res || !gpu_res->buffer || !gpu_res->buffer->is_valid()) {
        return false;
    }
    
    cudaError_t err = cudaSetDevice(gpu_res->gpu_id);
    if (err != cudaSuccess) {
        return false;
    }
    
    Digest original_target = hex_to_digest(puzzle.threshold);
    Digest target = g_test_mode ? make_target_easier(original_target, 100000) : original_target;
    PowMastPaths mast_paths = convertToPowMastPaths(puzzle.auth_paths);
    uint64_t start_nonce = getNextNonceRange(gpu_res, gpu_res->optimal_max_nonces);
    
    auto result = mine_pow_with_buffer(
        *gpu_res->buffer,
        target,
        mast_paths,
        start_nonce,
        gpu_res->optimal_max_nonces,
        puzzle.consensus_rule_set,
        nullptr
    );
    
    // Update nonce counter
    gpu_res->total_nonces_tested.fetch_add(gpu_res->optimal_max_nonces);
    
    return result.has_value();
}

bool continuousMiningLoop(GpuResources* gpu_res, UnifiedMiningController* controller) {
    if (!gpu_res || !controller) return false;
    
    auto last_status_time = std::chrono::steady_clock::now();
    const auto STATUS_UPDATE_INTERVAL = std::chrono::seconds(5);
    
    while (!stop_mining && !gpu_res->gpu_stop_flag) {
        if (gpu_res->gpu_pause_flag) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }
        
        if (gpu_res->event_handler && gpu_res->event_handler->hasEvents()) {
            break;
        }
        
        if (!gpu_res->buffer || !gpu_res->buffer->is_valid()) {
            gpu_res->set_paused(true);
            continue;
        }
        
        std::string proposal_id;
        {
            std::lock_guard<std::mutex> lock(gpu_res->state_mutex);
            proposal_id = gpu_res->current_proposal_id;
        }
        
        if (proposal_id.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }
        
        auto start_time = std::chrono::high_resolution_clock::now();
        uint64_t start_nonce = getNextNonceRange(gpu_res, gpu_res->optimal_max_nonces);
        Digest target;
        {
            std::lock_guard<std::mutex> lock(gpu_res->state_mutex);
            target = gpu_res->current_target;
        }
        
        auto result = mine_pow_with_buffer(
            *gpu_res->buffer,
            target,
            gpu_res->buffer->mast_paths,
            start_nonce,
            gpu_res->optimal_max_nonces,
            gpu_res->buffer->consensus_rule_set,
            nullptr
        );
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        
        // Update nonce counter
        gpu_res->total_nonces_tested.fetch_add(gpu_res->optimal_max_nonces);
        
        if (duration_ms > 0) {
            double hashrate_ms = static_cast<double>(gpu_res->optimal_max_nonces) / duration_ms;
            gpu_res->hash_tracker.add(hashrate_ms);
        }
        
        // Periodic status update
        auto now = std::chrono::steady_clock::now();
        if (now - last_status_time >= STATUS_UPDATE_INTERVAL) {
            double avg_hashrate = gpu_res->hash_tracker.get_average();
            double hashrate_hps = avg_hashrate * 1000.0; // Convert from K nonces/ms to H/s
            std::string hashrate_str = format_hashrate(hashrate_hps);
            
            uint64_t total_nonces = gpu_res->total_nonces_tested.load();
            uint64_t accepted = gpu_res->solutions_accepted.load();
            uint64_t rejected = gpu_res->solutions_rejected.load();
            
            std::cout << "[GPU " << gpu_res->gpu_id << "] " << Color::GREEN 
                      << "Mining..." << Color::RESET 
                      << " | Hash Rate: " << Color::YELLOW << hashrate_str << Color::RESET
                      << " | Nonces: " << total_nonces
                      << " | " << Color::GREEN << accepted << Color::RESET 
                      << " / " << Color::RED << rejected << Color::RESET 
                      << " (success / reject)" << std::endl;
            
            last_status_time = now;
        }
        
        if (result.has_value()) {
            gpu_res->solutions_found++;
            const char* submit_target = gpu_res->is_stratum_mode() ? "pool" : "node";
            std::cout << "[GPU " << gpu_res->gpu_id << "] " << Color::YELLOW << Color::BOLD
                      << "*** SOLUTION FOUND! ***" << Color::RESET 
                      << " Submitting to " << submit_target << "..." << std::endl;
            
            MiningClient* client = nullptr;
            {
                std::lock_guard<std::mutex> lock(gpu_res->state_mutex);
                client = gpu_res->client;
            }
            
            if (client && client->is_connected()) {
                PowMastPaths mast_paths = gpu_res->buffer->mast_paths;
                Digest solution_hash = mast_paths.fast_mast_hash(result.value());
                
                // Get the template for this proposal
                json template_obj;
                {
                    std::lock_guard<std::mutex> lock(gpu_res->state_mutex);
                    template_obj = gpu_res->current_template;
                }
                
                bool accepted = client->submit_solution(
                    proposal_id,
                    result.value(),
                    solution_hash,
                    template_obj
                );
                
                if (accepted) {
                    gpu_res->solutions_accepted++;
                    const char* accepted_msg = gpu_res->is_stratum_mode() ? "SHARE ACCEPTED" : "BLOCK ACCEPTED";
                    std::cout << "[GPU " << gpu_res->gpu_id << "] " << Color::GREEN << Color::BOLD 
                              << "*** " << accepted_msg << "! ***" << Color::RESET 
                      << " | Total: " << Color::GREEN << gpu_res->solutions_accepted.load() << Color::RESET << std::endl;
                } else {
                    gpu_res->solutions_rejected++;
                }
            } else {
                gpu_res->solutions_rejected++;
                std::cout << "[GPU " << gpu_res->gpu_id << "] " << Color::RED 
                          << "Cannot submit - not connected" << Color::RESET << std::endl;
            }
        }
    }
    
    return true;
}

// ===== NONCE MANAGEMENT =====

uint64_t getNextNonceRange(GpuResources* gpu_res, uint64_t batch_size) {
    if (!gpu_res) return 0;
    
    uint64_t current = gpu_res->gpu_puzzle_nonce_counter.fetch_add(batch_size);
    return gpu_res->gpu_puzzle_random_start + current;
}

void resetNonceCounter(GpuResources* gpu_res, const std::string& puzzle_id) {
    if (!gpu_res) return;
    
    // Generate new random start for this puzzle
    uint64_t random_start = generate_secure_random_start(
        puzzle_id,
        gpu_res->gpu_id,
        gpu_res->gpu_uuid
    );
    
    gpu_res->gpu_puzzle_random_start = random_start;
    gpu_res->gpu_puzzle_nonce_counter = 0;
}

// ===== SIGNAL HANDLING =====

void signal_handler(int signal) {
    if (signal == SIGINT) {
        std::cout << "\n\nReceived interrupt signal, shutting down..." << std::endl;
        stop_mining = true;
        
        std::lock_guard<std::mutex> lock(g_all_gpu_resources_mutex);
        for (GpuResources* gpu : g_all_gpu_resources) {
            if (gpu && gpu->event_handler) {
                gpu->event_handler->postEvent(EventType::STOP_MINING);
            }
        }
    }
}

void install_signal_handlers() {
    signal(SIGINT, signal_handler);
#ifndef _WIN32
    signal(SIGTERM, signal_handler);
#endif
}

void puzzleFetcher(GpuResources* gpu_res, UnifiedMiningController* controller) {
    if (!gpu_res || !controller) return;
    
    if (gpu_res->gpu_id == 0) {
        std::cout << "[Fetcher] Block proposal fetcher started" << std::endl;
    }
    
    auto last_poll_time = std::chrono::steady_clock::now();
    std::string last_template_id;
    
    while (!stop_mining && !gpu_res->gpu_stop_flag && controller->isRunning()) {
        auto now = std::chrono::steady_clock::now();
        auto time_since_poll = std::chrono::duration_cast<std::chrono::seconds>(
            now - last_poll_time).count();
        
        bool should_poll = false;
        
        if (time_since_poll >= g_fetch_interval_sec) {
            should_poll = true;
        }
        
        {
            std::lock_guard<std::mutex> lock(gpu_res->state_mutex);
            if (gpu_res->current_proposal_id.empty()) {
                should_poll = true;
            }
        }
        
        if (gpu_res->get_time_since_last_job() > JOB_STALENESS_THRESHOLD_SEC) {
            should_poll = true;
        }
        
        if (should_poll) {
            MiningClient* client = nullptr;
            {
                std::lock_guard<std::mutex> lock(gpu_res->state_mutex);
                client = gpu_res->client;
            }
            
            if (client && client->is_connected()) {
                json template_response = client->getBlockTemplate();
                
                if (!template_response.empty() && template_response.contains("result")) {
                    json result = template_response["result"];
                    if (!result.contains("template") || result["template"].is_null()) {
                        // Template is null - node may be syncing
                        if (gpu_res->gpu_id == 0) {
                            static auto last_null_log = std::chrono::steady_clock::now();
                            auto now = std::chrono::steady_clock::now();
                            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - last_null_log).count();
                            if (elapsed >= 30) {  // Log every 30 seconds to avoid spam
                                std::cout << "[Fetcher] Waiting for block proposal (node may be syncing)..." << std::endl;
                                last_null_log = now;
                            }
                        }
                        continue;
                    }
                    json template_obj = result["template"];
                    if (!template_obj.contains("metadata") || template_obj["metadata"].is_null()) {
                        continue;
                    }
                    json metadata = template_obj["metadata"];
                    std::string template_id = metadata.contains("digest") && !metadata["digest"].is_null() 
                        ? metadata.value("digest", "") : "";
                    
                    if (template_id != last_template_id && !template_id.empty()) {
                        // For solo mode, verify template is not stale by checking against tip
                        NeptuneCudaMinerClient* solo_client = dynamic_cast<NeptuneCudaMinerClient*>(client);
                        XntRpcClient* rpc_client = solo_client ? solo_client->get_rpc_client() : nullptr;
                        
                        std::string tip_digest;
                        if (rpc_client) {
                            tip_digest = rpc_client->getTipDigest();
                        }
                        
                        // Handle both snake_case (prev_block) and camelCase (prevBlock)
                        std::string prev_block;
                        if (metadata.contains("prev_block") && !metadata["prev_block"].is_null()) {
                            prev_block = metadata.value("prev_block", "");
                        } else if (metadata.contains("prevBlock") && !metadata["prevBlock"].is_null()) {
                            prev_block = metadata.value("prevBlock", "");
                        }
                        
                        // If we can't get tip_digest (no RPC client), skip the staleness check
                        if (tip_digest.empty() || tip_digest == prev_block) {
                            PowPuzzle puzzle = parseRpcTemplate(template_response);
                            
                            if (puzzle.is_valid()) {
                                if (solo_client) {
                                    solo_client->cache_puzzle(template_obj);
                                }
                                // Pass full response so parsePowPuzzle can find result.template
                                std::string puzzle_data = template_response.dump();
                                gpu_res->event_handler->postEvent(EventType::NEW_PUZZLE, "", puzzle_data);
                                gpu_res->update_job_received();
                                last_template_id = template_id;
                                
                            } else {
                                std::cout << "[GPU " << gpu_res->gpu_id << "] " << Color::RED 
                                          << "✗ Block proposal invalid (missing required fields)" << Color::RESET << std::endl;
                            }
                        } else {
                            LOG_DEBUG("[GPU " << gpu_res->gpu_id << "] Template outdated, skipping");
                        }
                    }
                } else if (!template_response.empty() && template_response.contains("error")) {
                    if (gpu_res->gpu_id == 0) {
                        std::cerr << Color::RED << "[Fetcher] RPC error: " 
                                  << template_response["error"].value("message", "Unknown error") 
                                  << Color::RESET << std::endl;
                    }
                } else if (template_response.empty()) {
                    if (gpu_res->gpu_id == 0) {
                        static auto last_empty_log = std::chrono::steady_clock::now();
                        auto now = std::chrono::steady_clock::now();
                        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - last_empty_log).count();
                        if (elapsed >= 30) {
                            std::cout << "[Fetcher] " << Color::YELLOW << "Empty response from RPC" << Color::RESET << std::endl;
                            last_empty_log = now;
                        }
                    }
                }
                
                last_poll_time = std::chrono::steady_clock::now();
            } else {
                if (!gpu_res->gpu_reconnection_in_progress) {
                    if (gpu_res->gpu_id == 0) {
                        static auto last_reconnect_log = std::chrono::steady_clock::now();
                        auto now = std::chrono::steady_clock::now();
                        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - last_reconnect_log).count();
                        if (elapsed >= 10) {
                            std::cout << "[Fetcher] " << Color::YELLOW << "Not connected, attempting reconnect..." << Color::RESET << std::endl;
                            last_reconnect_log = now;
                        }
                    }
                    gpu_res->event_handler->postEvent(EventType::RECONNECT);
                }
            }
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    
    if (gpu_res->gpu_id == 0) {
        std::cout << "[Fetcher] Block proposal fetcher stopped" << std::endl;
    }
}

bool verifySolution(
    const Pow& pow,
    const PowPuzzle& puzzle,
    const Digest& commitment,
    const Digest& target) {
    
    auto [index_a, index_b] = Pow::indices(commitment, pow.nonce);
    Digest leaf_a = Pow::compute_leaf_from_commitment_host(commitment, index_a, MERKLE_NUM_LEAFS);
    Digest leaf_b = Pow::compute_leaf_from_commitment_host(commitment, index_b, MERKLE_NUM_LEAFS);
    
    bool path_a_valid = Pow::verify_merkle_path_host(pow.root, index_a, pow.path_a, leaf_a);
    bool path_b_valid = Pow::verify_merkle_path_host(pow.root, index_b, pow.path_b, leaf_b);
    
    if (!path_a_valid || !path_b_valid) {
        LOG_DEBUG("Merkle path verification failed");
        return false;
    }
    
    PowMastPaths mast_paths = convertToPowMastPaths(puzzle.auth_paths);
    Digest final_hash = mast_paths.fast_mast_hash(pow);
    
    if (!digest_less_than_or_equal(final_hash, target)) {
        LOG_DEBUG("Hash does not meet target");
        return false;
    }
    
    return true;
}