#include "mining.cuh"
#include "rpc_client.cuh"
#include "connection_multiplexer.cuh"
#include "common.cuh"

// ============================================================================
// GpuWorker Implementation
// ============================================================================

GpuWorker::GpuWorker(int gpu_id, GpuResources* resources)
    : gpu_id(gpu_id)
    , gpu_resources(resources)
    , worker_handle(nullptr) {
}

GpuWorker::~GpuWorker() {
    stop();
}

void GpuWorker::start() {
    gpu_resources->event_handler = std::make_unique<EventHandler>();
    
    // Register with the connection multiplexer
    ConnectionMultiplexer& mux = ConnectionMultiplexer::getInstance();
    worker_handle = mux.registerWorker(gpu_id, gpu_resources);
    
    if (!worker_handle) {
        std::cerr << "[GPU " << gpu_id << "] Failed to register with multiplexer" << std::endl;
        return;
    }
    
    // Initialize CUDA for this GPU
    if (!initializeCuda()) {
        std::cerr << "[GPU " << gpu_id << "] Failed to initialize CUDA" << std::endl;
        return;
    }
    
    if (gpu_id == 0) {
        std::cout << Color::GREEN << "CUDA initialized successfully" << Color::RESET << std::endl;
        std::cout << "Mining loop started. Waiting for block proposals..." << std::endl;
    }
    
    worker_running = true;
    gpu_resources->gpu_node_connected = true;  // Multiplexer handles connection
    
    // Run the mining loop directly (no fetcher thread needed)
    miningLoop();
    
    // Unregister from multiplexer
    if (worker_handle) {
        mux.unregisterWorker(worker_handle);
        worker_handle = nullptr;
    }
}

void GpuWorker::stop() {
    worker_running = false;
    gpu_resources->gpu_stop_flag = true;
    
    if (gpu_resources->event_handler) {
        gpu_resources->event_handler->shutdown();
    }
}

bool GpuWorker::initializeCuda() {
    cudaError_t err = cudaSetDevice(gpu_id);
    if (err != cudaSuccess) {
        LOG_ERROR("cudaSetDevice", err);
        return false;
    }
    
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, gpu_id);
    if (err != cudaSuccess) {
        LOG_ERROR("cudaGetDeviceProperties", err);
        return false;
    }
    
    gpu_resources->gpu_name = prop.name;
    gpu_resources->gpu_vram_total = prop.totalGlobalMem;
    gpu_resources->gpu_uuid = get_gpu_uuid(gpu_id);
    gpu_resources->optimal_max_nonces = 1000000ULL;
    
    return true;
}

void GpuWorker::miningLoop() {
    while (worker_running && !stop_mining && !gpu_resources->gpu_stop_flag) {
        MiningEvent event;
        bool has_event = gpu_resources->event_handler->waitForEvent(
            event, std::chrono::milliseconds(1000));
        
        if (!has_event) {
            continue;
        }
        
        switch (event.type) {
            case EventType::NEW_PUZZLE:
                handleNewPuzzle(event);
                break;
                
            case EventType::STOP_MINING:
                worker_running = false;
                break;
                
            default:
                break;
        }
    }
}

void GpuWorker::handleNewPuzzle(const MiningEvent& event) {
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
        gpu_resources->current_template = template_obj;
        Digest original_target = hex_to_digest(puzzle.threshold);
        gpu_resources->current_real_target = original_target;
        if (g_test_mode) {
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
    
    // Check if we can reuse existing buffer
    bool need_preprocess = true;
    {
        std::lock_guard<std::mutex> lock(gpu_resources->state_mutex);
        if (gpu_resources->buffer && gpu_resources->buffer->is_valid()) {
            bool prev_block_match = true;
            for (int i = 0; i < DIGEST_LEN; ++i) {
                if (gpu_resources->cached_prev_block.values[i] != prev_block.values[i]) {
                    prev_block_match = false;
                    break;
                }
            }
            
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
            }
            
            if (prev_block_match && mast_paths_match) {
                need_preprocess = false;
                gpu_resources->buffer->mast_paths = mast_paths;
                gpu_resources->cached_mast_paths = mast_paths;
                std::cout << "[GPU " << gpu_id << "] " << Color::GREEN 
                          << "Reusing cached buffer" << Color::RESET << std::endl;
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
        
        {
            std::lock_guard<std::mutex> lock(gpu_resources->state_mutex);
            gpu_resources->cached_prev_block = prev_block;
            gpu_resources->cached_mast_paths = mast_paths;
        }
    }
    
    gpu_resources->update_job_received();
    gpu_resources->set_paused(false);
    
    std::string short_id = puzzle.id.length() > 20 ? puzzle.id.substr(0, 12) + "..." + puzzle.id.substr(puzzle.id.length() - 8) : puzzle.id;
    std::cout << "[GPU " << gpu_id << "] " << Color::CYAN << Color::BOLD 
              << "New job received" << Color::RESET 
              << " | Template ID: " << Color::CYAN << short_id << Color::RESET << std::endl;
    
    // Run the continuous mining loop
    continuousMiningLoop(gpu_resources, this);
}

std::future<bool> GpuWorker::submitSolution(
    const std::string& proposal_id,
    const Pow& pow_solution,
    const Digest& solution_hash,
    const json& template_obj) {
    
    ConnectionMultiplexer& mux = ConnectionMultiplexer::getInstance();
    return mux.submitSolution(gpu_id, proposal_id, pow_solution, solution_hash, template_obj);
}

// ============================================================================
// MultiGpuManager Implementation
// ============================================================================

MultiGpuManager::MultiGpuManager(
    const std::string& rpc_url,
    int specific_gpu)
    : rpc_url(rpc_url)
    , single_gpu_id(specific_gpu) {
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
    if (vram_gb < 6) {
        LOG_DEBUG("GPU " << device_id << " has insufficient VRAM (" << vram_gb << " GB)");
        return false;
    }
    
    gpu_res->gpu_name = prop.name;
    gpu_res->gpu_vram_total = prop.totalGlobalMem;
    gpu_res->gpu_uuid = get_gpu_uuid(device_id);
    gpu_res->optimal_max_nonces = 1000000ULL;
    
    // Create GpuWorker (uses shared connection through multiplexer)
    auto worker = std::make_unique<GpuWorker>(device_id, gpu_res.get());
    gpu_resources.push_back(std::move(gpu_res));
    workers.push_back(std::move(worker));
    
    std::cout << "Initialized GPU " << device_id << ": " << prop.name 
              << " (" << vram_gb << " GB)" << std::endl;
    
    return true;
}

void MultiGpuManager::startAll() {
    if (gpu_resources.empty()) {
        std::cerr << "No GPUs initialized" << std::endl;
        return;
    }
    
    // Initialize the connection multiplexer first
    ConnectionMultiplexer& mux = ConnectionMultiplexer::getInstance();
    if (!mux.initialize(rpc_url, g_miner_wallet_address)) {
        std::cerr << "Failed to initialize connection multiplexer" << std::endl;
        return;
    }
    
    // Start GPU worker threads
    for (size_t i = 0; i < workers.size(); ++i) {
        gpu_threads.emplace_back(&MultiGpuManager::gpuWorkerThread, this, i);
        if (i < workers.size() - 1) {
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
    
    // Stop workers first
    for (auto& worker : workers) {
        worker->stop();
    }
    
    // Wait for threads
    for (auto& thread : gpu_threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    
    // Shutdown multiplexer after workers are done
    ConnectionMultiplexer::destroyInstance();
    
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

void MultiGpuManager::gpuWorkerThread(size_t index) {
    if (index >= workers.size()) return;
    workers[index]->start();
}

void startUnifiedMining(
    const std::string& rpc_url,
    int specific_gpu) {
    
    MultiGpuManager manager(rpc_url, specific_gpu);
    
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

// ============================================================================
// Continuous Mining Loop
// ============================================================================

bool continuousMiningLoop(GpuResources* gpu_res, GpuWorker* worker) {
    if (!gpu_res || !worker) return false;
    
    auto last_status_time = std::chrono::steady_clock::now();
    const auto STATUS_UPDATE_INTERVAL = std::chrono::seconds(5);
    
    while (!stop_mining && !gpu_res->gpu_stop_flag) {
        if (gpu_res->gpu_pause_flag) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }
        
        // Check for new events (new puzzle)
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
        
        gpu_res->total_nonces_tested.fetch_add(gpu_res->optimal_max_nonces);
        
        if (duration_ms > 0) {
            double hashrate_ms = static_cast<double>(gpu_res->optimal_max_nonces) / duration_ms;
            gpu_res->hash_tracker.add(hashrate_ms);
        }
        
        // Periodic status update
        auto now = std::chrono::steady_clock::now();
        if (now - last_status_time >= STATUS_UPDATE_INTERVAL) {
            double avg_hashrate = gpu_res->hash_tracker.get_average();
            double hashrate_hps = avg_hashrate * 1000.0;
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
            std::cout << "[GPU " << gpu_res->gpu_id << "] " << Color::YELLOW << Color::BOLD
                      << "*** SOLUTION FOUND! ***" << Color::RESET 
                      << " Submitting to node..." << std::endl;
            
            PowMastPaths mast_paths = gpu_res->buffer->mast_paths;
            Digest solution_hash = mast_paths.fast_mast_hash(result.value());
            
            json template_obj;
            {
                std::lock_guard<std::mutex> lock(gpu_res->state_mutex);
                template_obj = gpu_res->current_template;
            }
            
            // Submit through multiplexer (async, get future)
            auto future = worker->submitSolution(
                proposal_id,
                result.value(),
                solution_hash,
                template_obj
            );
            
            // Wait for result (with timeout)
            if (future.wait_for(std::chrono::seconds(30)) == std::future_status::ready) {
                bool accepted = future.get();
                if (accepted) {
                    gpu_res->solutions_accepted++;
                    std::cout << "[GPU " << gpu_res->gpu_id << "] " << Color::GREEN << Color::BOLD 
                              << "*** BLOCK ACCEPTED! ***" << Color::RESET 
                              << " | Total blocks mined: " << Color::GREEN << gpu_res->solutions_accepted.load() << Color::RESET << std::endl;
                } else {
                    gpu_res->solutions_rejected++;
                }
            } else {
                gpu_res->solutions_rejected++;
                std::cout << "[GPU " << gpu_res->gpu_id << "] " << Color::RED 
                          << "Solution submission timed out" << Color::RESET << std::endl;
            }
        }
    }
    
    return true;
}

// ============================================================================
// Helper Functions
// ============================================================================

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

// ============================================================================
// Nonce Management
// ============================================================================

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

// ============================================================================
// Signal Handling
// ============================================================================

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
