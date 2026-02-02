#include "mining.cuh"
#include "rpc_client.cuh"
#include "connection_multiplexer.cuh"
#include "stratum_client.cuh"
#include "common.cuh"

// ============================================================================
// GpuWorker Implementation
// ============================================================================

GpuWorker::GpuWorker(int gpu_id, GpuResources* resources)
    : gpu_id(gpu_id)
    , gpu_resources(resources)
    , worker_handle(nullptr) {
    mining_mode = gpu_resources ? gpu_resources->mining_mode : MiningMode::Solo;
}

GpuWorker::~GpuWorker() {
    stop();
}

void GpuWorker::start() {
    if (!gpu_resources->event_handler) {
        gpu_resources->event_handler = std::make_unique<EventHandler>();
    }
    // GpuWorker doesn't manage clients directly - ConnectionMultiplexer does
    // Just register with the multiplexer
    ConnectionMultiplexer& mux = ConnectionMultiplexer::getInstance();
    worker_handle = mux.registerWorker(gpu_id, gpu_resources);
    
    if (!worker_handle) {
        std::cerr << "[GPU " << gpu_id << "] Failed to register with multiplexer" << std::endl;
        return;
    }
    
    // The multiplexer will handle connection and job fetching
    // GpuWorker just needs to wait for events
    worker_running = true;
    
    // Start mining thread
    mining_thread = std::thread(&GpuWorker::miningLoop, this);
}

// Legacy UnifiedMiningController code removed - using GpuWorker architecture now.

void GpuWorker::stop() {
    worker_running = false;
    gpu_resources->gpu_stop_flag = true;
    
    if (gpu_resources->event_handler) {
        gpu_resources->event_handler->shutdown();
    }

    if (mining_thread.joinable()) {
        mining_thread.join();
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
    
    // Calculate optimal batch size based on GPU capabilities
    // For high-end GPUs (RTX 5090, etc.), use larger batches for better performance
    // Target: ~900ms batch duration at ~16-20 MH/s = ~15-20M nonces per batch
    int num_sms = prop.multiProcessorCount;
    
    // Scale by SM count (RTX 5090 has ~256 SMs, older GPUs have fewer)
    // Increased batch sizes for better GPU utilization and reduced kernel launch overhead
    if (num_sms >= 200) {
        // High-end GPU (RTX 5090, A100, H100, etc.)
        gpu_resources->optimal_max_nonces = 50000000ULL; // 50M nonces (increased from 20M)
    } else if (num_sms >= 100) {
        // Mid-high end GPU (RTX 4090, A6000, etc.)
        gpu_resources->optimal_max_nonces = 30000000ULL; // 30M nonces (increased from 15M)
    } else if (num_sms >= 50) {
        // Mid-range GPU
        gpu_resources->optimal_max_nonces = 20000000ULL; // 20M nonces (increased from 10M)
    } else {
        // Lower-end GPU
        gpu_resources->optimal_max_nonces = 10000000ULL; // 10M nonces (increased from 5M)
    }
    
    return true;
}

void GpuWorker::miningLoop() {
    while (worker_running && !stop_mining && !gpu_resources->gpu_stop_flag) {
        // Wait for mining events from EventHandler
        MiningEvent event;
        bool has_event = gpu_resources->event_handler->waitForEvent(
            event, std::chrono::milliseconds(100));
        
        if (has_event) {
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
        // For stratum mode, always process job notifications even if same ID
        // (pool may resend same job, and we should restart mining)
        // For solo mode, skip duplicates to avoid unnecessary restarts
        bool is_duplicate = (puzzle.id == gpu_resources->current_proposal_id);
        if (is_duplicate && mining_mode == MiningMode::Solo) {
            return;  // Same puzzle in solo mode, skip
        }
        // In stratum mode, always update to restart mining (even if same ID)
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
    Digest commitment = mast_paths.commit();
    
    // Check if we can reuse existing buffer (XNT uses commitment only)
    bool need_preprocess = true;
    {
        std::lock_guard<std::mutex> lock(gpu_resources->state_mutex);
        if (gpu_resources->buffer && gpu_resources->buffer->is_valid()) {
            bool commitment_match = true;
            for (int i = 0; i < DIGEST_LEN; ++i) {
                if (gpu_resources->cached_commitment.values[i] != commitment.values[i]) {
                    commitment_match = false;
                    break;
                }
            }
            
            bool prev_block_match = true;
            if (puzzle.consensus_rule_set != CONSENSUS_XNT) {
                for (int i = 0; i < DIGEST_LEN; ++i) {
                    if (gpu_resources->cached_prev_block.values[i] != prev_block.values[i]) {
                        prev_block_match = false;
                        break;
                    }
                }
            }

            if (commitment_match && prev_block_match) {
                need_preprocess = false;
                gpu_resources->buffer->mast_paths = mast_paths;
                gpu_resources->buffer->prev_block_digest = prev_block;
                gpu_resources->cached_mast_paths = mast_paths;
                gpu_resources->cached_commitment = commitment;
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
            gpu_resources->cached_commitment = commitment;
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

// Legacy UnifiedMiningController methods removed.

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
    if (vram_gb < 6) {
        LOG_DEBUG("GPU " << device_id << " has insufficient VRAM (" << vram_gb << " GB)");
        return false;
    }
    
    gpu_res->gpu_name = prop.name;
    gpu_res->gpu_vram_total = prop.totalGlobalMem;
    gpu_res->gpu_uuid = get_gpu_uuid(device_id);
    
    // Calculate optimal batch size based on GPU capabilities
    int num_sms = prop.multiProcessorCount;
    if (num_sms >= 200) {
        // High-end GPU (RTX 5090, A100, H100, etc.)
        gpu_res->optimal_max_nonces = 50000000ULL; // 50M nonces (increased from 20M)
    } else if (num_sms >= 100) {
        // Mid-high end GPU (RTX 4090, A6000, etc.)
        gpu_res->optimal_max_nonces = 30000000ULL; // 30M nonces (increased from 15M)
    } else if (num_sms >= 50) {
        // Mid-range GPU
        gpu_res->optimal_max_nonces = 20000000ULL; // 20M nonces (increased from 10M)
    } else {
        // Lower-end GPU
        gpu_res->optimal_max_nonces = 10000000ULL; // 10M nonces (increased from 5M)
    }
    gpu_res->mining_mode = mining_mode;
    
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
    if (!mux.initialize(endpoint, g_miner_wallet_address, stratum_password)) {
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
    
    // Wait for the worker to finish (blocks until mining stops)
    while (!stop_mining && workers[index]->isRunning()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
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

// ============================================================================
// Continuous Mining Loop
// ============================================================================

bool continuousMiningLoop(GpuResources* gpu_res, GpuWorker* worker) {
    if (!gpu_res || !worker) return false;
    
    auto last_status_time = std::chrono::steady_clock::now();
    const auto STATUS_UPDATE_INTERVAL = std::chrono::seconds(5);
    
    while (!stop_mining && !gpu_res->gpu_stop_flag) {
        // Check for new events (new puzzle) - even when paused
        if (gpu_res->event_handler && gpu_res->event_handler->hasEvents()) {
            break; // Break to let handleNewPuzzle process the new job
        }
        
        if (gpu_res->gpu_pause_flag) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }
        
        if (!gpu_res->buffer || !gpu_res->buffer->is_valid()) {
            gpu_res->set_paused(true);
            continue;
        }
        
        std::string proposal_id;
        json template_for_this_batch;
        {
            std::lock_guard<std::mutex> lock(gpu_res->state_mutex);
            proposal_id = gpu_res->current_proposal_id;
            template_for_this_batch = gpu_res->current_template;  // Capture template with proposal_id
        }
        
        if (proposal_id.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }
        
        // Check if template is stale - if so, pause mining and wait for new job
        if (!template_for_this_batch.is_null() && !template_for_this_batch.empty()) {
            auto& multiplexer = ConnectionMultiplexer::getInstance();
            if (multiplexer.isTemplateStale(template_for_this_batch)) {
                // Template is stale, pause mining to save GPU power
                std::string prev_block = "unknown";
                if (template_for_this_batch.contains("metadata") && !template_for_this_batch["metadata"].is_null()) {
                    json metadata = template_for_this_batch["metadata"];
                    if (metadata.contains("prevBlock") && !metadata["prevBlock"].is_null()) {
                        prev_block = metadata.value("prevBlock", "");
                    } else if (metadata.contains("prev_block") && !metadata["prev_block"].is_null()) {
                        prev_block = metadata.value("prev_block", "");
                    }
                }
                std::string short_prev = prev_block.length() > 20 
                    ? prev_block.substr(0, 12) + "..." + prev_block.substr(prev_block.length() - 8) 
                    : prev_block;
                
                std::cout << "[GPU " << gpu_res->gpu_id << "] " << Color::YELLOW
                          << "Template stale (prev_block=" << short_prev 
                          << " != current tip), pausing mining to save power..." << Color::RESET << std::endl;
                
                gpu_res->set_paused(true);
                
                // Wait for new job or template update
                while (!stop_mining && !gpu_res->gpu_stop_flag) {
                    // Check if pause flag was cleared (new job received)
                    if (!gpu_res->gpu_pause_flag) {
                        break;
                    }
                    
                    // Check for new events (new puzzle/job)
                    if (gpu_res->event_handler && gpu_res->event_handler->hasEvents()) {
                        break;
                    }
                    
                    // Check if proposal ID changed (new template)
                    std::string current_proposal_id;
                    {
                        std::lock_guard<std::mutex> lock(gpu_res->state_mutex);
                        current_proposal_id = gpu_res->current_proposal_id;
                    }
                    if (current_proposal_id != proposal_id) {
                        break;
                    }
                    
                    // Check if template is no longer stale
                    json current_template;
                    {
                        std::lock_guard<std::mutex> lock(gpu_res->state_mutex);
                        current_template = gpu_res->current_template;
                    }
                    if (!current_template.is_null() && !current_template.empty()) {
                        if (!multiplexer.isTemplateStale(current_template)) {
                            break;
                        }
                    }
                    
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                }
                
                // Resume mining if we broke out of the wait loop
                if (!stop_mining && !gpu_res->gpu_stop_flag) {
                    gpu_res->set_paused(false);
                    continue; // Start over to get fresh proposal_id and template_obj
                }
                
                continue;
            }
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
            
            // Check if proposal has changed - if so, this solution is for a stale template
            std::string current_proposal_id;
            {
                std::lock_guard<std::mutex> lock(gpu_res->state_mutex);
                current_proposal_id = gpu_res->current_proposal_id;
            }
            
            if (current_proposal_id != proposal_id) {
                std::cout << "[GPU " << gpu_res->gpu_id << "] " << Color::YELLOW 
                          << "Solution discarded - template changed during mining" << Color::RESET << std::endl;
                // Don't count as rejected since we caught it ourselves
                continue;
            }
            
            const char* submit_target = gpu_res->is_stratum_mode() ? "pool" : "node";
            std::cout << "[GPU " << gpu_res->gpu_id << "] " << Color::YELLOW << Color::BOLD
                      << "*** SOLUTION FOUND! ***" << Color::RESET 
                      << " Submitting to " << submit_target << "..." << std::endl;
            
                PowMastPaths mast_paths = gpu_res->buffer->mast_paths;
                Digest solution_hash = mast_paths.fast_mast_hash(result.value().pow);
                
            // Use the template captured at the start of this mining batch
            // This ensures we submit with the same template we were mining for
            auto future = worker->submitSolution(
                    proposal_id,
                    result.value().pow,
                    solution_hash,
                    template_for_this_batch
                );
                
            // Wait for result (with timeout)
            if (future.wait_for(std::chrono::seconds(30)) == std::future_status::ready) {
                bool accepted = future.get();
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

// Legacy puzzleFetcher removed; ConnectionMultiplexer handles polling.

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
