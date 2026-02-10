#include "mining.cuh"
#include "rpc_client.cuh"
#include "connection_multiplexer.cuh"
#include "stratum_client.cuh"
#include "common.cuh"
#include "kernels.cuh"

// ========== Inlined from gpu_resources.cu (start) ==========
// ===== GLOBAL VARIABLES =====
std::vector<GpuResources*> g_all_gpu_resources;
std::mutex g_all_gpu_resources_mutex;


std::string get_gpu_uuid(int device_id) {
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device_id);
    
    if (err != cudaSuccess) {
        return "unknown";
    }
    
    // Convert UUID bytes to hex string
    std::ostringstream oss;
    oss << std::hex << std::setfill('0');
    
    for (int i = 0; i < 16; ++i) {
        oss << std::setw(2) << static_cast<int>(static_cast<unsigned char>(prop.uuid.bytes[i]));
        if (i == 3 || i == 5 || i == 7 || i == 9) {
            oss << "-";
        }
    }
    
    return oss.str();
}

void broadcast_stop_to_all_gpus(int source_gpu_id, const char* reason) {
    std::lock_guard<std::mutex> lock(g_all_gpu_resources_mutex);
    
    int stopped_count = 0;
    for (GpuResources* gpu : g_all_gpu_resources) {
        if (gpu) {
            try {
                bool was_stopped = gpu->gpu_stop_flag.load();
                if (!was_stopped) {
                    gpu->gpu_stop_flag = true;
                    stopped_count++;
                    if (gpu->event_handler) {
                        gpu->event_handler->postEvent(EventType::STOP_MINING);
                    }
                }
            } catch (...) {
                continue;
            }
        }
    }
    
    if (stopped_count > 0) {
        LOG_DEBUG("[GPU " << source_gpu_id << "] " << reason
                  << " - Broadcasting stop to " << stopped_count << " GPU(s)");
    }
}

GpuResources* find_gpu_resources(int gpu_id) {
    std::lock_guard<std::mutex> lock(g_all_gpu_resources_mutex);
    
    for (GpuResources* gpu : g_all_gpu_resources) {
        if (gpu && gpu->gpu_id == gpu_id) {
            return gpu;
        }
    }
    
    return nullptr;
}

double get_total_hashrate() {
    std::lock_guard<std::mutex> lock(g_all_gpu_resources_mutex);
    
    double total = 0.0;
    for (GpuResources* gpu : g_all_gpu_resources) {
        if (gpu && gpu->is_mining()) {
            total += gpu->hash_tracker.get_average();
        }
    }
    
    return total;
}

uint64_t get_total_solutions_found() {
    std::lock_guard<std::mutex> lock(g_all_gpu_resources_mutex);
    
    uint64_t total = 0;
    for (GpuResources* gpu : g_all_gpu_resources) {
        if (gpu) {
            total += gpu->solutions_found.load();
        }
    }
    
    return total;
}

void cleanup_gpu_memory() {
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    
    if (error != cudaSuccess || device_count == 0) {
        return;
    }
    
    for (int gpu_id = 0; gpu_id < device_count; ++gpu_id) {
        cudaSetDevice(gpu_id);
        cudaDeviceSynchronize();
        cudaGetLastError();
        cudaDeviceReset();
    }
    
    {
        std::lock_guard<std::mutex> lock(g_all_gpu_resources_mutex);
        g_all_gpu_resources.clear();
    }
}
// ========== Inlined from gpu_resources.cu (end) ==========

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
    
    // Batch size: use global override if set, otherwise auto-detect for this GPU
    gpu_resources->optimal_max_nonces = (g_batch_size > 0) ? g_batch_size : get_optimal_batch_size(gpu_id);
    
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
    
    Digest prev_block = hex_to_digest(puzzle.prev_block);
    PowMastPaths mast_paths = convertToPowMastPaths(puzzle.auth_paths);
    Digest commitment = mast_paths.commit();
    Digest original_target = hex_to_digest(puzzle.threshold);
    Digest effective_target = g_test_mode ? make_target_easier(original_target, 100000) : original_target;
    
    // Check for duplicate (solo mode only)
    {
        std::lock_guard<std::mutex> lock(gpu_resources->state_mutex);
        bool is_duplicate = (puzzle.id == gpu_resources->current_proposal_id);
        if (is_duplicate && mining_mode == MiningMode::Solo) {
            return;
        }
    }
    
    // Check if we can reuse existing buffer (XNT uses commitment only)
    bool need_preprocess = true;
    bool have_existing_buffer = false;
    {
        std::lock_guard<std::mutex> lock(gpu_resources->state_mutex);
        if (gpu_resources->buffer && gpu_resources->buffer->is_valid()) {
            have_existing_buffer = true;
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
        if (have_existing_buffer) {
            // ---- P2.7: Async preprocessing ----
            // We have a valid old buffer — launch preprocessing in background and
            // continue mining the OLD puzzle. The mining loop will swap buffers when done.
            
            // Wait for any prior async preprocessing to finish first
            if (gpu_resources->async_preprocess_thread.joinable()) {
                gpu_resources->async_preprocess_thread.join();
            }
            gpu_resources->async_preprocess_done = false;
            gpu_resources->async_preprocess_running = true;
            
            // Store new puzzle metadata (will be applied on buffer swap)
            {
                std::lock_guard<std::mutex> lock(gpu_resources->state_mutex);
                gpu_resources->pending_proposal_id = puzzle.id;
                gpu_resources->pending_template = template_obj;
                gpu_resources->pending_target = effective_target;
                gpu_resources->pending_real_target = original_target;
                gpu_resources->pending_prev_block = prev_block;
                gpu_resources->pending_mast_paths = mast_paths;
                gpu_resources->pending_commitment = commitment;
            }
            
            // Launch background preprocessing thread
            int bg_gpu_id = gpu_id;
            int bg_consensus = puzzle.consensus_rule_set;
            gpu_resources->async_preprocess_thread = std::thread(
                [this, bg_gpu_id, mast_paths, prev_block, bg_consensus]() {
                    cudaError_t err = cudaSetDevice(bg_gpu_id);
                    if (err != cudaSuccess) {
                        LOG_ERROR("cudaSetDevice in async preprocess", err);
                        gpu_resources->async_preprocess_running = false;
                        return;
                    }
                    
                    auto start = std::chrono::steady_clock::now();
                    auto new_buffer = Pow::preprocess(mast_paths, prev_block, bg_consensus, nullptr);
                    auto end = std::chrono::steady_clock::now();
                    double elapsed_s = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0;
                    
                    if (new_buffer.is_valid()) {
                        gpu_resources->pending_buffer = std::make_unique<GuesserBuffer>(std::move(new_buffer));
                        gpu_resources->async_preprocess_done = true;
                        std::cout << "[GPU " << bg_gpu_id << "] " << Color::GREEN 
                                  << "Async preprocessing complete" << Color::RESET 
                                  << " (" << std::fixed << std::setprecision(2) << elapsed_s << "s, mining continued)" << std::endl;
                    } else {
                        std::cout << "[GPU " << bg_gpu_id << "] " << Color::RED 
                                  << "Async preprocessing failed" << Color::RESET << std::endl;
                    }
                    gpu_resources->async_preprocess_running = false;
                }
            );
            
            std::cout << "[GPU " << gpu_id << "] " << Color::YELLOW 
                      << "Async preprocessing started (mining continues with old buffer)" << Color::RESET << std::endl;
            
            // DON'T update current_* metadata yet — keep mining old puzzle.
            // The mining loop will apply pending metadata on buffer swap.
            // But we DO need to resume the mining loop, so fall through to it.
        } else {
            // No existing buffer (first puzzle) — must block on preprocessing
            {
                std::lock_guard<std::mutex> lock(gpu_resources->state_mutex);
                gpu_resources->current_proposal_id = puzzle.id;
                gpu_resources->current_template = template_obj;
                gpu_resources->current_target = effective_target;
                gpu_resources->current_real_target = original_target;
            }
            
            if (g_test_mode) {
                std::cout << "[GPU " << gpu_id << "] " << Color::YELLOW 
                          << "TEST MODE: Target made 100000x easier" << Color::RESET << std::endl;
            }
            
            resetNonceCounter(gpu_resources, puzzle.id);
            
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
    } else {
        // Buffer reused — update current metadata to new puzzle
        {
            std::lock_guard<std::mutex> lock(gpu_resources->state_mutex);
            gpu_resources->current_proposal_id = puzzle.id;
            gpu_resources->current_template = template_obj;
            gpu_resources->current_target = effective_target;
            gpu_resources->current_real_target = original_target;
        }
        resetNonceCounter(gpu_resources, puzzle.id);
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
    
    // Batch size: use global override if set, otherwise auto-detect for this GPU
    gpu_res->optimal_max_nonces = (g_batch_size > 0) ? g_batch_size : get_optimal_batch_size(device_id);
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
        // Join async preprocessing thread before cleaning up buffers
        if (res->async_preprocess_thread.joinable()) {
            res->async_preprocess_thread.join();
        }
        if (res->pending_buffer) {
            res->pending_buffer->cleanup();
            res->pending_buffer.reset();
        }
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
// Continuous Mining Loop - Double-Buffered Async Version
// ============================================================================
// TRUE PIPELINING: We use 2 independent CUDA streams. Each stream has its own
// output buffers. The pipeline works as follows:
//
// Time:    |----Batch 0----|----Batch 2----|----Batch 4----|
// Stream A: [==KERNEL 0===][==KERNEL 2===][==KERNEL 4===]
// Stream B:      [==KERNEL 1===][==KERNEL 3===][==KERNEL 5===]
// CPU:      [Launch0][Chk-1][Launch2][Chk0][Launch4][Chk2]...
//
// Key: We NEVER block. After launching batch N, we check if batch N-2 (same
// stream, previous kernel) is done. If not done, we continue - the kernel
// will complete eventually and we'll catch it next time around.

bool continuousMiningLoop(GpuResources* gpu_res, GpuWorker* worker) {
    if (!gpu_res || !worker) return false;
    
    auto last_status_time = std::chrono::steady_clock::now();
    const auto STATUS_UPDATE_INTERVAL = std::chrono::seconds(5);
    
    // Initialize double-buffered miner
    if (!gpu_res->async_miner) {
        gpu_res->async_miner = std::make_unique<DoubleBufferedMiner>();
    }
    
    DoubleBufferedMiner& miner = *gpu_res->async_miner;
    if (!miner.initialize()) {
        std::cerr << "[GPU " << gpu_res->gpu_id << "] Failed to initialize async miner, falling back to sync" << std::endl;
        return continuousMiningLoopSync(gpu_res, worker);
    }
    
    // Track batch info for each slot
    struct BatchInfo {
        std::string proposal_id;
        json template_obj;
        uint64_t start_nonce;
        uint64_t batch_size;
        std::chrono::high_resolution_clock::time_point start_time;
        bool pending_result;  // True if kernel launched and result not yet processed
    };
    BatchInfo batch_info[2] = {};
    
    // Pipeline state
    int active_slot = 0;  // Which slot to use for NEXT launch
    int batches_in_flight = 0;
    
    while (!stop_mining && !gpu_res->gpu_stop_flag) {
        // Check for new events
        if (gpu_res->event_handler && gpu_res->event_handler->hasEvents()) {
            // Drain pipeline before breaking
            for (int i = 0; i < 2; i++) {
                if (batch_info[i].pending_result) {
                    cudaStreamSynchronize(miner.slots[i].stream);
                }
            }
            break;
        }
        
        // ---- P2.7: Check if async preprocessing completed → swap buffers ----
        if (gpu_res->async_preprocess_done.load(std::memory_order_acquire)) {
            // Drain in-flight mining batches before swapping (they reference old buffer)
            for (int i = 0; i < 2; i++) {
                if (batch_info[i].pending_result) {
                    cudaStreamSynchronize(miner.slots[i].stream);
                    // Count nonces from the batch we're draining
                    gpu_res->total_nonces_tested.fetch_add(batch_info[i].batch_size);
                    batch_info[i].pending_result = false;
                    miner.slots[i].kernel_launched = false;
                }
            }
            batches_in_flight = 0;
            
            // Swap buffer and apply pending puzzle metadata
            gpu_res->buffer = std::move(gpu_res->pending_buffer);
            {
                std::lock_guard<std::mutex> lock(gpu_res->state_mutex);
                gpu_res->current_proposal_id = gpu_res->pending_proposal_id;
                gpu_res->current_template = gpu_res->pending_template;
                gpu_res->current_target = gpu_res->pending_target;
                gpu_res->current_real_target = gpu_res->pending_real_target;
                gpu_res->cached_prev_block = gpu_res->pending_prev_block;
                gpu_res->cached_mast_paths = gpu_res->pending_mast_paths;
                gpu_res->cached_commitment = gpu_res->pending_commitment;
            }
            
            // Reset nonce counter for new puzzle and force re-init of GPU constants
            resetNonceCounter(gpu_res, gpu_res->pending_proposal_id);
            gpu_res->buffer->gpu_range_initialized = false;
            
            gpu_res->async_preprocess_done.store(false, std::memory_order_release);
            
            std::cout << "[GPU " << gpu_res->gpu_id << "] " << Color::GREEN << Color::BOLD
                      << "Buffer swapped — now mining new template" << Color::RESET << std::endl;
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
            template_for_this_batch = gpu_res->current_template;
        }
        
        if (proposal_id.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }
        
        // Check template staleness
        if (!template_for_this_batch.is_null() && !template_for_this_batch.empty()) {
            auto& multiplexer = ConnectionMultiplexer::getInstance();
            if (multiplexer.isTemplateStale(template_for_this_batch)) {
                // Wait for in-flight batches then pause
                for (int i = 0; i < 2; i++) {
                    if (batch_info[i].pending_result) {
                        cudaStreamSynchronize(miner.slots[i].stream);
                        batch_info[i].pending_result = false;
                    }
                }
                batches_in_flight = 0;
                
                std::cout << "[GPU " << gpu_res->gpu_id << "] " << Color::YELLOW
                          << "Template stale, pausing..." << Color::RESET << std::endl;
                gpu_res->set_paused(true);
                
                while (!stop_mining && !gpu_res->gpu_stop_flag && gpu_res->gpu_pause_flag) {
                    if (gpu_res->event_handler && gpu_res->event_handler->hasEvents()) break;
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                }
                
                if (!stop_mining && !gpu_res->gpu_stop_flag) {
                    gpu_res->set_paused(false);
                }
                continue;
            }
        }
        
        // =====================================================================
        // STEP 1: Check if CURRENT slot's previous batch is complete (non-blocking)
        // =====================================================================
        AsyncMiningSlot& current_slot = miner.slots[active_slot];
        BatchInfo& current_batch_info = batch_info[active_slot];
        
        if (current_batch_info.pending_result) {
            // Check if this slot's kernel is done (NON-BLOCKING)
            cudaError_t status = cudaEventQuery(current_slot.completion_event);
            
            if (status == cudaSuccess) {
                // Kernel completed! Process result
                auto now = std::chrono::high_resolution_clock::now();
                auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    now - current_batch_info.start_time).count();
                
                gpu_res->total_nonces_tested.fetch_add(current_batch_info.batch_size);
                batches_in_flight--;
                
                if (duration_ms > 0) {
                    double hashrate_ms = static_cast<double>(current_batch_info.batch_size) / duration_ms;
                    gpu_res->hash_tracker.add(hashrate_ms);
                }
                
                // Check for solution (pinned memory was async copied)
                int solution_found = *current_slot.h_solution_found_pinned;
                
                if (solution_found) {
                    gpu_res->solutions_found++;
                    
                    // Retrieve solution data
                    MiningSolution solution;
                    cudaMemcpy(&solution.pow.nonce, current_slot.d_solution_nonce_digest, 
                               sizeof(Digest), cudaMemcpyDeviceToHost);
                    cudaMemcpy(solution.pow.path_a, current_slot.d_solution_path_a,
                               MERKLE_TREE_HEIGHT_ * sizeof(Digest), cudaMemcpyDeviceToHost);
                    cudaMemcpy(solution.pow.path_b, current_slot.d_solution_path_b,
                               MERKLE_TREE_HEIGHT_ * sizeof(Digest), cudaMemcpyDeviceToHost);
                    cudaMemcpy(&solution.kernel_final_hash, current_slot.d_solution_final_hash,
                               sizeof(Digest), cudaMemcpyDeviceToHost);
                    solution.pow.root = gpu_res->buffer->merkle_root;
                    
                    // Check if proposal still current
                    std::string current_proposal_id;
                    {
                        std::lock_guard<std::mutex> lock(gpu_res->state_mutex);
                        current_proposal_id = gpu_res->current_proposal_id;
                    }
                    
                    if (current_proposal_id != current_batch_info.proposal_id) {
                        std::cout << "[GPU " << gpu_res->gpu_id << "] " << Color::YELLOW 
                                  << "Solution discarded - template changed" << Color::RESET << std::endl;
                    } else {
                        const char* target = gpu_res->is_stratum_mode() ? "pool" : "node";
                        std::cout << "[GPU " << gpu_res->gpu_id << "] " << Color::YELLOW << Color::BOLD
                                  << "*** SOLUTION FOUND! ***" << Color::RESET 
                                  << " Submitting to " << target << "..." << std::endl;
                        
                        PowMastPaths mast_paths = gpu_res->buffer->mast_paths;
                        Digest solution_hash = mast_paths.fast_mast_hash(solution.pow);
                        
                        // Submit asynchronously
                        auto future = worker->submitSolution(
                            current_batch_info.proposal_id,
                            solution.pow,
                            solution_hash,
                            current_batch_info.template_obj);
                        
                        // Non-blocking check with short timeout
                        if (future.wait_for(std::chrono::seconds(30)) == std::future_status::ready) {
                            bool accepted = future.get();
                            if (accepted) {
                                gpu_res->solutions_accepted++;
                                const char* msg = gpu_res->is_stratum_mode() ? "SHARE ACCEPTED" : "BLOCK ACCEPTED";
                                std::cout << "[GPU " << gpu_res->gpu_id << "] " << Color::GREEN << Color::BOLD 
                                          << "*** " << msg << "! ***" << Color::RESET << std::endl;
                            } else {
                                gpu_res->solutions_rejected++;
                            }
                        } else {
                            gpu_res->solutions_rejected++;
                        }
                    }
                }
                
                current_batch_info.pending_result = false;
                current_slot.kernel_launched = false;
                
            } else if (status == cudaErrorNotReady) {
                // Kernel still running - switch to other slot and continue
                // DON'T BLOCK! Just use the other slot
                active_slot = 1 - active_slot;
                
                // If other slot also busy, we need to wait for one
                if (batch_info[active_slot].pending_result) {
                    cudaError_t other_status = cudaEventQuery(miner.slots[active_slot].completion_event);
                    if (other_status == cudaErrorNotReady) {
                        // Both slots busy - wait for current one (this is expected with 2 slots)
                        cudaStreamSynchronize(current_slot.stream);
                        // Will process on next iteration
                        continue;
                    }
                }
                continue;
            }
        }
        
        // =====================================================================
        // STEP 2: Launch new batch on current slot (non-blocking)
        // =====================================================================
        Digest target;
        {
            std::lock_guard<std::mutex> lock(gpu_res->state_mutex);
            target = gpu_res->current_target;
        }
        
        uint64_t start_nonce = getNextNonceRange(gpu_res, gpu_res->optimal_max_nonces);
        
        // Reset slot for new batch
        *current_slot.h_solution_found_pinned = 0;
        cudaMemsetAsync(current_slot.d_solution_found, 0, sizeof(int), current_slot.stream);
        
        // Get launch config
        int threads_per_block, blocks_per_grid;
        int gpu_id;
        cudaGetDevice(&gpu_id);
        calculate_mining_launch_config(gpu_res->optimal_max_nonces, threads_per_block, blocks_per_grid, gpu_id);
        
        // Ensure GPU range initialized
        if (!gpu_res->buffer->gpu_range_initialized) {
            int actual_gpu_count = g_total_gpu_count.load();
            GpuNonceRange gpu_range = calculate_gpu_range(gpu_id, actual_gpu_count);
            cudaMemcpyToSymbol(d_gpu_range_start, &gpu_range.range_start, sizeof(uint64_t));
            cudaMemcpyToSymbol(d_gpu_range_size, &gpu_range.range_size, sizeof(uint64_t));
            initialize_top_tree_cache(gpu_res->buffer->d_merkle_tree, gpu_res->buffer->num_leafs);
            gpu_res->buffer->gpu_range_initialized = true;
        }
        
        // Launch kernel
        MiningKernelType kernel_type = select_mining_kernel(gpu_id);
        
        if (kernel_type == MiningKernelType::HIGH_VRAM) {
            parallel_mining_kernel_high_vram<<<blocks_per_grid, threads_per_block, 0, current_slot.stream>>>(
                gpu_res->buffer->d_leafs,
                gpu_res->buffer->d_merkle_tree,
                gpu_res->buffer->index_picker_preimage,
                target,
                start_nonce,
                gpu_res->optimal_max_nonces,
                gpu_res->buffer->num_leafs,
                MERKLE_TREE_HEIGHT_,
                gpu_res->buffer->mast_paths,
                gpu_res->buffer->hash,
                gpu_res->buffer->consensus_rule_set,
                current_slot.d_solution_nonce,
                current_slot.d_solution_found,
                current_slot.d_solution_path_a,
                current_slot.d_solution_path_b,
                current_slot.d_solution_nonce_digest,
                current_slot.d_solution_final_hash);
        } else {
            parallel_mining_kernel_low_vram<<<blocks_per_grid, threads_per_block, 0, current_slot.stream>>>(
                nullptr,
                gpu_res->buffer->d_merkle_tree,
                gpu_res->buffer->index_picker_preimage,
                target,
                start_nonce,
                gpu_res->optimal_max_nonces,
                gpu_res->buffer->num_leafs,
                MERKLE_TREE_HEIGHT_,
                gpu_res->buffer->tree_size,
                gpu_res->buffer->mast_paths,
                gpu_res->buffer->hash,
                gpu_res->buffer->consensus_rule_set,
                current_slot.d_solution_nonce,
                current_slot.d_solution_found,
                current_slot.d_solution_path_a,
                current_slot.d_solution_path_b,
                current_slot.d_solution_nonce_digest,
                current_slot.d_solution_final_hash);
        }
        
        // Queue async copy of solution flag to pinned memory
        cudaMemcpyAsync(current_slot.h_solution_found_pinned, current_slot.d_solution_found,
                        sizeof(int), cudaMemcpyDeviceToHost, current_slot.stream);
        
        // Record completion event
        cudaEventRecord(current_slot.completion_event, current_slot.stream);
        
        // Store batch info
        current_batch_info.proposal_id = proposal_id;
        current_batch_info.template_obj = template_for_this_batch;
        current_batch_info.start_nonce = start_nonce;
        current_batch_info.batch_size = gpu_res->optimal_max_nonces;
        current_batch_info.start_time = std::chrono::high_resolution_clock::now();
        current_batch_info.pending_result = true;
        current_slot.kernel_launched = true;
        batches_in_flight++;
        
        // Switch to other slot for next iteration
        active_slot = 1 - active_slot;
        
        // =====================================================================
        // STEP 3: Status update (non-blocking)
        // =====================================================================
        auto now = std::chrono::steady_clock::now();
        if (now - last_status_time >= STATUS_UPDATE_INTERVAL) {
            double avg_hashrate = gpu_res->hash_tracker.get_average();
            double hashrate_hps = avg_hashrate * 1000.0;
            std::string hashrate_str = format_hashrate(hashrate_hps);
            
            std::cout << "[GPU " << gpu_res->gpu_id << "] " << Color::GREEN 
                      << "Mining (async x2)..." << Color::RESET 
                      << " | Rate: " << Color::YELLOW << hashrate_str << Color::RESET
                      << " | Nonces: " << gpu_res->total_nonces_tested.load()
                      << " | " << Color::GREEN << gpu_res->solutions_accepted.load() << Color::RESET 
                      << "/" << Color::RED << gpu_res->solutions_rejected.load() << Color::RESET 
                      << std::endl;
            
            last_status_time = now;
        }
    }
    
    // Cleanup: drain pipeline
    for (int i = 0; i < 2; i++) {
        if (batch_info[i].pending_result) {
            cudaStreamSynchronize(miner.slots[i].stream);
        }
    }
    
    return true;
}

// ============================================================================
// Synchronous Mining Loop (Fallback)
// ============================================================================
// Original synchronous implementation used as fallback if async init fails

bool continuousMiningLoopSync(GpuResources* gpu_res, GpuWorker* worker) {
    if (!gpu_res || !worker) return false;
    
    auto last_status_time = std::chrono::steady_clock::now();
    const auto STATUS_UPDATE_INTERVAL = std::chrono::seconds(5);
    
    while (!stop_mining && !gpu_res->gpu_stop_flag) {
        if (gpu_res->event_handler && gpu_res->event_handler->hasEvents()) {
            break;
        }
        
        // ---- P2.7: Check if async preprocessing completed → swap buffers (sync loop) ----
        if (gpu_res->async_preprocess_done.load(std::memory_order_acquire)) {
            gpu_res->buffer = std::move(gpu_res->pending_buffer);
            {
                std::lock_guard<std::mutex> lock(gpu_res->state_mutex);
                gpu_res->current_proposal_id = gpu_res->pending_proposal_id;
                gpu_res->current_template = gpu_res->pending_template;
                gpu_res->current_target = gpu_res->pending_target;
                gpu_res->current_real_target = gpu_res->pending_real_target;
                gpu_res->cached_prev_block = gpu_res->pending_prev_block;
                gpu_res->cached_mast_paths = gpu_res->pending_mast_paths;
                gpu_res->cached_commitment = gpu_res->pending_commitment;
            }
            resetNonceCounter(gpu_res, gpu_res->pending_proposal_id);
            gpu_res->buffer->gpu_range_initialized = false;
            gpu_res->async_preprocess_done.store(false, std::memory_order_release);
            
            std::cout << "[GPU " << gpu_res->gpu_id << "] " << Color::GREEN << Color::BOLD
                      << "Buffer swapped — now mining new template" << Color::RESET << std::endl;
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
            template_for_this_batch = gpu_res->current_template;
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
        
        auto now = std::chrono::steady_clock::now();
        if (now - last_status_time >= STATUS_UPDATE_INTERVAL) {
            double avg_hashrate = gpu_res->hash_tracker.get_average();
            double hashrate_hps = avg_hashrate * 1000.0;
            std::string hashrate_str = format_hashrate(hashrate_hps);
            
            uint64_t total_nonces = gpu_res->total_nonces_tested.load();
            uint64_t accepted = gpu_res->solutions_accepted.load();
            uint64_t rejected = gpu_res->solutions_rejected.load();
            
            std::cout << "[GPU " << gpu_res->gpu_id << "] " << Color::GREEN 
                      << "Mining (sync)..." << Color::RESET 
                      << " | Hash Rate: " << Color::YELLOW << hashrate_str << Color::RESET
                      << " | Nonces: " << total_nonces
                      << " | " << Color::GREEN << accepted << Color::RESET 
                      << " / " << Color::RED << rejected << Color::RESET 
                      << " (success / reject)" << std::endl;
            
            last_status_time = now;
        }
        
        if (result.has_value()) {
            gpu_res->solutions_found++;
            
            std::string current_proposal_id;
            {
                std::lock_guard<std::mutex> lock(gpu_res->state_mutex);
                current_proposal_id = gpu_res->current_proposal_id;
            }
            
            if (current_proposal_id != proposal_id) {
                std::cout << "[GPU " << gpu_res->gpu_id << "] " << Color::YELLOW 
                          << "Solution discarded - template changed" << Color::RESET << std::endl;
                continue;
            }
            
            const char* submit_target = gpu_res->is_stratum_mode() ? "pool" : "node";
            std::cout << "[GPU " << gpu_res->gpu_id << "] " << Color::YELLOW << Color::BOLD
                      << "*** SOLUTION FOUND! ***" << Color::RESET 
                      << " Submitting to " << submit_target << "..." << std::endl;
            
            PowMastPaths mast_paths = gpu_res->buffer->mast_paths;
            Digest solution_hash = mast_paths.fast_mast_hash(result.value().pow);
            
            auto future = worker->submitSolution(
                proposal_id,
                result.value().pow,
                solution_hash,
                template_for_this_batch);
            
            if (future.wait_for(std::chrono::seconds(30)) == std::future_status::ready) {
                bool accepted = future.get();
                if (accepted) {
                    gpu_res->solutions_accepted++;
                    const char* msg = gpu_res->is_stratum_mode() ? "SHARE ACCEPTED" : "BLOCK ACCEPTED";
                    std::cout << "[GPU " << gpu_res->gpu_id << "] " << Color::GREEN << Color::BOLD 
                              << "*** " << msg << "! ***" << Color::RESET 
                              << " | Total: " << Color::GREEN << gpu_res->solutions_accepted.load() 
                              << Color::RESET << std::endl;
                } else {
                    gpu_res->solutions_rejected++;
                }
            } else {
                gpu_res->solutions_rejected++;
                std::cout << "[GPU " << gpu_res->gpu_id << "] " << Color::RED 
                          << "Submission timed out" << Color::RESET << std::endl;
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
