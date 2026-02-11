#include "mining.cuh"
#include "connection_multiplexer.cuh"
#include "mining_client.h"
#include "network.cuh"
#include "rpc_client.cuh"
#include <fstream>
#include <iomanip>
#include <chrono>
#include <cstdlib>

void print_usage(const char* program_name) {
    std::cerr << "\n" << Color::BOLD << "Usage:" << Color::RESET << std::endl;
    std::cerr << "  " << program_name << " [OPTIONS] -w, --wallet ADDRESS\n" << std::endl;
    
    std::cerr << Color::BOLD << "Required:" << Color::RESET << std::endl;
    std::cerr << "  -w, --wallet ADDRESS  Your Neptune wallet address\n" << std::endl;
    
    std::cerr << Color::BOLD << "Solo Mining (default):" << Color::RESET << std::endl;
    std::cerr << "  --rpc-url URL         RPC endpoint URL (default: http://127.0.0.1:9897)" << std::endl;
    std::cerr << "  -H, --host HOST       RPC host" << std::endl;
    std::cerr << "  -p, --port PORT       RPC port\n" << std::endl;
    
    std::cerr << Color::BOLD << "Pool Mining (Stratum):" << Color::RESET << std::endl;
    std::cerr << "  --stratum URL         Stratum pool URL (e.g., stratum://pool.example.com:3333)" << std::endl;
    std::cerr << "  --stratum-pass PASS   Stratum password (default: x)" << std::endl;
    std::cerr << "  --stratum-worker NAME Stratum worker name (default: xnt-miner)\n" << std::endl;
    
    std::cerr << Color::BOLD << "General Options:" << Color::RESET << std::endl;
    std::cerr << "  -d, --device ID       Use specific GPU device ID (default: all GPUs)" << std::endl;
    std::cerr << "  --rpc-url URL         RPC endpoint URL (default: http://127.0.0.1:9897)" << std::endl;
    std::cerr << "  --test-mode           Enable test mode (100,000x easier target)" << std::endl;
    std::cerr << "  --benchmark           Run mining benchmark (saves/loads test template)" << std::endl;
    std::cerr << "  --fetch-interval SEC  Job fetch interval in seconds (default: 5)" << std::endl;
    std::cerr << "  --batch N             Nonces per kernel (0=auto, e.g. 20000000 for 20M)" << std::endl;
    std::cerr << "                        Or set XNT_BATCH_SIZE env for quick tuning" << std::endl;
    std::cerr << "  -h, --help            Show this help message\n" << std::endl;
    
    std::cerr << Color::BOLD << "Examples:" << Color::RESET << std::endl;
    std::cerr << "  " << Color::DIM << "# Solo mining to local node" << Color::RESET << std::endl;
    std::cerr << "  " << program_name << " -w nolgam..." << std::endl;
    std::cerr << std::endl;
    std::cerr << "  " << Color::DIM << "# Solo mining to remote node" << Color::RESET << std::endl;
    std::cerr << "  " << program_name << " -w nolgam... --rpc-url http://192.168.1.100:9897" << std::endl;
    std::cerr << std::endl;
    std::cerr << "  " << Color::DIM << "# Pool mining via stratum" << Color::RESET << std::endl;
    std::cerr << "  " << program_name << " -w nolgam... --stratum stratum://pool.example.com:3333" << std::endl;
    std::cerr << std::endl;
    std::cerr << "  " << Color::DIM << "# Benchmark mining speed" << Color::RESET << std::endl;
    std::cerr << "  " << program_name << " -w nolgam... --benchmark" << std::endl;
    std::cerr << std::endl;
}

void print_system_info() {
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    
    if (error != cudaSuccess) {
        std::cerr << Color::RED << "CUDA Error: " << cudaGetErrorString(error) << Color::RESET << std::endl;
        return;
    }
    
    std::cout << Color::BOLD << "System Information:" << Color::RESET << std::endl;
    std::cout << "  CUDA Devices: " << device_count << std::endl;
    
    for (int i = 0; i < device_count; ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        
        size_t vram_gb = prop.totalGlobalMem / (1024ULL * 1024ULL * 1024ULL);
        
        std::cout << "  GPU " << i << ": " << prop.name 
                  << " (" << vram_gb << " GB)" << std::endl;
    }
    std::cout << std::endl;
}

// ============================================================================
// Benchmark Functions
// ============================================================================
// Benchmark mode allows testing mining performance without requiring an active
// RPC connection. It saves a block template to disk and reuses it for
// consistent performance measurements.

const std::string BENCHMARK_FILE = "benchmark_template.json";

/// Save a block template to disk for offline benchmark testing.
/// 
/// @param template_response JSON response containing the block template
/// @return true if template was successfully saved, false otherwise
bool saveBenchmarkTemplate(const json& template_response) {
    std::ofstream file(BENCHMARK_FILE);
    if (!file.is_open()) {
        std::cerr << Color::RED << "Error: Cannot create benchmark file: " << BENCHMARK_FILE << Color::RESET << std::endl;
        return false;
    }
    // Write formatted JSON (indented for readability)
    file << std::setw(2) << template_response << std::endl;
    file.close();
    std::cout << Color::GREEN << "Benchmark template saved to " << BENCHMARK_FILE << Color::RESET << std::endl;
    return true;
}

/// Load a previously saved block template from disk.
/// 
/// @param template_response Output parameter to receive the loaded template
/// @return true if template was successfully loaded, false if file doesn't exist or is invalid
bool loadBenchmarkTemplate(json& template_response) {
    std::ifstream file(BENCHMARK_FILE);
    if (!file.is_open()) {
        return false;
    }
    try {
        file >> template_response;
        file.close();
        return true;
    } catch (const std::exception& e) {
        std::cerr << Color::RED << "Error: Invalid benchmark file: " << e.what() << Color::RESET << std::endl;
        return false;
    }
}

/// Run mining benchmark to measure hash rate and validate solutions.
/// 
/// Benchmark mode operates in two phases:
/// 1. Template acquisition: Fetches a block template from the node (if not cached)
/// 2. Mining loop: Mines using an easier target (100000x) to find solutions quickly
///    and validates them using the same logic as the Rust node.
/// 
/// @param endpoint RPC endpoint URL (used only if template needs to be fetched)
/// @param gpu_id GPU device ID to use for mining (-1 for auto-select)
void runBenchmark(const std::string& endpoint, int gpu_id) {
    std::cout << Color::BOLD << "\n=== Mining Benchmark ===" << Color::RESET << std::endl;
    
    json template_response;
    bool template_exists = loadBenchmarkTemplate(template_response);
    
    if (!template_exists) {
        std::cout << "Benchmark template not found. Fetching from node..." << std::endl;
        
        // Wallet address is required only when fetching a new template from the node.
        // Once the template is saved, the benchmark can run offline.
        if (g_miner_wallet_address.empty()) {
            std::cerr << Color::RED << "Error: Wallet address is required to fetch template from node" << std::endl;
            std::cerr << "Please provide wallet address with -w/--wallet or ensure " << BENCHMARK_FILE << " exists" << Color::RESET << std::endl;
            return;
        }
        
        // Try to fetch a template from the node
        try {
            XntRpcClient rpc_client(endpoint);
            if (!rpc_client.testConnection()) {
                std::cerr << Color::RED << "Error: Cannot connect to node at " << endpoint << std::endl;
                std::cerr << "Please ensure the node is running or provide a valid RPC URL." << Color::RESET << std::endl;
                return;
            }
            
            template_response = rpc_client.getBlockTemplate(g_miner_wallet_address);
            if (template_response.empty() || !template_response.contains("result")) {
                std::cerr << Color::RED << "Error: Failed to fetch block template from node" << Color::RESET << std::endl;
                return;
            }
            
            if (!saveBenchmarkTemplate(template_response)) {
                return;
            }
        } catch (const std::exception& e) {
            std::cerr << Color::RED << "Error fetching template: " << e.what() << Color::RESET << std::endl;
            return;
        }
    } else {
        std::cout << "Using existing benchmark template from " << BENCHMARK_FILE << std::endl;
        std::cout << Color::GREEN << "No RPC connection required - running offline benchmark" << Color::RESET << std::endl;
    }
    
    // Extract template object from JSON response (handles both RPC and direct formats)
    json template_obj;
    if (template_response.contains("result") && template_response["result"].contains("template")) {
        template_obj = template_response["result"]["template"];
    } else if (template_response.contains("template")) {
        template_obj = template_response["template"];
    } else {
        std::cerr << Color::RED << "Error: Invalid template format" << Color::RESET << std::endl;
        return;
    }
    
    // Parse the template into a PowPuzzle structure for mining
    std::string template_json_str = template_response.dump();
    PowPuzzle puzzle = parsePowPuzzle(template_json_str);
    if (!puzzle.is_valid()) {
        std::cerr << Color::RED << "Error: Invalid puzzle in template" << Color::RESET << std::endl;
        return;
    }
    
    // Initialize GPU
    int device_id = (gpu_id >= 0) ? gpu_id : 0;
    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        std::cerr << Color::RED << "CUDA Error: " << cudaGetErrorString(err) << Color::RESET << std::endl;
        return;
    }
    
    // Initialize GPU resources
    auto gpu_res = std::make_unique<GpuResources>(device_id);
    gpu_res->mining_mode = MiningMode::Solo;
    
    if (!preprocessPuzzle(puzzle, gpu_res.get())) {
        std::cerr << Color::RED << "Failed to initialize GPU resources" << Color::RESET << std::endl;
        return;
    }
    
    // Batch size: use global override if set, otherwise auto-detect for this GPU
    const uint64_t min_batch = 1;
    if (g_batch_size > 0) {
        gpu_res->optimal_max_nonces = std::max(g_batch_size, min_batch);
    } else {
        gpu_res->optimal_max_nonces = get_optimal_batch_size(device_id);
    }
    
    std::cout << "\n" << Color::BOLD << "Starting benchmark..." << Color::RESET << std::endl;
    std::cout << "  Batch size: " << gpu_res->optimal_max_nonces << " (" << (gpu_res->optimal_max_nonces / 1000000.0) << "M)" << std::endl;
    std::cout << "Press Ctrl+C to stop\n" << std::endl;
    
    // Benchmark configuration
    int BENCHMARK_DURATION_SEC = 10;        // Total benchmark duration
    if (const char* bench_env = std::getenv("XNT_BENCHMARK_SEC"); bench_env && bench_env[0] != '\0') {
        int v = std::atoi(bench_env);
        // Keep bounds sane so a typo doesn't run forever
        if (v > 0 && v <= 600) {
            BENCHMARK_DURATION_SEC = v;
        }
    }
    // Use GPU's optimal batch size for maximum performance
    // For RTX 5090 (256 SMs), this will be ~20M nonces, reducing kernel launch overhead
    uint64_t NONCES_PER_BATCH = gpu_res->optimal_max_nonces;
    
    // Extract the original difficulty threshold from the puzzle
    Digest original_threshold = hex_to_digest(puzzle.threshold);
    
    // Use a significantly easier target for benchmark mode to ensure we always
    // find solutions for validation testing. 1M× gives ~20-30 solutions per
    // 10-second run while keeping enough full batches for accurate sustained rate.
    constexpr uint64_t BENCHMARK_EASY_FACTOR = 1000000ULL;  // 1 million
    Digest test_target = make_target_easier(original_threshold, BENCHMARK_EASY_FACTOR);
    std::string test_target_hex = digest_to_hex(test_target);
    std::string original_threshold_hex = digest_to_hex(original_threshold);
    
    
    uint64_t total_nonces = 0;        // Always increases to avoid nonce reuse
    uint64_t measured_nonces = 0;     // Excludes warmup batches for steady-state rate
    uint64_t solutions_found = 0;
    uint64_t valid_solutions = 0;
    uint64_t invalid_threshold = 0;
    
    const char* warmup_env = std::getenv("XNT_WARMUP_BATCHES");
    int warmup_batches = 5;
    if (warmup_env && warmup_env[0] != '\0') {
        warmup_batches = std::max(0, std::atoi(warmup_env));
    }
    
    auto start_time = std::chrono::steady_clock::now();
    auto last_update = start_time;
    uint64_t last_nonces = 0;  // Track nonces at last update for instantaneous rate calculation
    int iteration = 0;
    
    install_signal_handlers();
    
    std::cout << "\n" << Color::BOLD << "Validation:" << Color::RESET << std::endl;
    std::cout << "  Original Threshold: " << original_threshold_hex << std::endl;
    std::cout << "  Test Threshold:     " << test_target_hex << " (" << BENCHMARK_EASY_FACTOR << "x easier)" << std::endl;
    std::cout << "  Validating against:  " << Color::YELLOW << "Test Threshold (" << BENCHMARK_EASY_FACTOR << "x easier)" << Color::RESET << std::endl;
    std::cout << "  Checking trailing zeros and threshold comparison" << std::endl;
    std::cout << std::endl;
    
    // Debug: per-batch timing (disabled by default for steady performance)
    bool debug_batch_timing = false;
    if (const char* dbg_env = std::getenv("XNT_DEBUG_BATCH"); dbg_env && dbg_env[0] == '1') {
        debug_batch_timing = true;
    }
    
    // Async mode: use double-buffered pipelining (set XNT_ASYNC_BENCH=1)
    bool use_async = false;
    if (const char* async_env = std::getenv("XNT_ASYNC_BENCH"); async_env && async_env[0] == '1') {
        use_async = true;
        std::cout << Color::CYAN << "  Mode: ASYNC (double-buffered, 2 streams)" << Color::RESET << std::endl;
    } else {
        std::cout << Color::CYAN << "  Mode: SYNC (single stream)" << Color::RESET << std::endl;
    }
    std::cout << std::endl;
    
    int batch_count = 0;
    int measured_batch_count = 0;
    int full_batch_count = 0;
    double full_batch_total_ms = 0.0;
    
    // Initialize async miner if using async mode
    std::unique_ptr<DoubleBufferedMiner> async_miner;
    int active_slot = 0;
    struct AsyncBatchInfo {
        uint64_t start_nonce;
        std::chrono::steady_clock::time_point start_time;
        bool pending;
    };
    AsyncBatchInfo async_batch_info[2] = {};
    
    if (use_async) {
        async_miner = std::make_unique<DoubleBufferedMiner>();
        if (!async_miner->initialize()) {
            std::cerr << Color::RED << "Failed to initialize async miner" << Color::RESET << std::endl;
            return;
        }
    }
    
    while (!stop_mining) {
        std::optional<MiningSolution> result;
        auto batch_start = std::chrono::steady_clock::now();
        Digest target = test_target;
        
        if (use_async) {
            // ============ ASYNC DOUBLE-BUFFERED MODE ============
            AsyncMiningSlot& slot = async_miner->slots[active_slot];
            AsyncBatchInfo& info = async_batch_info[active_slot];
            
            // Check if current slot has a pending result
            if (info.pending) {
                cudaError_t status = cudaEventQuery(slot.completion_event);
                if (status == cudaSuccess) {
                    // Kernel done - process result
                    auto now = std::chrono::steady_clock::now();
                    auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(now - info.start_time).count();
                    double batch_duration_ms = duration_us / 1000.0;
                    
                    batch_count++;
                    total_nonces += NONCES_PER_BATCH;
                    
                    const bool in_warmup = batch_count <= warmup_batches;
                    if (!in_warmup) {
                        measured_batch_count++;
                        measured_nonces += NONCES_PER_BATCH;
                        
                        int sol_found = *slot.h_solution_found_pinned;
                        if (!sol_found) {
                            full_batch_count++;
                            full_batch_total_ms += batch_duration_ms;
                        } else {
                            // Retrieve solution
                            MiningSolution sol;
                            cudaMemcpy(&sol.pow.nonce, slot.d_solution_nonce_digest, sizeof(Digest), cudaMemcpyDeviceToHost);
                            cudaMemcpy(sol.pow.path_a, slot.d_solution_path_a, MERKLE_TREE_HEIGHT_ * sizeof(Digest), cudaMemcpyDeviceToHost);
                            cudaMemcpy(sol.pow.path_b, slot.d_solution_path_b, MERKLE_TREE_HEIGHT_ * sizeof(Digest), cudaMemcpyDeviceToHost);
                            cudaMemcpy(&sol.kernel_final_hash, slot.d_solution_final_hash, sizeof(Digest), cudaMemcpyDeviceToHost);
                            sol.pow.root = gpu_res->buffer->merkle_root;
                            solutions_found++;
                            
                            // Validate solution
                            bool meets_threshold = digest_less_than_or_equal(sol.kernel_final_hash, test_target);
                            if (meets_threshold) {
                                valid_solutions++;
                                if (solutions_found <= 3) {
                                    std::string hash_hex = digest_to_hex(sol.kernel_final_hash);
                                    std::cout << "\n" << Color::GREEN << "[ASYNC VALIDATION] Valid solution found!" << Color::RESET << std::endl;
                                    std::cout << "  Kernel final_hash: " << hash_hex << std::endl;
                                }
                            } else {
                                invalid_threshold++;
                            }
                        }
                    } else if (batch_count == warmup_batches) {
                        // Reset after warmup
                        measured_nonces = 0;
                        measured_batch_count = 0;
                        full_batch_count = 0;
                        full_batch_total_ms = 0.0;
                        solutions_found = 0;
                        valid_solutions = 0;
                        invalid_threshold = 0;
                        start_time = std::chrono::steady_clock::now();
                        last_update = start_time;
                        last_nonces = 0;
                    }
                    
                    info.pending = false;
                } else if (status == cudaErrorNotReady) {
                    // Switch to other slot
                    active_slot = 1 - active_slot;
                    if (async_batch_info[active_slot].pending) {
                        // Both busy, wait
                        cudaStreamSynchronize(slot.stream);
                    }
                    continue;
                }
            }
            
            // Launch new batch on current slot
            *slot.h_solution_found_pinned = 0;
            cudaMemsetAsync(slot.d_solution_found, 0, sizeof(int), slot.stream);
            
            int threads_per_block, blocks_per_grid;
            calculate_mining_launch_config(NONCES_PER_BATCH, threads_per_block, blocks_per_grid, device_id);
            
            if (!gpu_res->buffer->gpu_range_initialized) {
                GpuNonceRange gpu_range = calculate_gpu_range(device_id, 1);
                cudaMemcpyToSymbol(d_gpu_range_start, &gpu_range.range_start, sizeof(uint64_t));
                cudaMemcpyToSymbol(d_gpu_range_size, &gpu_range.range_size, sizeof(uint64_t));
                initialize_top_tree_cache(gpu_res->buffer->d_merkle_tree, gpu_res->buffer->num_leafs);
                gpu_res->buffer->gpu_range_initialized = true;
            }
            
            parallel_mining_kernel_high_vram<<<blocks_per_grid, threads_per_block, 0, slot.stream>>>(
                gpu_res->buffer->d_leafs,
                gpu_res->buffer->d_merkle_tree,
                gpu_res->buffer->index_picker_preimage,
                target,
                total_nonces,
                NONCES_PER_BATCH,
                gpu_res->buffer->num_leafs,
                MERKLE_TREE_HEIGHT_,
                gpu_res->buffer->mast_paths,
                gpu_res->buffer->hash,
                gpu_res->buffer->consensus_rule_set,
                slot.d_solution_nonce,
                slot.d_solution_found,
                slot.d_solution_path_a,
                slot.d_solution_path_b,
                slot.d_solution_nonce_digest,
                slot.d_solution_final_hash);
            
            cudaMemcpyAsync(slot.h_solution_found_pinned, slot.d_solution_found, sizeof(int), cudaMemcpyDeviceToHost, slot.stream);
            cudaEventRecord(slot.completion_event, slot.stream);
            
            info.start_nonce = total_nonces;
            info.start_time = std::chrono::steady_clock::now();
            info.pending = true;
            
            active_slot = 1 - active_slot;
            
            // Update progress display
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();
            auto since_update = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_update).count();
            if (since_update >= 1000 && measured_nonces > 0) {
                uint64_t nonces_since_update = measured_nonces - last_nonces;
                double time_since_update_sec = since_update / 1000.0;
                double hash_rate = (nonces_since_update / 1000000.0) / std::max(0.001, time_since_update_sec);
                std::cout << "\r[Benchmark ASYNC] Hash Rate: " << Color::CYAN << std::fixed << std::setprecision(2) 
                          << hash_rate << " MH/s" << Color::RESET 
                          << " | Nonces: " << measured_nonces 
                          << " | Time: " << elapsed << "s" << std::flush;
                last_update = now;
                last_nonces = measured_nonces;
            }
            if (elapsed >= BENCHMARK_DURATION_SEC) break;
        } else {
            // ============ SYNC MODE (original) ============
            result = mine_pow_with_buffer(
                *gpu_res->buffer,
                target,
                gpu_res->buffer->mast_paths,
                total_nonces,
                NONCES_PER_BATCH,
                gpu_res->buffer->consensus_rule_set,
                nullptr
            );
        
            auto batch_end = std::chrono::steady_clock::now();
            auto batch_duration_us = std::chrono::duration_cast<std::chrono::microseconds>(batch_end - batch_start).count();
            double batch_duration_ms = batch_duration_us / 1000.0;
            double batch_hashrate = (NONCES_PER_BATCH / 1000000.0) / (batch_duration_ms / 1000.0);
            
            batch_count++;
            total_nonces += NONCES_PER_BATCH;
            
            const bool in_warmup = batch_count <= warmup_batches;
            if (in_warmup) {
                if (debug_batch_timing) {
                    std::cout << "\n[DEBUG] Warmup Batch #" << batch_count 
                              << " | Nonces: " << NONCES_PER_BATCH 
                              << " | Duration: " << std::fixed << std::setprecision(2) << batch_duration_ms << " ms"
                              << " | Rate: " << std::setprecision(2) << batch_hashrate << " MH/s"
                              << " | Solution: " << (result.has_value() ? "YES" : "NO")
                              << std::endl;
                }
                
                if (batch_count == warmup_batches) {
                    // Reset measurement after warmup to avoid boost/thermal ramp affecting results
                    measured_nonces = 0;
                    measured_batch_count = 0;
                    full_batch_count = 0;
                    full_batch_total_ms = 0.0;
                    solutions_found = 0;
                    valid_solutions = 0;
                    invalid_threshold = 0;
                    start_time = std::chrono::steady_clock::now();
                    last_update = start_time;
                    last_nonces = 0;
                }
                continue;
            }
            
            measured_batch_count++;
            measured_nonces += NONCES_PER_BATCH;
            
            // Track full batches (no solution found = processed all nonces)
            if (!result.has_value()) {
                full_batch_count++;
                full_batch_total_ms += batch_duration_ms;
            }
            
            if (debug_batch_timing) {
                std::cout << "\n[DEBUG] Batch #" << measured_batch_count 
                          << " | Nonces: " << NONCES_PER_BATCH 
                          << " | Duration: " << std::fixed << std::setprecision(2) << batch_duration_ms << " ms"
                          << " | Rate: " << std::setprecision(2) << batch_hashrate << " MH/s"
                          << " | Solution: " << (result.has_value() ? "YES" : "NO")
                          << std::endl;
            }
            
            // Validate any solution found by the kernel
            if (result.has_value()) {
                solutions_found++;
                
                // Extract solution components
                const MiningSolution& mining_solution = result.value();
                const Digest& kernel_final_hash = mining_solution.kernel_final_hash;
                
                // Convert to hex for display
                std::string kernel_final_hash_hex = digest_to_hex(kernel_final_hash);
                
                // Validate using the kernel's final_hash (the value the kernel actually checked)
                // This matches the Rust node's validation: final_hash <= threshold
                bool meets_threshold = digest_less_than_or_equal(kernel_final_hash, test_target);
                
                // Extract trailing hex digits for display (matches the format shown in node logs)
                std::string pow_trailing = "";
                std::string threshold_trailing = "";
                if (kernel_final_hash_hex.length() >= 16) {
                    pow_trailing = kernel_final_hash_hex.substr(kernel_final_hash_hex.length() - 16);
                }
                if (test_target_hex.length() >= 16) {
                    threshold_trailing = test_target_hex.substr(test_target_hex.length() - 16);
                }
                
                // Validate solution: kernel_final_hash must be <= threshold
                if (meets_threshold) {
                    valid_solutions++;
                    
                    // Log first few valid solutions for verification (limit output to avoid spam)
                    if (solutions_found <= 3) {
                        std::cout << "\n" << Color::GREEN << "[VALIDATION] ✓ Valid solution found! (Kernel final_hash <= threshold)" << Color::RESET << std::endl;
                        std::cout << "  Kernel final_hash: " << kernel_final_hash_hex << std::endl;
                        std::cout << "  Test Threshold:    " << test_target_hex << std::endl;
                        std::cout << "  Meets threshold:    " << Color::GREEN << "YES" << Color::RESET << std::endl;
                        if (!pow_trailing.empty()) {
                            std::cout << "  Final Hash Trailing: " << pow_trailing << std::endl;
                        }
                        if (!threshold_trailing.empty()) {
                            std::cout << "  Threshold Trailing:  " << threshold_trailing << std::endl;
                        }
                    }
                } else {
                    invalid_threshold++;
                    // Log first few invalid solutions for debugging
                    if (solutions_found <= 3) {
                        std::cout << "\n" << Color::YELLOW << "[VALIDATION] ✗ Solution does not meet threshold" << Color::RESET << std::endl;
                        std::cout << "  " << Color::DIM << "Note: kernel_final_hash > threshold" << Color::RESET << std::endl;
                        std::cout << "  Kernel final_hash: " << kernel_final_hash_hex << std::endl;
                        std::cout << "  Test Threshold:    " << test_target_hex << std::endl;
                        std::cout << "  Meets threshold:   " << Color::RED << "NO" << Color::RESET << std::endl;
                        if (!pow_trailing.empty()) {
                            std::cout << "  Final Hash Trailing: " << pow_trailing << std::endl;
                            // Check for trailing zeros (valid solutions typically have trailing zeros,
                            // matching the pattern shown in the Rust node's validation logs)
                            bool has_trailing_zeros = (pow_trailing.find_first_not_of('0') == std::string::npos) || 
                                                      (pow_trailing.back() == '0');
                            std::cout << "  Has trailing zeros:  " << (has_trailing_zeros ? Color::GREEN : Color::RED) 
                                      << (has_trailing_zeros ? "YES" : "NO") << Color::RESET << std::endl;
                        }
                        if (!threshold_trailing.empty()) {
                            std::cout << "  Threshold Trailing:  " << threshold_trailing << std::endl;
                        }
                    }
                }
            }
            
            iteration++;
            
            // Update progress display and check if benchmark duration has elapsed
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();
            auto since_update = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_update).count();
            
            // Update hash rate display every second
            // Calculate instantaneous rate (nonces processed in last second) instead of cumulative average
            if (since_update >= 1000) {
                uint64_t nonces_since_update = measured_nonces - last_nonces;
                double time_since_update_sec = since_update / 1000.0;
                // Calculate instantaneous hash rate: nonces in last second / time elapsed
                double hash_rate = (nonces_since_update / 1000000.0) / std::max(0.001, time_since_update_sec);
                std::cout << "\r[Benchmark] Hash Rate: " << Color::CYAN << std::fixed << std::setprecision(2) 
                          << hash_rate << " MH/s" << Color::RESET 
                          << " | Nonces: " << measured_nonces 
                          << " | Time: " << elapsed << "s" << std::flush;
                last_update = now;
                last_nonces = measured_nonces;  // Update tracked nonces for next calculation
            }
            
            // Stop benchmark after the configured duration
            if (elapsed >= BENCHMARK_DURATION_SEC) {
                break;
            }
        } // end else (sync mode)
    }
    
    // Drain any pending async batches
    if (use_async && async_miner) {
        for (int i = 0; i < 2; i++) {
            if (async_batch_info[i].pending) {
                cudaStreamSynchronize(async_miner->slots[i].stream);
            }
        }
    }
    
    auto end_time = std::chrono::steady_clock::now();
    auto total_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    double total_hash_rate = (measured_nonces / 1000000.0) / (total_elapsed / 1000.0);
    
    std::cout << "\n\n" << Color::BOLD << "=== Benchmark Results ===" << Color::RESET << std::endl;
    std::cout << "  Total Nonces:  " << measured_nonces << std::endl;
    std::cout << "  Total Batches: " << measured_batch_count << std::endl;
    if (warmup_batches > 0) {
        std::cout << "  Warmup Batches: " << warmup_batches << std::endl;
    }
    std::cout << "  Batch Size:    " << NONCES_PER_BATCH << " (" << (NONCES_PER_BATCH / 1000000.0) << "M)" << std::endl;
    std::cout << "  Duration:      " << (total_elapsed / 1000.0) << " seconds" << std::endl;
    if (measured_batch_count > 0) {
        std::cout << "  Avg ms/batch:  " << std::fixed << std::setprecision(2) << (total_elapsed / (double)measured_batch_count) << " ms" << std::endl;
    }
    std::cout << "  Average Rate:  " << Color::GREEN << std::fixed << std::setprecision(2) 
              << total_hash_rate << " MH/s" << Color::RESET << " (includes early exits)" << std::endl;
    
    // Show sustained rate from full batches only (only meaningful for sync mode)
    if (full_batch_count > 0 && !use_async) {
        double full_batch_avg_ms = full_batch_total_ms / full_batch_count;
        double sustained_rate = (NONCES_PER_BATCH / 1000000.0) / (full_batch_avg_ms / 1000.0);
        std::cout << "  " << Color::BOLD << "Sustained Rate: " << Color::YELLOW << std::setprecision(2) 
                  << sustained_rate << " MH/s" << Color::RESET << " (full batches only, n=" << full_batch_count << ")" << std::endl;
    } else if (use_async) {
        // For async mode, the average rate IS the sustained rate due to double-buffering
        std::cout << "  " << Color::BOLD << "Sustained Rate: " << Color::YELLOW << std::setprecision(2) 
                  << total_hash_rate << " MH/s" << Color::RESET << " (async double-buffered)" << std::endl;
    }
    std::cout << std::endl;
    
    std::cout << Color::BOLD << "=== Validation Results ===" << Color::RESET << std::endl;
    std::cout << "  Solutions Found:     " << solutions_found << std::endl;
    std::cout << "  Valid (meets threshold): " << Color::GREEN << valid_solutions << Color::RESET << std::endl;
    if (invalid_threshold > 0) {
        std::cout << "  Invalid (threshold):  " << Color::YELLOW << invalid_threshold << Color::RESET << std::endl;
        std::cout << "    (kernel_final_hash > threshold, rejected like Rust node)" << std::endl;
    }
    if (solutions_found > 0) {
        double valid_rate = (valid_solutions * 100.0) / solutions_found;
        std::cout << "  Validation Rate:     " << std::fixed << std::setprecision(1) 
                  << valid_rate << "%" << std::endl;
        std::cout << "\n  " << Color::DIM << "Note: Trailing zeros in hex representation indicate" << std::endl;
        std::cout << "  the exact format used by the node for validation." << Color::RESET << std::endl;
    }
    std::cout << std::endl;
    
    // Cleanup GPU resources
    if (gpu_res->buffer) {
        gpu_res->buffer->cleanup();
        gpu_res->buffer.reset();
    }
}

int main(int argc, char* argv[]) {
    enable_ansi_colors();
    std::cout.setf(std::ios::unitbuf);
    std::cerr.setf(std::ios::unitbuf);
    
    // Batch size: env XNT_BATCH_SIZE as default (0=auto), --batch overrides
    if (const char* env = std::getenv("XNT_BATCH_SIZE"); env && env[0] != '\0') {
        try { g_batch_size = std::stoull(env); } catch (...) { /* keep default */ }
    }
    
    std::string endpoint = "http://127.0.0.1:9897";
    std::string stratum_password = "x";
    std::string stratum_worker_name = "xnt-miner";
    MiningMode mining_mode = MiningMode::Solo;
    bool show_help = false;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--help" || arg == "-h") {
            show_help = true;
        } else if ((arg == "--wallet" || arg == "-w") && i + 1 < argc) {
            g_miner_wallet_address = argv[++i];
        } else if ((arg == "--device" || arg == "-d") && i + 1 < argc) {
            try {
                g_gpu_device_id = std::stoi(argv[++i]);
                if (g_gpu_device_id < 0) {
                    std::cerr << Color::RED << "Error: Device ID must be non-negative" << Color::RESET << std::endl;
                    return 1;
                }
            } catch (const std::exception& e) {
                std::cerr << Color::RED << "Error: Invalid device ID: " << argv[i] << Color::RESET << std::endl;
                return 1;
            }
        } else if ((arg == "--host" || arg == "-H") && i + 1 < argc) {
            std::string host = argv[++i];
            // Host option only applies to solo mining mode (not stratum pools)
            if (mining_mode == MiningMode::Solo) {
                size_t port_start = endpoint.find(":", 7);
                if (port_start != std::string::npos) {
                    std::string port = endpoint.substr(port_start + 1);
                    endpoint = "http://" + host + ":" + port;
                } else {
                    endpoint = "http://" + host + ":9897";
                }
            }
        } else if ((arg == "--port" || arg == "-p") && i + 1 < argc) {
            try {
                int port = std::stoi(argv[++i]);
                if (port < 1 || port > 65535) {
                    std::cerr << Color::RED << "Error: Port must be between 1 and 65535" << Color::RESET << std::endl;
                    return 1;
                }
                // Port option only applies to solo mining mode (not stratum pools)
                if (mining_mode == MiningMode::Solo) {
                    size_t host_start = endpoint.find("://") + 3;
                    size_t port_start = endpoint.find(":", host_start);
                    if (port_start != std::string::npos) {
                        endpoint = endpoint.substr(0, port_start + 1) + std::to_string(port);
                    } else {
                        endpoint = endpoint + ":" + std::to_string(port);
                    }
                }
            } catch (const std::exception& e) {
                std::cerr << Color::RED << "Error: Invalid port number: " << argv[i] << Color::RESET << std::endl;
                return 1;
            }
        } else if ((arg == "--rpc-url") && i + 1 < argc) {
            endpoint = argv[++i];
            mining_mode = MiningMode::Solo;  // RPC URL explicitly sets solo mining mode
        } else if ((arg == "--stratum") && i + 1 < argc) {
            endpoint = argv[++i];
            mining_mode = MiningMode::Stratum;
        } else if ((arg == "--stratum-pass") && i + 1 < argc) {
            stratum_password = argv[++i];
        } else if ((arg == "--stratum-worker") && i + 1 < argc) {
            stratum_worker_name = argv[++i];
        } else if (arg == "--test-mode") {
            g_test_mode = true;
        } else if (arg == "--benchmark") {
            g_benchmark_mode = true;
        } else if ((arg == "--fetch-interval") && i + 1 < argc) {
            try {
                int interval = std::stoi(argv[++i]);
                if (interval < 1) {
                    std::cerr << Color::RED << "Error: Fetch interval must be at least 1 second" << Color::RESET << std::endl;
                    return 1;
                }
                g_fetch_interval_sec = interval;
            } catch (const std::exception& e) {
                std::cerr << Color::RED << "Error: Invalid fetch interval: " << argv[i] << Color::RESET << std::endl;
                return 1;
            }
        } else if ((arg == "--batch") && i + 1 < argc) {
            try {
                g_batch_size = std::stoull(argv[++i]);
            } catch (const std::exception& e) {
                std::cerr << Color::RED << "Error: Invalid batch size: " << argv[i] << Color::RESET << std::endl;
                return 1;
            }
        } else if (arg[0] == '-') {
            std::cerr << Color::RED << "Error: Unknown option: " << arg << Color::RESET << std::endl;
            std::cerr << "Use --help or -h for usage information" << std::endl;
            return 1;
        } else {
            std::cerr << Color::RED << "Error: Unexpected argument: " << arg << Color::RESET << std::endl;
            std::cerr << "Use --help or -h for usage information" << std::endl;
            return 1;
        }
    }
    
    if (show_help) {
        print_usage(argv[0]);
        return 0;
    }
    
    // Handle benchmark mode early (before normal mining setup)
    // Wallet address is only required if the benchmark template doesn't exist yet
    if (g_benchmark_mode) {
        // Check if a cached template exists - if so, wallet is not required
        json test_template;
        bool template_exists = loadBenchmarkTemplate(test_template);
        
        if (!template_exists && g_miner_wallet_address.empty()) {
            std::cerr << "\n" << Color::RED << Color::BOLD << "Error: Wallet address is required to fetch template" << Color::RESET << std::endl;
            std::cerr << "\nFirst-time benchmark requires wallet address to fetch template from node." << std::endl;
            std::cerr << "After template is saved, wallet is not required.\n" << std::endl;
            print_usage(argv[0]);
            return 1;
        }
        
        print_system_info();
        cudaDeviceReset();
        install_signal_handlers();
        runBenchmark(endpoint, g_gpu_device_id);
        cudaDeviceReset();
        return 0;
    }
    
    // Normal mining mode: wallet address is required for block rewards
    if (g_miner_wallet_address.empty()) {
        std::cerr << "\n" << Color::RED << Color::BOLD << "Error: Wallet address is required" << Color::RESET << std::endl;
        std::cerr << "\nMining requires your Neptune wallet address.\n" << std::endl;
        print_usage(argv[0]);
        return 1;
    }
    
    // Auto-detect mining mode from URL format if not explicitly set
    // (stratum:// URLs indicate pool mining, http:// indicates solo mining)
    if (mining_mode == MiningMode::Solo) {
        mining_mode = detect_mining_mode(endpoint);
    }
    
    g_miner_worker_name = stratum_worker_name;
    print_system_info();
    
    std::cout << Color::BOLD << "Configuration:" << Color::RESET << std::endl;
    std::cout << "  Mining Mode:   " << mining_mode_name(mining_mode) << std::endl;
    if (mining_mode == MiningMode::Stratum) {
        std::cout << "  Pool:          " << endpoint << std::endl;
        std::cout << "  Worker:        " << g_miner_worker_name << std::endl;
    } else {
        std::cout << "  RPC Endpoint:  " << endpoint << std::endl;
    }
    std::cout << "  Wallet:        " << shorten_address(g_miner_wallet_address) << std::endl;
    if (g_gpu_device_id >= 0) {
        std::cout << "  Device:        GPU " << g_gpu_device_id << std::endl;
    } else {
        std::cout << "  Device:        All available GPUs" << std::endl;
    }
    if (g_test_mode) {
        std::cout << "  Test Mode:     " << Color::YELLOW << "ENABLED" << Color::RESET << std::endl;
    }
    std::cout << "  Batch size:    " << (g_batch_size == 0 ? "auto" : std::to_string(g_batch_size) + " nonces") << std::endl;
    std::cout << std::endl;
    
    cudaDeviceReset();
    install_signal_handlers();
    
    try {
        startUnifiedMining(endpoint, g_gpu_device_id, mining_mode, stratum_password);
    } catch (const std::exception& e) {
        std::string error_msg = e.what();
        // Sanitize error messages: replace full wallet address with shortened version
        // to avoid exposing sensitive information in logs
        if (!g_miner_wallet_address.empty() && error_msg.find(g_miner_wallet_address) != std::string::npos) {
            size_t pos = 0;
            while ((pos = error_msg.find(g_miner_wallet_address, pos)) != std::string::npos) {
                error_msg.replace(pos, g_miner_wallet_address.length(), shorten_address(g_miner_wallet_address));
                pos += shorten_address(g_miner_wallet_address).length();
            }
        }
        std::cerr << Color::RED << "Error: " << error_msg << Color::RESET << std::endl;
        return 1;
    }
    
    // Cleanup CUDA resources before exit
    cudaDeviceReset();
    std::cout << "\n" << Color::GREEN << "Mining stopped" << Color::RESET << std::endl;
    
    return 0;
}
