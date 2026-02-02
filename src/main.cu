#include "mining.cuh"
#include "connection_multiplexer.cuh"
#include "mining_client.h"
#include "network.cuh"
#include "rpc_client.cuh"
#include <fstream>
#include <iomanip>
#include <chrono>

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
    
    std::cout << "\n" << Color::BOLD << "Starting benchmark..." << Color::RESET << std::endl;
    std::cout << "Press Ctrl+C to stop\n" << std::endl;
    
    // Benchmark configuration
    const int BENCHMARK_DURATION_SEC = 10;        // Total benchmark duration
    // Use GPU's optimal batch size for maximum performance
    // For RTX 5090 (256 SMs), this will be ~20M nonces, reducing kernel launch overhead
    uint64_t NONCES_PER_BATCH = gpu_res->optimal_max_nonces;
    
    // Extract the original difficulty threshold from the puzzle
    Digest original_threshold = hex_to_digest(puzzle.threshold);
    
    // Use a significantly easier target (100000x) for benchmark mode to ensure
    // we find solutions quickly for validation testing. This allows us to verify
    // that our validation logic matches the Rust node's behavior.
    Digest test_target = make_target_easier(original_threshold, 100000ULL);
    std::string test_target_hex = digest_to_hex(test_target);
    std::string original_threshold_hex = digest_to_hex(original_threshold);
    
    
    uint64_t total_nonces = 0;
    uint64_t solutions_found = 0;
    uint64_t valid_solutions = 0;
    uint64_t invalid_threshold = 0;
    
    auto start_time = std::chrono::steady_clock::now();
    auto last_update = start_time;
    int iteration = 0;
    
    install_signal_handlers();
    
    std::cout << "\n" << Color::BOLD << "Validation:" << Color::RESET << std::endl;
    std::cout << "  Original Threshold: " << original_threshold_hex << std::endl;
    std::cout << "  Test Threshold:     " << test_target_hex << " (100000x easier for testing)" << std::endl;
    std::cout << "  Validating against:  " << Color::YELLOW << "Test Threshold (100000x easier)" << Color::RESET << std::endl;
    std::cout << "  Checking trailing zeros and threshold comparison" << std::endl;
    std::cout << std::endl;
    
    while (!stop_mining) {
        auto batch_start = std::chrono::steady_clock::now();
        
        // Mine a batch of nonces using the easier test target
        Digest target = test_target;
        auto result = mine_pow_with_buffer(
            *gpu_res->buffer,
            target,
            gpu_res->buffer->mast_paths,
            total_nonces,
            NONCES_PER_BATCH,
            gpu_res->buffer->consensus_rule_set,
            nullptr
        );
        
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
        
        total_nonces += NONCES_PER_BATCH;
        iteration++;
        
        // Update progress display and check if benchmark duration has elapsed
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();
        auto since_update = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_update).count();
        
        // Update hash rate display every second
        if (since_update >= 1000) {
            double hash_rate = (total_nonces / 1000000.0) / std::max(1.0, static_cast<double>(elapsed));
            std::cout << "\r[Benchmark] Hash Rate: " << Color::CYAN << std::fixed << std::setprecision(2) 
                      << hash_rate << " MH/s" << Color::RESET 
                      << " | Nonces: " << total_nonces 
                      << " | Time: " << elapsed << "s" << std::flush;
            last_update = now;
        }
        
        // Stop benchmark after the configured duration
        if (elapsed >= BENCHMARK_DURATION_SEC) {
            break;
        }
    }
    
    auto end_time = std::chrono::steady_clock::now();
    auto total_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    double total_hash_rate = (total_nonces / 1000000.0) / (total_elapsed / 1000.0);
    
    std::cout << "\n\n" << Color::BOLD << "=== Benchmark Results ===" << Color::RESET << std::endl;
    std::cout << "  Total Nonces:  " << total_nonces << std::endl;
    std::cout << "  Duration:      " << (total_elapsed / 1000.0) << " seconds" << std::endl;
    std::cout << "  Average Rate:  " << Color::GREEN << std::fixed << std::setprecision(2) 
              << total_hash_rate << " MH/s" << Color::RESET << std::endl;
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
