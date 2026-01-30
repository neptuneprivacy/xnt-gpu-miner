#include "mining.cuh"
#include "connection_multiplexer.cuh"
#include "mining_client.h"

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
            // Only modify if we're in solo mode (not stratum)
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
                // Only modify if we're in solo mode (not stratum)
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
            mining_mode = MiningMode::Solo;  // Explicit solo mode
        } else if ((arg == "--stratum") && i + 1 < argc) {
            endpoint = argv[++i];
            mining_mode = MiningMode::Stratum;
        } else if ((arg == "--stratum-pass") && i + 1 < argc) {
            stratum_password = argv[++i];
        } else if ((arg == "--stratum-worker") && i + 1 < argc) {
            stratum_worker_name = argv[++i];
        } else if (arg == "--test-mode") {
            g_test_mode = true;
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
    
    if (g_miner_wallet_address.empty()) {
        std::cerr << "\n" << Color::RED << Color::BOLD << "Error: Wallet address is required" << Color::RESET << std::endl;
        std::cerr << "\nMining requires your Neptune wallet address.\n" << std::endl;
        print_usage(argv[0]);
        return 1;
    }
    
    // Auto-detect mode from URL if not explicitly set
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
        // Replace full wallet address with shortened version in error messages
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
    
    cudaDeviceReset();
    std::cout << "\n" << Color::GREEN << "Mining stopped" << Color::RESET << std::endl;
    
    return 0;
}
