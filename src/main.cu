#include "mining.cuh"

void print_usage(const char* program_name) {
    std::cerr << "\n" << Color::BOLD << "Usage:" << Color::RESET << std::endl;
    std::cerr << "  " << program_name << " [OPTIONS] -w, --wallet ADDRESS\n" << std::endl;
    
    std::cerr << Color::BOLD << "Required:" << Color::RESET << std::endl;
    std::cerr << "  -w, --wallet ADDRESS  Your Neptune wallet address\n" << std::endl;
    
    std::cerr << Color::BOLD << "Optional:" << Color::RESET << std::endl;
    std::cerr << "  -d, --device ID       Use specific GPU device ID (default: all GPUs)" << std::endl;
    std::cerr << "  --rpc-url URL        RPC endpoint URL (default: http://127.0.0.1:9899)" << std::endl;
    std::cerr << "  -h, --help            Show this help message\n" << std::endl;
    
    std::cerr << Color::BOLD << "Examples:" << Color::RESET << std::endl;
    std::cerr << "  " << program_name << " -w nolgam..." << std::endl;
    std::cerr << "  " << program_name << " --wallet nolgam... --device 0" << std::endl;
    std::cerr << "  " << program_name << " -w nolgam... --rpc-url http://192.168.1.100:9899\n" << std::endl;
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
    
    std::string rpc_url = "http://127.0.0.1:9899";
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
            size_t port_start = rpc_url.find(":", 7);
            if (port_start != std::string::npos) {
                std::string port = rpc_url.substr(port_start + 1);
                rpc_url = "http://" + host + ":" + port;
            } else {
                rpc_url = "http://" + host + ":9899";
            }
        } else if ((arg == "--port" || arg == "-p") && i + 1 < argc) {
            try {
                int port = std::stoi(argv[++i]);
                if (port < 1 || port > 65535) {
                    std::cerr << Color::RED << "Error: Port must be between 1 and 65535" << Color::RESET << std::endl;
                    return 1;
                }
                size_t host_start = rpc_url.find("://") + 3;
                size_t port_start = rpc_url.find(":", host_start);
                if (port_start != std::string::npos) {
                    rpc_url = rpc_url.substr(0, port_start + 1) + std::to_string(port);
                } else {
                    rpc_url = rpc_url + ":" + std::to_string(port);
                }
            } catch (const std::exception& e) {
                std::cerr << Color::RED << "Error: Invalid port number: " << argv[i] << Color::RESET << std::endl;
                return 1;
            }
        } else if ((arg == "--rpc-url") && i + 1 < argc) {
            rpc_url = argv[++i];
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
    
    print_system_info();
    
    std::cout << Color::BOLD << "Configuration:" << Color::RESET << std::endl;
    std::cout << "  Mining Mode:   SOLO" << std::endl;
    std::cout << "  RPC Endpoint:  " << rpc_url << std::endl;
    std::cout << "  Wallet:        " << g_miner_wallet_address.substr(0, 20) << "..." << std::endl;
    if (g_gpu_device_id >= 0) {
        std::cout << "  Device:        GPU " << g_gpu_device_id << std::endl;
    } else {
        std::cout << "  Device:        All available GPUs" << std::endl;
    }
    std::cout << std::endl;
    
    cudaDeviceReset();
    install_signal_handlers();
    
    try {
        startUnifiedMining(rpc_url, g_gpu_device_id);
    } catch (const std::exception& e) {
        Console::showCursor();
        std::cerr << Color::RED << "Error: " << e.what() << Color::RESET << std::endl;
        return 1;
    }
    
    Console::showCursor();
    cudaDeviceReset();
    std::cout << "\n" << Color::GREEN << "Mining stopped" << Color::RESET << std::endl;
    
    return 0;
}