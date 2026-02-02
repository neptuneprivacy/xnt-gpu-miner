#ifndef XNT_COMMON_CUH
#define XNT_COMMON_CUH

#include <iostream>
#include <chrono>
#include <vector>
#include <iomanip>
#include <atomic>
#include <thread>
#include <string>
#include <optional>
#include <functional>
#include <cstdint>
#include <array>
#include <sstream>
#include <algorithm>
#include <memory>
#include <fstream>
#include <ctime>
#include <cstdlib>
#include <signal.h>
#include <random>
#include <mutex>
#include <condition_variable>
#include <queue>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>

#ifdef _WIN32
    #include <winsock2.h>
    #include <ws2tcpip.h>
    #pragma comment(lib, "ws2_32.lib")
    #define NOMINMAX
    #include <windows.h>
    typedef int socklen_t;
    typedef SOCKET socket_t;
    typedef int ssize_t;
    #define close closesocket
    #define INVALID_SOCKET_VALUE INVALID_SOCKET
    #define SOCKET_ERROR_VALUE SOCKET_ERROR
    inline void platform_socket_init() {
        WSADATA wsaData;
        WSAStartup(MAKEWORD(2, 2), &wsaData);
    }
    inline void platform_socket_cleanup() {
        WSACleanup();
    }
#else
    #include <pthread.h>
    #include <sys/socket.h>
    #include <sys/select.h>
    #include <sys/types.h>
    #include <netinet/in.h>
    #include <arpa/inet.h>
    #include <unistd.h>
    #include <fcntl.h>
    #include <errno.h>
    #include <netdb.h>
    #include <pwd.h>
    typedef int socket_t;
    #define INVALID_SOCKET_VALUE -1
    #define SOCKET_ERROR_VALUE -1
    inline void platform_socket_init() {}
    inline void platform_socket_cleanup() {}
    #ifdef __linux__
        #include <endian.h>
    #elif defined(__APPLE__) || defined(__FreeBSD__)
        #include <machine/endian.h>
    #endif
#endif

#if __has_include("nlohmann/json.hpp")
    #include "nlohmann/json.hpp"
#else
    #include <nlohmann/json.hpp>
#endif
using json = nlohmann::json;

#define LOG_DEBUG(msg) do { } while(0)

#define LOG_ERROR(stage, err) do { \
    std::cerr << "[ERROR] " << stage << ": " << cudaGetErrorString(err) << std::endl; \
} while(0)

#define CUDA_CHECK(stmt, stage, retval) do { \
    cudaError_t err = (stmt); \
    if (err != cudaSuccess) { \
        LOG_ERROR(stage, err); \
        return retval; \
    } \
} while(0)

#define CUDA_SYNC(stage, retval) do { \
    cudaError_t err = cudaDeviceSynchronize(); \
    if (err != cudaSuccess) { \
        LOG_ERROR(stage, err); \
        return retval; \
    } \
} while(0)

inline void clear_cuda_errors(const char* stage) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        LOG_ERROR(stage, err);
    }
}

inline bool reset_gpu_on_error(const char* stage) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        LOG_ERROR(stage, err);
        if (err == cudaErrorIllegalAddress || 
            err == cudaErrorMemoryAllocation ||
            err == cudaErrorLaunchFailure ||
            err == cudaErrorUnknown) {
            cudaDeviceSynchronize();
            cudaError_t reset_err = cudaDeviceReset();
            return reset_err == cudaSuccess;
        }
    }
    return false;
}

constexpr int CONSENSUS_REBOOT = 0;
constexpr int CONSENSUS_HARDFORK_ALPHA = 1;
constexpr int CONSENSUS_XNT = 2;

extern std::atomic<bool> stop_mining;
extern std::string g_miner_wallet_address;
extern std::string g_miner_worker_name;
extern std::atomic<int> g_total_gpu_count;
extern int g_gpu_device_id;
extern bool g_test_mode;
extern bool g_benchmark_mode;
extern int g_fetch_interval_sec;
extern std::mutex g_log_mutex;

// Helper function to shorten wallet address for display
inline std::string shorten_address(const std::string& addr, size_t prefix_len = 12, size_t suffix_len = 8) {
    if (addr.length() <= prefix_len + suffix_len) {
        return addr;
    }
    return addr.substr(0, prefix_len) + "..." + addr.substr(addr.length() - suffix_len);
}

static constexpr uint64_t DEFAULT_BATCH_SIZE = 262144ULL;


inline void enable_ansi_colors() {
#ifdef _WIN32
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    if (hOut != INVALID_HANDLE_VALUE) {
        DWORD dwMode = 0;
        if (GetConsoleMode(hOut, &dwMode)) {
            dwMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
            SetConsoleMode(hOut, dwMode);
        }
    }
    HANDLE hErr = GetStdHandle(STD_ERROR_HANDLE);
    if (hErr != INVALID_HANDLE_VALUE) {
        DWORD dwMode = 0;
        if (GetConsoleMode(hErr, &dwMode)) {
            dwMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
            SetConsoleMode(hErr, dwMode);
        }
    }
#endif
}

namespace Color {
    constexpr const char* RESET = "\033[0m";
    constexpr const char* BOLD = "\033[1m";
    constexpr const char* DIM = "\033[2m";
    constexpr const char* RED = "\033[31m";
    constexpr const char* GREEN = "\033[32m";
    constexpr const char* YELLOW = "\033[33m";
    constexpr const char* BLUE = "\033[34m";
    constexpr const char* MAGENTA = "\033[35m";
    constexpr const char* CYAN = "\033[36m";
    constexpr const char* WHITE = "\033[37m";
}

inline std::string get_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    auto tm = *std::localtime(&time);
    std::ostringstream oss;
    oss << std::put_time(&tm, "%H:%M:%S");
    return oss.str();
}

inline uint64_t host_to_be64(uint64_t value) {
#ifdef _WIN32
    return _byteswap_uint64(value);
#elif defined(__linux__) || defined(__APPLE__) || defined(__FreeBSD__)
    return htobe64(value);
#else
    return ((value & 0xFFULL) << 56) |
           ((value & 0xFF00ULL) << 40) |
           ((value & 0xFF0000ULL) << 24) |
           ((value & 0xFF000000ULL) << 8) |
           ((value & 0xFF00000000ULL) >> 8) |
           ((value & 0xFF0000000000ULL) >> 24) |
           ((value & 0xFF000000000000ULL) >> 40) |
           ((value & 0xFF00000000000000ULL) >> 56);
#endif
}

inline uint64_t be64_to_host(uint64_t value) {
#ifdef _WIN32
    return _byteswap_uint64(value);
#elif defined(__linux__) || defined(__APPLE__) || defined(__FreeBSD__)
    return be64toh(value);
#else
    return host_to_be64(value);
#endif
}

inline std::string format_duration(int64_t seconds) {
    int h = seconds / 3600;
    int m = (seconds % 3600) / 60;
    int s = seconds % 60;
    std::ostringstream oss;
    if (h > 0) oss << h << "h " << m << "m " << s << "s";
    else if (m > 0) oss << m << "m " << s << "s";
    else oss << s << "s";
    return oss.str();
}

inline std::string format_hashrate(double rate) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2);
    if (rate >= 1e12) oss << (rate / 1e12) << " TH/s";
    else if (rate >= 1e9) oss << (rate / 1e9) << " GH/s";
    else if (rate >= 1e6) oss << (rate / 1e6) << " MH/s";
    else if (rate >= 1e3) oss << (rate / 1e3) << " KH/s";
    else oss << rate << " H/s";
    return oss.str();
}

// Format reward from NAU (Neptune Atomic Units) to XNT
// Conversion: 1 XNT = 4 * 10^30 NAU = 4,000,000,000,000,000,000,000,000,000,000 NAU
inline std::string format_reward_xnt(const std::string& nau_str) {
    if (nau_str.empty()) return "0 XNT";
    
    constexpr double CONVERSION_FACTOR = 4.0 * 1e30;
    
    try {
        // Parse as double (sufficient precision for display)
        // Double can represent integers exactly up to 2^53 (~9e15), but for display
        // we can accept some precision loss for very large numbers
        double nau_value = std::stod(nau_str);
        double xnt_value = nau_value / CONVERSION_FACTOR;
        
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(8);
        oss << xnt_value;
        
        std::string result = oss.str();
        
        // Remove trailing zeros after decimal point for cleaner display
        size_t dot_pos = result.find('.');
        if (dot_pos != std::string::npos) {
            size_t last_non_zero = result.find_last_not_of('0');
            if (last_non_zero != std::string::npos && last_non_zero > dot_pos) {
                result = result.substr(0, last_non_zero + 1);
            } else if (last_non_zero == dot_pos) {
                result = result.substr(0, dot_pos);
            }
        }
        
        return result + " XNT";
    } catch (const std::exception&) {
        // If parsing fails, return the original string with " NAU" suffix
        return nau_str + " NAU";
    }
}

inline std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\n\r");
    if (first == std::string::npos) return "";
    size_t last = str.find_last_not_of(" \t\n\r");
    return str.substr(first, last - first + 1);
}

#endif
