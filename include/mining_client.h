#ifndef XNT_MINING_CLIENT_H
#define XNT_MINING_CLIENT_H

#include "common.cuh"

class Pow;
struct Digest;
struct PowPuzzle;

// Mining mode enumeration
enum class MiningMode {
    Solo,      // HTTP JSON-RPC (polling-based)
    Stratum    // TCP JSON-RPC (push-based)
};

// Abstract base class for mining clients
// Provides unified interface for both Solo (HTTP) and Stratum (TCP) mining
class MiningClient {
public:
    virtual ~MiningClient() = default;
    
    // Connection management
    virtual bool connect() = 0;
    virtual void disconnect() = 0;
    virtual bool is_connected() const = 0;
    
    // Job fetching
    // For Solo mode: polls for new block template
    // For Stratum mode: returns cached job from push notification
    virtual json getBlockTemplate() = 0;
    
    // Solution submission
    virtual bool submit_solution(
        const std::string& proposal_id,
        const Pow& pow_solution,
        const Digest& solution_hash,
        const json& template_obj) = 0;
    
    // Get mining mode
    virtual MiningMode get_mode() const = 0;
    
    // For stratum: wait for new job notification (push-based)
    // For solo: returns immediately (polling handled elsewhere)
    // Returns true if a new job is available
    virtual bool wait_for_job(json& job, int timeout_ms = 5000) {
        (void)job;
        (void)timeout_ms;
        return false;  // Default: not supported (solo mode)
    }
    
    // Check if this client uses push-based job notifications
    virtual bool is_push_based() const {
        return get_mode() == MiningMode::Stratum;
    }
};

// Helper function to detect mining mode from URL
inline MiningMode detect_mining_mode(const std::string& url) {
    if (url.find("stratum://") == 0 || 
        url.find("stratum+tcp://") == 0 ||
        url.find("stratum+ssl://") == 0 ||
        url.find("tcp://") == 0) {
        return MiningMode::Stratum;
    }
    return MiningMode::Solo;
}

// Helper function to get mode name
inline const char* mining_mode_name(MiningMode mode) {
    switch (mode) {
        case MiningMode::Solo: return "SOLO";
        case MiningMode::Stratum: return "STRATUM";
        default: return "UNKNOWN";
    }
}

#endif // XNT_MINING_CLIENT_H
