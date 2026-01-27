#ifndef XNT_GPU_RESOURCES_CUH
#define XNT_GPU_RESOURCES_CUH

#include "pow.cuh"

class NeptuneCudaMinerClient;
class GuesserBuffer;

enum class EventType {
    NEW_PUZZLE,
    SOLUTION_FOUND,
    STOP_MINING,
    RECONNECT,
    ERROR_EVENT
};

struct MiningEvent {
    EventType type;
    std::string puzzle_id;
    std::string data;
    std::chrono::steady_clock::time_point timestamp;
    
    MiningEvent(EventType t, const std::string& id = "", const std::string& d = "")
        : type(t), puzzle_id(id), data(d), timestamp(std::chrono::steady_clock::now()) {}
    
    MiningEvent() 
        : type(EventType::ERROR_EVENT), timestamp(std::chrono::steady_clock::now()) {}
    
    bool is_valid() const {
        return type != EventType::ERROR_EVENT || !data.empty();
    }
    
    int64_t age_ms() const {
        auto now = std::chrono::steady_clock::now();
        return std::chrono::duration_cast<std::chrono::milliseconds>(now - timestamp).count();
    }
    
    const char* type_name() const {
        switch (type) {
            case EventType::NEW_PUZZLE: return "NEW_PUZZLE";
            case EventType::SOLUTION_FOUND: return "SOLUTION_FOUND";
            case EventType::STOP_MINING: return "STOP_MINING";
            case EventType::RECONNECT: return "RECONNECT";
            case EventType::ERROR_EVENT: return "ERROR";
            default: return "UNKNOWN";
        }
    }
};

class EventHandler {
private:
    std::queue<MiningEvent> event_queue;
    mutable std::mutex queue_mutex;
    std::condition_variable queue_cv;
    std::atomic<bool> shutdown_flag{false};
    
public:
    EventHandler() = default;
    ~EventHandler() { shutdown(); }
    
    EventHandler(const EventHandler&) = delete;
    EventHandler& operator=(const EventHandler&) = delete;
    
    void postEvent(const MiningEvent& event) {
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            event_queue.push(event);
        }
        queue_cv.notify_one();
    }
    
    void postEvent(EventType type, const std::string& puzzle_id = "", const std::string& data = "") {
        postEvent(MiningEvent(type, puzzle_id, data));
    }
    
    bool waitForEvent(MiningEvent& event, std::chrono::milliseconds timeout = std::chrono::milliseconds(5000)) {
        std::unique_lock<std::mutex> lock(queue_mutex);
        
        bool result = queue_cv.wait_for(lock, timeout, [this] { 
            return !event_queue.empty() || shutdown_flag.load(); 
        });
        
        if (result && !event_queue.empty()) {
            event = event_queue.front();
            event_queue.pop();
            return true;
        }
        return false;
    }
    
    MiningEvent waitForEvent(int timeout_ms = 1000) {
        MiningEvent event;
        waitForEvent(event, std::chrono::milliseconds(timeout_ms));
        return event;
    }
    
    bool hasEvents() const {
        std::lock_guard<std::mutex> lock(queue_mutex);
        return !event_queue.empty();
    }
    
    size_t queueSize() const {
        std::lock_guard<std::mutex> lock(queue_mutex);
        return event_queue.size();
    }
    
    void clearEvents() {
        std::lock_guard<std::mutex> lock(queue_mutex);
        while (!event_queue.empty()) {
            event_queue.pop();
        }
    }
    
    void shutdown() {
        shutdown_flag.store(true);
        queue_cv.notify_all();
    }
    
    bool is_shutdown() const {
        return shutdown_flag.load();
    }
    
    void reset() {
        shutdown_flag.store(false);
        clearEvents();
    }
};

struct HashRateTracker {
    static constexpr int HISTORY_SIZE = 10;
    double rates[HISTORY_SIZE] = {0};
    int index = 0;
    int count = 0;
    
    void add(double rate) {
        rates[index] = rate;
        index = (index + 1) % HISTORY_SIZE;
        if (count < HISTORY_SIZE) count++;
    }
    
    double get_average() const {
        if (count == 0) return 0.0;
        double sum = 0.0;
        for (int i = 0; i < count; ++i) {
            sum += rates[i];
        }
        return sum / count;
    }
    
    void clear() {
        for (int i = 0; i < HISTORY_SIZE; ++i) rates[i] = 0.0;
        index = 0;
        count = 0;
    }
};

struct GpuResources {
    int gpu_id;
    std::string gpu_name;
    size_t gpu_vram_total;
    std::string gpu_uuid;
    
    std::unique_ptr<EventHandler> event_handler;
    std::unique_ptr<GuesserBuffer> buffer;
    NeptuneCudaMinerClient* client;
    
    std::atomic<bool> gpu_stop_flag{false};
    std::atomic<bool> gpu_pause_flag{false};
    std::atomic<bool> gpu_node_connected{false};
    std::atomic<bool> gpu_reconnection_in_progress{false};
    
    mutable std::mutex state_mutex;
    
    HashRateTracker hash_tracker;
    std::string current_proposal_id;
    Digest current_target;
    Digest cached_prev_block;  // Track last preprocessed prev_block to avoid re-preprocessing
    PowMastPaths cached_mast_paths;  // Track MAST paths used for preprocessing
    
    std::atomic<uint64_t> gpu_puzzle_random_start{0};
    std::atomic<uint64_t> gpu_puzzle_nonce_counter{0};
    
    std::chrono::steady_clock::time_point last_template_received_time;
    std::chrono::steady_clock::time_point session_start_time;
    
    std::atomic<uint64_t> solutions_found{0};
    std::atomic<uint64_t> total_nonces_tested{0};
    
    uint64_t optimal_max_nonces;
    
    GpuResources(int id) 
        : gpu_id(id)
        , gpu_vram_total(0)
        , client(nullptr)
        , optimal_max_nonces(1000000ULL) {
        auto now = std::chrono::steady_clock::now();
        last_template_received_time = now;
        session_start_time = now;
        gpu_pause_flag = true;
    }
    
    ~GpuResources() = default;
    
    GpuResources(const GpuResources&) = delete;
    GpuResources& operator=(const GpuResources&) = delete;
    
    int64_t get_time_since_last_job() const {
        auto now = std::chrono::steady_clock::now();
        return std::chrono::duration_cast<std::chrono::seconds>(now - last_template_received_time).count();
    }
    
    bool is_mining() const {
        return gpu_node_connected && !gpu_pause_flag && !gpu_stop_flag;
    }
    
    void update_job_received() {
        last_template_received_time = std::chrono::steady_clock::now();
    }
    
    void set_paused(bool paused) {
        gpu_pause_flag = paused;
    }
};

std::string get_gpu_uuid(int device_id);

extern std::vector<GpuResources*> g_all_gpu_resources;
extern std::mutex g_all_gpu_resources_mutex;

void broadcast_stop_to_all_gpus(int source_gpu_id, const char* reason);
GpuResources* find_gpu_resources(int gpu_id);
double get_total_hashrate();
uint64_t get_total_solutions_found();
void cleanup_gpu_memory();

#endif
