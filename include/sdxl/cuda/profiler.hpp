#pragma once

#include "sdxl/cuda/runtime.hpp"

#include <cuda_runtime_api.h>

#include <chrono>
#include <filesystem>
#include <iosfwd>
#include <memory>
#include <string>
#include <string_view>
#include <span>
#include <unordered_map>
#include <vector>

namespace sdxl::cuda {

struct ProfileRecord {
    std::string label;
    float milliseconds = 0.0F;
    float start_milliseconds = 0.0F;
    std::size_t sequence = 0;
    bool host = false;
};

class ProfileLog;

void write_chrome_trace(const std::filesystem::path& path,
                        std::span<const ProfileRecord> records);

// Records a pair of CUDA events without synchronizing. All event pairs are
// resolved together when ProfileLog::resolve/print is called, so detailed
// profiling does not serialize every kernel launch.
class ProfileScope final {
public:
    ProfileScope() = default;
    ProfileScope(ProfileLog& log, std::string label);
    ~ProfileScope();

    ProfileScope(const ProfileScope&) = delete;
    ProfileScope& operator=(const ProfileScope&) = delete;
    ProfileScope(ProfileScope&& other) noexcept;
    ProfileScope& operator=(ProfileScope&& other) noexcept;

    void stop();

private:
    ProfileLog* log_ = nullptr;
    std::string label_;
    cudaEvent_t start_ = nullptr;
    bool stopped_ = true;
};

class ProfileLog final {
public:
    ProfileLog(const Runtime& runtime, bool enabled = false);
    ~ProfileLog();

    ProfileLog(const ProfileLog&) = delete;
    ProfileLog& operator=(const ProfileLog&) = delete;

    [[nodiscard]] bool enabled() const noexcept { return enabled_; }
    [[nodiscard]] ProfileScope scope(std::string_view label);
    void add_host(std::string_view label, double milliseconds, double start_milliseconds = 0.0);
    void resolve();
    void print(std::ostream& output, std::size_t max_detailed_records = 400) const;
    void write_chrome_trace(const std::filesystem::path& path) const;
    [[nodiscard]] const std::vector<ProfileRecord>& records() const noexcept { return records_; }

private:
    friend class ProfileScope;
    friend class HostTimer;
    struct PendingEvent {
        std::string label;
        cudaEvent_t start = nullptr;
        cudaEvent_t stop = nullptr;
        std::size_t sequence = 0;
    };

    void finish_scope(std::string label, cudaEvent_t start);
    void destroy_pending() noexcept;

    const Runtime* runtime_ = nullptr;
    bool enabled_ = false;
    bool resolved_ = false;
    std::size_t next_sequence_ = 0;
    cudaEvent_t origin_ = nullptr;
    std::chrono::steady_clock::time_point host_origin_{};
    std::vector<PendingEvent> pending_;
    std::vector<ProfileRecord> records_;
};

// Installs a profile log for low-level Ops calls on the current host thread.
// The inference engine is single-dispatch-thread by design; CUDA work remains
// asynchronous on the runtime stream.
class ActiveProfileGuard final {
public:
    explicit ActiveProfileGuard(ProfileLog* log) noexcept;
    ~ActiveProfileGuard();
    ActiveProfileGuard(const ActiveProfileGuard&) = delete;
    ActiveProfileGuard& operator=(const ActiveProfileGuard&) = delete;
private:
    ProfileLog* previous_ = nullptr;
};

[[nodiscard]] ProfileLog* active_profile() noexcept;
[[nodiscard]] ProfileScope profile_scope(std::string_view label);

class HostTimer final {
public:
    HostTimer(ProfileLog& log, std::string label);
    ~HostTimer();
    HostTimer(const HostTimer&) = delete;
    HostTimer& operator=(const HostTimer&) = delete;
    void stop();
private:
    ProfileLog* log_ = nullptr;
    std::string label_;
    std::chrono::steady_clock::time_point begin_{};
    double start_milliseconds_ = 0.0;
};

// Compatibility helper for single outer CUDA measurements.
class CudaEventTimer final {
public:
    explicit CudaEventTimer(const Runtime& runtime);
    ~CudaEventTimer();
    CudaEventTimer(const CudaEventTimer&) = delete;
    CudaEventTimer& operator=(const CudaEventTimer&) = delete;
    [[nodiscard]] float stop();
private:
    const Runtime* runtime_ = nullptr;
    cudaEvent_t start_ = nullptr;
    cudaEvent_t stop_ = nullptr;
    bool stopped_ = false;
    float elapsed_ms_ = 0.0F;
};

} // namespace sdxl::cuda
