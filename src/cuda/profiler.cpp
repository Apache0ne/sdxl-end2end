#include "sdxl/cuda/profiler.hpp"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <map>
#include <ostream>
#include <sstream>
#include <utility>

namespace sdxl::cuda {
namespace {
thread_local ProfileLog* current_profile = nullptr;

[[nodiscard]] std::string json_escape(std::string_view value) {
    std::string output;
    output.reserve(value.size() + 8);
    for (const char c : value) {
        switch (c) {
        case '\\': output += "\\\\"; break;
        case '"': output += "\\\""; break;
        case '\n': output += "\\n"; break;
        case '\r': output += "\\r"; break;
        case '\t': output += "\\t"; break;
        default: output += c; break;
        }
    }
    return output;
}
} // namespace

ProfileScope::ProfileScope(ProfileLog& log, std::string label)
    : log_(&log), label_(std::move(label)), stopped_(!log.enabled()) {
#if defined(SDXL_ENABLE_PROFILING)
    if (!stopped_) {
        SDXL_CUDA_CHECK(cudaEventCreateWithFlags(&start_, cudaEventDefault));
        SDXL_CUDA_CHECK(cudaEventRecord(start_, log.runtime_->stream()));
    }
#else
    (void)log;
#endif
}

ProfileScope::~ProfileScope() { stop(); }

ProfileScope::ProfileScope(ProfileScope&& other) noexcept
    : log_(std::exchange(other.log_, nullptr)),
      label_(std::move(other.label_)),
      start_(std::exchange(other.start_, nullptr)),
      stopped_(std::exchange(other.stopped_, true)) {}

ProfileScope& ProfileScope::operator=(ProfileScope&& other) noexcept {
    if (this == &other) return *this;
    stop();
    log_ = std::exchange(other.log_, nullptr);
    label_ = std::move(other.label_);
    start_ = std::exchange(other.start_, nullptr);
    stopped_ = std::exchange(other.stopped_, true);
    return *this;
}

void ProfileScope::stop() {
#if defined(SDXL_ENABLE_PROFILING)
    if (!stopped_ && log_ != nullptr) {
        stopped_ = true;
        log_->finish_scope(std::move(label_), std::exchange(start_, nullptr));
    }
#endif
}

ProfileLog::ProfileLog(const Runtime& runtime, bool enabled)
    : runtime_(&runtime), enabled_(enabled), host_origin_(std::chrono::steady_clock::now()) {
#if defined(SDXL_ENABLE_PROFILING)
    if (enabled_) {
        SDXL_CUDA_CHECK(cudaEventCreateWithFlags(&origin_, cudaEventDefault));
        SDXL_CUDA_CHECK(cudaEventRecord(origin_, runtime_->stream()));
    }
#else
    if (enabled_) throw CudaError("CUDA profiling was not compiled into this build");
#endif
}

ProfileLog::~ProfileLog() {
    destroy_pending();
#if defined(SDXL_ENABLE_PROFILING)
    if (origin_ != nullptr) cudaEventDestroy(origin_);
#endif
}

ProfileScope ProfileLog::scope(std::string_view label) {
    return ProfileScope(*this, std::string(label));
}

void ProfileLog::finish_scope(std::string label, cudaEvent_t start) {
#if defined(SDXL_ENABLE_PROFILING)
    if (!enabled_) {
        if (start != nullptr) cudaEventDestroy(start);
        return;
    }
    cudaEvent_t stop = nullptr;
    SDXL_CUDA_CHECK(cudaEventCreateWithFlags(&stop, cudaEventDefault));
    SDXL_CUDA_CHECK(cudaEventRecord(stop, runtime_->stream()));
    pending_.push_back(PendingEvent{std::move(label), start, stop, next_sequence_++});
    resolved_ = false;
#else
    (void)label;
    (void)start;
#endif
}

void ProfileLog::add_host(std::string_view label, double milliseconds,
                          double start_milliseconds) {
    if (!enabled_) return;
    records_.push_back(ProfileRecord{
        std::string(label), static_cast<float>(milliseconds),
        static_cast<float>(start_milliseconds), next_sequence_++, true});
}

void ProfileLog::resolve() {
#if defined(SDXL_ENABLE_PROFILING)
    if (!enabled_ || resolved_) return;
    if (!pending_.empty()) runtime_->synchronize();
    for (PendingEvent& event : pending_) {
        float elapsed = 0.0F;
        float start = 0.0F;
        SDXL_CUDA_CHECK(cudaEventElapsedTime(&elapsed, event.start, event.stop));
        SDXL_CUDA_CHECK(cudaEventElapsedTime(&start, origin_, event.start));
        records_.push_back(ProfileRecord{event.label, elapsed, start, event.sequence, false});
        cudaEventDestroy(event.start);
        cudaEventDestroy(event.stop);
        event.start = nullptr;
        event.stop = nullptr;
    }
    pending_.clear();
    std::sort(records_.begin(), records_.end(), [](const ProfileRecord& a, const ProfileRecord& b) {
        return a.sequence < b.sequence;
    });
    resolved_ = true;
#endif
}

void ProfileLog::destroy_pending() noexcept {
    for (PendingEvent& event : pending_) {
        if (event.start != nullptr) cudaEventDestroy(event.start);
        if (event.stop != nullptr) cudaEventDestroy(event.stop);
    }
    pending_.clear();
}

void ProfileLog::print(std::ostream& output, std::size_t max_detailed_records) const {
    if (!enabled_ || records_.empty()) return;
    output << "\nCUDA/host profile (ordered trace):\n";
    const std::size_t count = std::min(max_detailed_records, records_.size());
    for (std::size_t index = 0; index < count; ++index) {
        const ProfileRecord& record = records_[index];
        output << "  " << std::left << std::setw(62) << record.label
               << std::right << std::fixed << std::setprecision(3)
               << record.milliseconds << " ms\n";
    }
    if (count < records_.size()) {
        output << "  ... " << (records_.size() - count)
               << " additional detailed records omitted; use --profile-json for all events.\n";
    }

    struct Aggregate { double total = 0.0; double maximum = 0.0; std::size_t count = 0; };
    std::map<std::string, Aggregate> aggregates;
    for (const ProfileRecord& record : records_) {
        Aggregate& aggregate = aggregates[record.label];
        aggregate.total += record.milliseconds;
        aggregate.maximum = std::max(aggregate.maximum, static_cast<double>(record.milliseconds));
        ++aggregate.count;
    }
    std::vector<std::pair<std::string, Aggregate>> ranked(aggregates.begin(), aggregates.end());
    std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) {
        return a.second.total > b.second.total;
    });
    output << "\nAggregated profile (largest accumulated time first):\n";
    const std::size_t aggregate_count = std::min<std::size_t>(80, ranked.size());
    for (std::size_t index = 0; index < aggregate_count; ++index) {
        const auto& [label, aggregate] = ranked[index];
        output << "  " << std::left << std::setw(52) << label
               << std::right << std::fixed << std::setprecision(3)
               << std::setw(11) << aggregate.total << " ms  x"
               << std::setw(5) << aggregate.count << "  max "
               << std::setw(10) << aggregate.maximum << " ms\n";
    }
}

void write_chrome_trace(const std::filesystem::path& path,
                        std::span<const ProfileRecord> records) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) throw CudaError("cannot create profile trace: " + path.string());
    output << "{\"traceEvents\":[";
    for (std::size_t index = 0; index < records.size(); ++index) {
        if (index != 0) output << ',';
        const ProfileRecord& record = records[index];
        const double duration_us = static_cast<double>(record.milliseconds) * 1000.0;
        const double start_us = static_cast<double>(record.start_milliseconds) * 1000.0;
        output << "{\"name\":\"" << json_escape(record.label)
               << "\",\"cat\":\"" << (record.host ? "host" : "cuda")
               << "\",\"ph\":\"X\",\"pid\":1,\"tid\":"
               << (record.host ? 2 : 1) << ",\"ts\":"
               << std::fixed << std::setprecision(3) << start_us
               << ",\"dur\":" << duration_us << '}';
    }
    output << "]}\n";
}

void ProfileLog::write_chrome_trace(const std::filesystem::path& path) const {
    if (!enabled_) return;
    sdxl::cuda::write_chrome_trace(path, records_);
}

ActiveProfileGuard::ActiveProfileGuard(ProfileLog* log) noexcept
    : previous_(current_profile) { current_profile = log; }
ActiveProfileGuard::~ActiveProfileGuard() { current_profile = previous_; }
ProfileLog* active_profile() noexcept { return current_profile; }

ProfileScope profile_scope(std::string_view label) {
    if (current_profile == nullptr || !current_profile->enabled()) return {};
    return current_profile->scope(label);
}

HostTimer::HostTimer(ProfileLog& log, std::string label)
    : log_(&log), label_(std::move(label)), begin_(std::chrono::steady_clock::now()) {
    start_milliseconds_ = std::chrono::duration<double, std::milli>(
        begin_ - log.host_origin_).count();
}
HostTimer::~HostTimer() { stop(); }
void HostTimer::stop() {
    if (log_ == nullptr) return;
    const auto end = std::chrono::steady_clock::now();
    log_->add_host(label_, std::chrono::duration<double, std::milli>(end - begin_).count(),
                   start_milliseconds_);
    log_ = nullptr;
}

CudaEventTimer::CudaEventTimer(const Runtime& runtime) : runtime_(&runtime) {
#if defined(SDXL_ENABLE_PROFILING)
    SDXL_CUDA_CHECK(cudaEventCreateWithFlags(&start_, cudaEventDefault));
    SDXL_CUDA_CHECK(cudaEventCreateWithFlags(&stop_, cudaEventDefault));
    SDXL_CUDA_CHECK(cudaEventRecord(start_, runtime.stream()));
#else
    (void)runtime;
#endif
}
CudaEventTimer::~CudaEventTimer() {
#if defined(SDXL_ENABLE_PROFILING)
    if (stop_ != nullptr) cudaEventDestroy(stop_);
    if (start_ != nullptr) cudaEventDestroy(start_);
#endif
}
float CudaEventTimer::stop() {
#if defined(SDXL_ENABLE_PROFILING)
    if (!stopped_) {
        SDXL_CUDA_CHECK(cudaEventRecord(stop_, runtime_->stream()));
        SDXL_CUDA_CHECK(cudaEventSynchronize(stop_));
        SDXL_CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms_, start_, stop_));
        stopped_ = true;
    }
    return elapsed_ms_;
#else
    throw CudaError("CUDA profiling was not compiled into this build");
#endif
}

} // namespace sdxl::cuda
