#pragma once

#include "sdxl/cuda/denoise_graph.hpp"
#include "sdxl/cuda/image.hpp"
#include "sdxl/cuda/profiler.hpp"
#include "sdxl/cuda/text_encoder.hpp"
#include "sdxl/cuda/vae.hpp"
#include "sdxl/cuda/weights.hpp"
#include "sdxl/sdxl.hpp"

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace sdxl::cuda {

enum class MemoryMode : std::uint8_t { Low, Balanced, High };

struct PrecisionProfile {
    bool fp8 = true;
    FP8WeightLoadOptions weights;
    FP8BackendPreference runtime_backend = FP8BackendPreference::Auto;
    std::string canonical = "fp8-auto";
};

[[nodiscard]] PrecisionProfile parse_precision_profile(std::string value);
[[nodiscard]] MemoryMode parse_memory_mode(std::string value);
[[nodiscard]] const char* memory_mode_name(MemoryMode mode) noexcept;

struct EngineOptions {
    std::filesystem::path model_path;
    std::optional<std::filesystem::path> tokenizer_override;
    int device = 0;
    MemoryMode memory_mode = MemoryMode::Balanced;
    PrecisionProfile precision;
    AttentionBackend attention_backend = AttentionBackend::Auto;
    bool finite_checks = false;
    bool preload_components = false;
    bool enable_fp8_cache = true;
    std::optional<std::filesystem::path> fp8_cache_dir;
    std::size_t arena_reserve_bytes = 0;
    std::size_t arena_cache_limit_bytes = static_cast<std::size_t>(-1);
    std::size_t cublas_workspace_bytes = 64ULL * 1024ULL * 1024ULL;
    std::size_t cudnn_workspace_limit_bytes = 512ULL * 1024ULL * 1024ULL;
};

struct GenerationRequest {
    std::vector<std::string> prompts;
    std::vector<std::string> negative_prompts;
    std::vector<std::uint64_t> seeds;
    std::size_t width = 1024;
    std::size_t height = 1024;
    std::size_t steps = 30;
    float guidance = 5.0F;
    float guidance_rescale = 0.0F;
    float ddim_eta = 0.0F;
    SchedulerKind scheduler = SchedulerKind::EulerDiscrete;
    bool cuda_graph = false;
    bool force_cfg = false;
    bool profile = false;
    std::optional<std::filesystem::path> profile_json;
};

struct GenerationResult {
    std::vector<RGBImage> images;
    std::vector<ProfileRecord> profile_records;
    bool graph_replay = false;
    bool cfg_bypassed = false;
    FP8CacheStats fp8_cache;
    MemoryArenaStats arena_before;
    MemoryArenaStats arena_after;
};

class SDXLEngine final {
public:
    explicit SDXLEngine(EngineOptions options);
    ~SDXLEngine();

    SDXLEngine(const SDXLEngine&) = delete;
    SDXLEngine& operator=(const SDXLEngine&) = delete;

    [[nodiscard]] GenerationResult generate(const GenerationRequest& request,
                                            std::ostream* profile_output = nullptr);
    [[nodiscard]] const Runtime& runtime() const noexcept { return *runtime_; }
    [[nodiscard]] const SDXLModel& model() const noexcept { return model_; }
    [[nodiscard]] MemoryMode memory_mode() const noexcept { return options_.memory_mode; }
    [[nodiscard]] double checkpoint_map_milliseconds() const noexcept { return checkpoint_map_ms_; }
    [[nodiscard]] const LoadResult& load_result() const noexcept { return load_result_; }
    [[nodiscard]] const FP8CacheOptions& fp8_cache_options() const noexcept { return fp8_cache_; }

    void preload(bool profile = false, std::ostream* profile_output = nullptr);
    void trim_temporary_memory();

private:
    [[nodiscard]] WeightLoadStats ensure_clip(ProfileLog* profile);
    [[nodiscard]] WeightLoadStats ensure_unet(ProfileLog* profile, FP8CacheStats* cache_stats);
    [[nodiscard]] WeightLoadStats ensure_vae(ProfileLog* profile);
    void preload_resident_components();
    void apply_post_encode_policy();
    void apply_post_denoise_policy();
    void apply_post_decode_policy();
    [[nodiscard]] std::string graph_key(const GenerationRequest& request) const;

    EngineOptions options_;
    SDXLModel model_;
    LoadResult load_result_;
    double checkpoint_map_ms_ = 0.0;
    std::unique_ptr<Runtime> runtime_;
    std::unique_ptr<WeightStore> weights_;
    FP8CacheOptions fp8_cache_;
    std::unique_ptr<DenoiseGraph> graph_;
    std::string graph_key_;
};

[[nodiscard]] std::string checkpoint_fingerprint(const std::filesystem::path& path);
[[nodiscard]] std::filesystem::path default_fp8_cache_path(
    const std::filesystem::path& model_path,
    const std::optional<std::filesystem::path>& cache_dir,
    std::string_view precision,
    int sm);

} // namespace sdxl::cuda
