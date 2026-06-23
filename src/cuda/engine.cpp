#include "sdxl/cuda/engine.hpp"

#include <cudnn.h>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <fstream>
#include <iomanip>
#include <ostream>
#include <sstream>
#include <system_error>

namespace sdxl::cuda {
namespace {

[[nodiscard]] std::string lowercase(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

void fnv_mix(std::uint64_t& hash, std::string_view value) {
    constexpr std::uint64_t prime = 1099511628211ULL;
    for (const unsigned char byte : value) {
        hash ^= byte;
        hash *= prime;
    }
}

template <typename T>
void fnv_mix_value(std::uint64_t& hash, const T& value) {
    fnv_mix(hash, std::string_view(reinterpret_cast<const char*>(&value), sizeof(value)));
}

[[nodiscard]] std::vector<std::filesystem::path> model_files(const std::filesystem::path& path) {
    std::vector<std::filesystem::path> files;
    if (std::filesystem::is_regular_file(path)) {
        files.push_back(path);
    } else if (std::filesystem::is_directory(path)) {
        for (const auto& item : std::filesystem::recursive_directory_iterator(path)) {
            if (!item.is_regular_file()) continue;
            const std::string extension = lowercase(item.path().extension().string());
            if (extension == ".safetensors" || extension == ".json") files.push_back(item.path());
        }
        std::sort(files.begin(), files.end());
    }
    return files;
}

[[nodiscard]] std::uint64_t first_seed(const GenerationRequest& request) {
    return request.seeds.empty() ? 0ULL : request.seeds.front();
}

} // namespace

PrecisionProfile parse_precision_profile(std::string value) {
    value = lowercase(std::move(value));
    if (value == "fp8") value = "fp8-auto";
    if (value == "int8") value = "int8-convrot";
    PrecisionProfile profile;
    profile.canonical = value;
    if (value == "fp16") {
        profile.fp8 = false;
        return profile;
    }
    if (value.rfind("int8", 0) == 0) {
        profile.fp8 = false;
        profile.int8 = true;
        profile.int8_clip = value.find("unet-only") == std::string::npos;
        profile.int8_weights.enable_convrot = value.find("row") == std::string::npos;
        profile.int8_weights.strict = value.find("strict") != std::string::npos;
        profile.int8_weights.require_prequantized =
            value.find("prequantized") != std::string::npos ||
            value.find("prequant") != std::string::npos;
        profile.int8_weights.quantize_floating_weights =
            !profile.int8_weights.require_prequantized;
        const bool recognized =
            value == "int8-convrot" || value == "int8-convrot-strict" ||
            value == "int8-convrot-prequantized" ||
            value == "int8-convrot-prequantized-strict" ||
            value == "int8-row" || value == "int8-row-strict" ||
            value == "int8-convrot-unet-only" ||
            value == "int8-convrot-unet-only-strict";
        if (!recognized) throw CudaError("unknown INT8 precision profile: " + value);
        return profile;
    }
    profile.weights.format = FP8Format::Auto;
    profile.weights.scale_mode = FP8ScaleMode::TensorWide;
    profile.weights.backend = FP8Backend::Auto;
    if (value.find("e4m3") != std::string::npos) {
        profile.weights.format = FP8Format::E4M3;
    } else if (value.find("e5m2") != std::string::npos) {
        profile.weights.format = FP8Format::E5M2;
    } else if (value != "fp8-auto") {
        throw CudaError("unknown precision profile: " + value);
    }
    if (value.find("channel") != std::string::npos) {
        profile.weights.scale_mode = FP8ScaleMode::PerOutputChannel;
        profile.weights.backend = FP8Backend::WeightOnly;
        profile.runtime_backend = FP8BackendPreference::WeightOnly;
    } else if (value.find("native") != std::string::npos) {
        profile.weights.backend = FP8Backend::Native;
        profile.runtime_backend = FP8BackendPreference::NativeOnly;
    } else if (value.find("weight") != std::string::npos) {
        profile.weights.backend = FP8Backend::WeightOnly;
        profile.runtime_backend = FP8BackendPreference::WeightOnly;
    }
    return profile;
}

MemoryMode parse_memory_mode(std::string value) {
    value = lowercase(std::move(value));
    if (value == "low") return MemoryMode::Low;
    if (value == "balanced") return MemoryMode::Balanced;
    if (value == "high") return MemoryMode::High;
    throw CudaError("memory mode must be low, balanced, or high");
}

const char* memory_mode_name(MemoryMode mode) noexcept {
    switch (mode) {
    case MemoryMode::Low: return "low";
    case MemoryMode::Balanced: return "balanced";
    case MemoryMode::High: return "high";
    }
    return "unknown";
}

std::string checkpoint_fingerprint(const std::filesystem::path& path) {
    std::uint64_t hash = 1469598103934665603ULL;
    const std::filesystem::path canonical = std::filesystem::weakly_canonical(path);
    fnv_mix(hash, canonical.generic_string());
    const auto files = model_files(canonical);
    if (files.empty()) throw CudaError("model path contains no checkpoint files");
    for (const auto& file : files) {
        fnv_mix(hash, std::filesystem::relative(file, canonical.parent_path()).generic_string());
        const auto size = std::filesystem::file_size(file);
        const auto modified = std::filesystem::last_write_time(file).time_since_epoch().count();
        fnv_mix_value(hash, size);
        fnv_mix_value(hash, modified);
    }
    std::ostringstream output;
    output << std::hex << std::setfill('0') << std::setw(16) << hash;
    return output.str();
}

std::filesystem::path default_fp8_cache_path(
    const std::filesystem::path& model_path,
    const std::optional<std::filesystem::path>& cache_dir,
    std::string_view precision,
    int sm) {
    std::filesystem::path directory;
    std::string stem;
    if (cache_dir.has_value()) {
        directory = *cache_dir;
    } else if (std::filesystem::is_directory(model_path)) {
        directory = model_path / ".sdxl_cuda_cache";
    } else {
        directory = model_path.parent_path() / ".sdxl_cuda_cache";
    }
    stem = std::filesystem::is_directory(model_path)
        ? model_path.filename().string() : model_path.stem().string();
    return directory / (stem + "." + std::string(precision) + ".sm" +
                        std::to_string(sm) + ".sdxlfp8");
}

SDXLEngine::SDXLEngine(EngineOptions options) : options_(std::move(options)) {
    if (options_.model_path.empty()) throw CudaError("engine model path is empty");
    const auto map_begin = std::chrono::steady_clock::now();
    SDXLWeightLoader loader;
    load_result_ = loader.load(model_, options_.model_path);
    const auto map_end = std::chrono::steady_clock::now();
    checkpoint_map_ms_ = std::chrono::duration<double, std::milli>(map_end - map_begin).count();
    if (!load_result_.validation.ok()) throw CudaError("checkpoint validation failed");

    RuntimeOptions runtime_options;
    runtime_options.device = options_.device;
    runtime_options.cublas_workspace_bytes = options_.cublas_workspace_bytes;
    runtime_options.cudnn_workspace_limit_bytes = options_.cudnn_workspace_limit_bytes;
    runtime_options.arena_reserve_bytes = options_.arena_reserve_bytes;
    runtime_options.arena_cache_limit_bytes = options_.arena_cache_limit_bytes;
    runtime_options.strict_non_vae_fp32 = true;
    runtime_options.non_vae_accumulation = NonVAEAccumulation::Float16;
    runtime_options.fp8_backend = options_.precision.runtime_backend;
    runtime_options.attention_backend = options_.attention_backend;
    runtime_ = std::make_unique<Runtime>(runtime_options);
    weights_ = std::make_unique<WeightStore>(*runtime_, model_);

    int runtime_version = 0;
    cudaRuntimeGetVersion(&runtime_version);
    const int sm = runtime_->device_properties().major * 10 + runtime_->device_properties().minor;
    if (options_.precision.int8 && sm < 75) {
        throw CudaError("native W8A8 INT8 requires an NVIDIA Turing-class SM75 or newer GPU");
    }
    fp8_cache_.path = default_fp8_cache_path(
        options_.model_path, options_.fp8_cache_dir, options_.precision.canonical, sm);
    std::ostringstream key;
    key << "sdxl-cuda-fp8-cache-v3-kmajor|" << checkpoint_fingerprint(options_.model_path)
        << '|' << options_.precision.canonical << "|sm" << sm
        << "|cuda" << runtime_version << "|cudnn" << cudnnGetVersion();
    fp8_cache_.key = key.str();
    fp8_cache_.read = options_.enable_fp8_cache && options_.precision.fp8;
    fp8_cache_.write = options_.enable_fp8_cache && options_.precision.fp8;
    if (options_.preload_components) preload(false, nullptr);
}

SDXLEngine::~SDXLEngine() = default;

WeightLoadStats SDXLEngine::ensure_clip(ProfileLog* profile) {
    auto scope = profile != nullptr ? profile->scope("stage/clip_weight_upload") : ProfileScope{};
    if (options_.precision.int8 && options_.precision.int8_clip) {
        return weights_->load_prefixes_int8(
            {"text_encoder.", "text_encoder_2."}, options_.precision.int8_weights);
    }
    return weights_->load_prefixes({"text_encoder.", "text_encoder_2."});
}

WeightLoadStats SDXLEngine::ensure_unet(ProfileLog* profile, FP8CacheStats* cache_stats) {
    auto scope = profile != nullptr ? profile->scope("stage/unet_weight_upload_or_cache") : ProfileScope{};
    if (options_.precision.fp8) {
        return weights_->load_unet_fp8(options_.precision.weights,
            options_.enable_fp8_cache ? &fp8_cache_ : nullptr, cache_stats);
    }
    if (options_.precision.int8) {
        return weights_->load_prefixes_int8({"unet."}, options_.precision.int8_weights);
    }
    return weights_->load_prefix("unet.");
}

WeightLoadStats SDXLEngine::ensure_vae(ProfileLog* profile) {
    auto scope = profile != nullptr ? profile->scope("stage/vae_weight_upload") : ProfileScope{};
    return weights_->load_prefix("vae.", ScalarType::Float32);
}

void SDXLEngine::preload_resident_components() {
    if (options_.memory_mode == MemoryMode::High) {
        [[maybe_unused]] const WeightLoadStats clip_stats = ensure_clip(nullptr);
    }
    if (options_.memory_mode == MemoryMode::Balanced ||
        options_.memory_mode == MemoryMode::High) {
        FP8CacheStats ignored;
        [[maybe_unused]] const WeightLoadStats unet_stats = ensure_unet(nullptr, &ignored);
    }
    if (options_.memory_mode == MemoryMode::High) {
        [[maybe_unused]] const WeightLoadStats vae_stats = ensure_vae(nullptr);
    }
    runtime_->synchronize();
}

void SDXLEngine::preload(bool profile_enabled, std::ostream* profile_output) {
    ProfileLog profile(*runtime_, profile_enabled);
    ActiveProfileGuard active(profile_enabled ? &profile : nullptr);
    if (options_.memory_mode == MemoryMode::High) {
        [[maybe_unused]] const WeightLoadStats clip_stats = ensure_clip(&profile);
    }
    if (options_.memory_mode != MemoryMode::Low) {
        FP8CacheStats ignored;
        [[maybe_unused]] const WeightLoadStats unet_stats = ensure_unet(&profile, &ignored);
    }
    if (options_.memory_mode == MemoryMode::High) {
        [[maybe_unused]] const WeightLoadStats vae_stats = ensure_vae(&profile);
    }
    profile.resolve();
    if (profile_output != nullptr) profile.print(*profile_output);
}

void SDXLEngine::apply_post_encode_policy() {
    if (options_.memory_mode != MemoryMode::High) {
        weights_->unload_prefix("text_encoder.");
        weights_->unload_prefix("text_encoder_2.");
    }
}
void SDXLEngine::apply_post_denoise_policy() {
    if (options_.memory_mode == MemoryMode::Low) weights_->unload_prefix("unet.");
}
void SDXLEngine::apply_post_decode_policy() {
    if (options_.memory_mode != MemoryMode::High) weights_->unload_prefix("vae.");
}

std::string SDXLEngine::graph_key(const GenerationRequest& request) const {
    std::ostringstream key;
    key << request.width << 'x' << request.height << '|' << request.steps << '|'
        << request.prompts.size() << '|' << static_cast<int>(request.sampler) << '|'
        << static_cast<int>(request.scheduler) << '|'
        << request.guidance << '|' << request.guidance_rescale << '|' << request.ddim_eta << '|'
        << request.sampler_config.eta << '|' << request.sampler_config.s_noise << '|'
        << request.sampler_config.r << '|' << request.sampler_config.s_churn << '|'
        << request.sampler_config.s_tmin << '|' << request.sampler_config.s_tmax << '|'
        << static_cast<int>(request.sampler_config.noise_device) << '|'
        << static_cast<int>(request.sampler_config.state_precision) << '|'
        << static_cast<int>(request.sampler_config.initial_noise_scaling) << '|'
        << request.scheduler_config.training_timesteps << '|' << request.scheduler_config.beta_start << '|'
        << request.scheduler_config.beta_end << '|' << request.scheduler_config.karras_rho << '|'
        << request.scheduler_config.beta_schedule_alpha << '|' << request.scheduler_config.beta_schedule_beta << '|'
        << request.scheduler_config.linear_quadratic_threshold << '|'
        << request.scheduler_config.gits_coeff << '|' << request.scheduler_config.denoise << '|'
        << (request.scheduler_config.set_alpha_to_one ? 1 : 0) << '|'
        << (request.force_cfg ? 1 : 0) << '|'
        << ((request.force_cfg || request.guidance > 1.0F || request.sampler == SamplerKind::DPMpp2SAncestralCFGpp) ? 1 : 0);
    return key.str();
}

GenerationResult SDXLEngine::generate(const GenerationRequest& request,
                                      std::ostream* profile_output) {
    if (request.prompts.empty()) throw CudaError("generation request has no prompts");
    if (!request.negative_prompts.empty() && request.negative_prompts.size() != 1 &&
        request.negative_prompts.size() != request.prompts.size()) {
        throw CudaError("negative prompt count must be one or equal the prompt count");
    }
    if (!request.seeds.empty() && request.seeds.size() != request.prompts.size()) {
        throw CudaError("seed count must equal prompt count");
    }
    const bool stochastic_sampler = request.sampler == SamplerKind::DPMppSDE ||
        request.sampler == SamplerKind::EulerAncestral ||
        request.sampler == SamplerKind::DPMpp2SAncestralCFGpp;
    if (request.cuda_graph && stochastic_sampler && request.sampler_config.eta > 0.0F &&
        request.sampler_config.s_noise > 0.0F) {
        throw CudaError("stochastic ancestral/SDE samplers require --eta 0 or --s-noise 0 for CUDA Graph replay");
    }
    if (request.cuda_graph && options_.memory_mode == MemoryMode::Low) {
        throw CudaError("CUDA Graph replay requires balanced or high memory mode so UNet weights remain resident");
    }

#if defined(SDXL_ENABLE_PROFILING)
    // Top-level stage events are always collected: a handful of asynchronous
    // CUDA events has negligible cost and prevents unprofiled 80-second runs.
    // --profile additionally enables every low-level operator/block scope.
    constexpr bool stage_profiling = true;
#else
    const bool stage_profiling = request.profile;
#endif
    ProfileLog profile(*runtime_, stage_profiling);
    ActiveProfileGuard active(request.profile ? &profile : nullptr);
    profile.add_host("stage/checkpoint_map", checkpoint_map_ms_);
    GenerationResult result;
    result.arena_before = runtime_->memory_arena_stats();

    TextEncoderOptions text_options;
    text_options.check_finite_outputs = options_.finite_checks;
    text_options.unload_weights_after_encode = false;
    TextConditioner conditioner = options_.tokenizer_override.has_value()
        ? TextConditioner::from_model_directory(*runtime_, model_, *weights_,
                                                *options_.tokenizer_override, text_options)
        : TextConditioner::builtin_sdxl(*runtime_, model_, *weights_, text_options);
    // Match the standard diffusion-pipeline behavior: CFG is only useful above
    // 1.0. At guidance=1 the conditional prediction is already the exact
    // result, so encoding/running the unconditional branch would double CLIP
    // and UNet work for no numerical benefit. --force-cfg preserves an
    // explicit A/B/debug path.
    const bool classifier_free = request.force_cfg || request.guidance > 1.0F ||
                                 request.sampler == SamplerKind::DPMpp2SAncestralCFGpp;
    result.cfg_bypassed = !classifier_free;
    profile.add_host(classifier_free ? "config/cfg_enabled" : "config/cfg_bypassed", 0.0);
    TokenizedClassifierFree tokenized;
    {
        HostTimer timer(profile, "stage/clip_tokenize_host");
        tokenized = conditioner.tokenize_classifier_free(
            request.prompts, request.negative_prompts, classifier_free);
    }
    WeightLoadStats clip_stats;
    {
        HostTimer timer(profile, "stage/clip_weight_prepare_host");
        clip_stats = ensure_clip(&profile);
    }
    (void)clip_stats;
    ClassifierFreeConditioning conditioning;
    {
        auto scope = profile.scope("stage/clip_encode");
        conditioning = conditioner.encode_tokenized(tokenized);
    }
    apply_post_encode_policy();

    WeightLoadStats unet_stats;
    {
        HostTimer timer(profile, "stage/unet_weight_prepare_host");
        unet_stats = ensure_unet(&profile, &result.fp8_cache);
    }
    (void)unet_stats;
    DenoiseOptions denoise;
    denoise.width = request.width;
    denoise.height = request.height;
    denoise.inference_steps = request.steps;
    denoise.guidance_scale = request.guidance;
    denoise.guidance_rescale = request.guidance_rescale;
    denoise.ddim_eta = request.ddim_eta;
    denoise.seed = first_seed(request);
    denoise.batch_seeds = request.seeds;
    denoise.sampler = request.sampler;
    denoise.scheduler = request.scheduler;
    denoise.sampler_config = request.sampler_config;
    denoise.scheduler_config = request.scheduler_config;
    denoise.profile_steps = false;
    UNetOptions unet_options;
    unet_options.check_finite_output = options_.finite_checks;

    DenoiseResult denoised;
    const std::string requested_graph_key = graph_key(request);
    if (request.cuda_graph && graph_ != nullptr && graph_key_ == requested_graph_key) {
        auto scope = profile.scope("stage/cuda_graph_denoise_replay");
        if (request.seeds.empty()) denoised = graph_->launch(conditioning, denoise.seed);
        else denoised = graph_->launch(conditioning, request.seeds);
        result.graph_replay = true;
    } else {
        {
            auto scope = profile.scope("stage/complete_denoise_loop");
            Denoiser denoiser(*runtime_, model_, *weights_, unet_options);
            denoised = denoiser.denoise(conditioning, denoise);
        }
        if (request.cuda_graph) {
            ActiveProfileGuard disabled(nullptr);
            DenoiseOptions graph_options = denoise;
            graph_options.batch_seeds.clear();
            graph_ = std::make_unique<DenoiseGraph>(
                *runtime_, model_, *weights_, conditioning, graph_options, unet_options);
            graph_key_ = requested_graph_key;
        }
    }
    conditioning = {};
    apply_post_denoise_policy();

    WeightLoadStats vae_stats;
    {
        HostTimer timer(profile, "stage/vae_weight_prepare_host");
        vae_stats = ensure_vae(&profile);
    }
    (void)vae_stats;
    VAEOptions vae_options;
    vae_options.scaling_factor = model_.config().vae_scaling_factor;
    vae_options.norm_groups = static_cast<std::size_t>(model_.config().vae_norm_groups);
    vae_options.check_finite_output = options_.finite_checks;
    Tensor decoded;
    {
        auto scope = profile.scope("stage/vae_decode");
        VAE vae(*runtime_, model_, *weights_, vae_options);
        decoded = vae.decode(denoised.latents);
    }
    {
        auto scope = profile.scope("stage/rgb8_conversion_download");
        ImageConverter converter(*runtime_);
        result.images = converter.to_rgb8(decoded);
    }
    apply_post_decode_policy();

    profile.resolve();
    if (profile_output != nullptr) profile.print(*profile_output);
    result.profile_records = profile.records();
    result.arena_after = runtime_->memory_arena_stats();
    return result;
}

void SDXLEngine::trim_temporary_memory() { runtime_->trim_memory_arena(); }

} // namespace sdxl::cuda
