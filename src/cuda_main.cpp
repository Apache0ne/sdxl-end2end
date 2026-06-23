#include "sdxl/cuda/async_image_writer.hpp"
#include "sdxl/cuda/engine.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace {

struct Job {
    std::string prompt;
    std::string negative;
    std::filesystem::path output = "output.png";
    std::vector<std::uint64_t> seeds;
};

struct Options {
    std::optional<std::filesystem::path> tokenizer_dir;
    std::optional<std::filesystem::path> jobs;
    std::optional<std::filesystem::path> profile_json;
    std::optional<std::filesystem::path> benchmark_csv;
    std::optional<std::filesystem::path> fp8_cache_dir;
    std::string precision = "fp8-auto";
    bool precision_explicit = false;
    std::string memory = "balanced";
    std::string attention = "auto";
    std::string sampler = "dpmpp_2m";
    std::string scheduler = "normal";
    bool sampler_explicit = false;
    bool scheduler_explicit = false;
    bool batch_explicit = false;
    bool denoise_explicit = false;
    bool noise_device_explicit = false;
    bool sampler_state_explicit = false;
    bool initial_noise_scaling_explicit = false;
    bool comfyui_parity = false;
    bool hyper_sdxl = false;
    int device = 0;
    std::size_t batch = 1;
    std::size_t arena_reserve_mib = 0;
    std::size_t arena_cache_mib = static_cast<std::size_t>(-1);
    std::size_t int8_group_size = 256;
    bool int8_strict = false;
    bool int8_tensor_cores_only = false;
    bool int8_prequantized_only = false;
    bool int8_unet_only = false;
    float guidance_rescale = 0.0F;
    float ddim_eta = 0.0F;
    bool ddim_set_alpha_to_one = false;
    float sampler_eta = 1.0F;
    float sampler_s_noise = 1.0F;
    float sampler_r = 0.5F;
    float sampler_s_churn = 0.0F;
    float sampler_s_tmin = 0.0F;
    float sampler_s_tmax = 1.0e30F;
    std::string noise_device = "cpu";
    std::string sampler_state = "fp32";
    std::string initial_noise_scaling = "comfyui";
    std::size_t training_timesteps = 1000;
    float beta_start = 0.00085F;
    float beta_end = 0.012F;
    float karras_rho = 7.0F;
    float beta_alpha = 0.6F;
    float beta_beta = 0.6F;
    float linear_quadratic_threshold = 0.025F;
    float gits_coeff = 1.20F;
    float scheduler_denoise = 1.0F;
    bool profile = false;
    bool finite_checks = false;
    bool cuda_graph = false;
    bool force_cfg = false;
    bool async_png = true;
    bool fp8_cache = true;
    bool trim_between_jobs = false;
    bool preload = false;
    bool raw_rgb = false;
    std::vector<std::string> positional;
};

[[nodiscard]] std::string value_after(int& index, int argc, char** argv, std::string_view flag) {
    if (index + 1 >= argc) throw std::runtime_error(std::string(flag) + " requires a value");
    return argv[++index];
}

[[nodiscard]] Options parse(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--tokenizer-dir") options.tokenizer_dir = value_after(index, argc, argv, argument);
        else if (argument == "--jobs") options.jobs = value_after(index, argc, argv, argument);
        else if (argument == "--profile-json") options.profile_json = value_after(index, argc, argv, argument);
        else if (argument == "--benchmark-csv") options.benchmark_csv = value_after(index, argc, argv, argument);
        else if (argument == "--fp8-cache-dir") options.fp8_cache_dir = value_after(index, argc, argv, argument);
        else if (argument == "--precision") { options.precision = value_after(index, argc, argv, argument); options.precision_explicit = true; }
        else if (argument == "--memory") options.memory = value_after(index, argc, argv, argument);
        else if (argument == "--attention") options.attention = value_after(index, argc, argv, argument);
        else if (argument == "--sampler") { options.sampler = value_after(index, argc, argv, argument); options.sampler_explicit = true; }
        else if (argument == "--hyper-sdxl") options.hyper_sdxl = true;
        else if (argument == "--comfyui-parity" || argument == "--comfyui-ksampler") options.comfyui_parity = true;
        else if (argument == "--scheduler") { options.scheduler = value_after(index, argc, argv, argument); options.scheduler_explicit = true; }
        else if (argument == "--device") options.device = std::stoi(value_after(index, argc, argv, argument));
        else if (argument == "--batch") { options.batch = std::stoull(value_after(index, argc, argv, argument)); options.batch_explicit = true; }
        else if (argument == "--arena-reserve-mib") options.arena_reserve_mib = std::stoull(value_after(index, argc, argv, argument));
        else if (argument == "--arena-cache-mib") options.arena_cache_mib = std::stoull(value_after(index, argc, argv, argument));
        else if (argument == "--int8-group-size") options.int8_group_size = std::stoull(value_after(index, argc, argv, argument));
        else if (argument == "--int8-strict") options.int8_strict = true;
        else if (argument == "--int8-tensor-cores-only") options.int8_tensor_cores_only = true;
        else if (argument == "--int8-prequantized-only") options.int8_prequantized_only = true;
        else if (argument == "--int8-unet-only") options.int8_unet_only = true;
        else if (argument == "--guidance-rescale") options.guidance_rescale = std::stof(value_after(index, argc, argv, argument));
        else if (argument == "--ddim-eta") options.ddim_eta = std::stof(value_after(index, argc, argv, argument));
        else if (argument == "--ddim-set-alpha-to-one") options.ddim_set_alpha_to_one = true;
        else if (argument == "--eta" || argument == "--sampler-eta") options.sampler_eta = std::stof(value_after(index, argc, argv, argument));
        else if (argument == "--s-noise" || argument == "--sampler-s-noise") options.sampler_s_noise = std::stof(value_after(index, argc, argv, argument));
        else if (argument == "--r" || argument == "--sampler-r") options.sampler_r = std::stof(value_after(index, argc, argv, argument));
        else if (argument == "--s-churn") options.sampler_s_churn = std::stof(value_after(index, argc, argv, argument));
        else if (argument == "--s-tmin") options.sampler_s_tmin = std::stof(value_after(index, argc, argv, argument));
        else if (argument == "--s-tmax") options.sampler_s_tmax = std::stof(value_after(index, argc, argv, argument));
        else if (argument == "--noise-device" || argument == "--latent-noise-device") { options.noise_device = value_after(index, argc, argv, argument); options.noise_device_explicit = true; }
        else if (argument == "--sampler-state") { options.sampler_state = value_after(index, argc, argv, argument); options.sampler_state_explicit = true; }
        else if (argument == "--initial-noise-scaling") { options.initial_noise_scaling = value_after(index, argc, argv, argument); options.initial_noise_scaling_explicit = true; }
        else if (argument == "--training-timesteps") options.training_timesteps = std::stoull(value_after(index, argc, argv, argument));
        else if (argument == "--beta-start") options.beta_start = std::stof(value_after(index, argc, argv, argument));
        else if (argument == "--beta-end") options.beta_end = std::stof(value_after(index, argc, argv, argument));
        else if (argument == "--karras-rho") options.karras_rho = std::stof(value_after(index, argc, argv, argument));
        else if (argument == "--alpha" || argument == "--beta-alpha") options.beta_alpha = std::stof(value_after(index, argc, argv, argument));
        else if (argument == "--beta" || argument == "--beta-beta") options.beta_beta = std::stof(value_after(index, argc, argv, argument));
        else if (argument == "--linear-quadratic-threshold") options.linear_quadratic_threshold = std::stof(value_after(index, argc, argv, argument));
        else if (argument == "--coeff" || argument == "--gits-coeff") options.gits_coeff = std::stof(value_after(index, argc, argv, argument));
        else if (argument == "--denoise" || argument == "--scheduler-denoise") { options.scheduler_denoise = std::stof(value_after(index, argc, argv, argument)); options.denoise_explicit = true; }
        else if (argument == "--profile") options.profile = true;
        else if (argument == "--finite-checks") options.finite_checks = true;
        else if (argument == "--cuda-graph") options.cuda_graph = true;
        else if (argument == "--force-cfg") options.force_cfg = true;
        else if (argument == "--sync-png") options.async_png = false;
        else if (argument == "--no-fp8-cache") options.fp8_cache = false;
        else if (argument == "--trim-between-jobs") options.trim_between_jobs = true;
        else if (argument == "--preload") options.preload = true;
        else if (argument == "--raw-rgb") options.raw_rgb = true;
        else if (argument == "--keep-resident") options.memory = "high";
        else if (argument == "--help" || argument == "-h") return options;
        else options.positional.push_back(argument);
    }
    return options;
}

void usage() {
    std::cout
        << "Usage:\n  sdxl_cuda_denoise <model> <width> <height> <steps> <cfg> <seed> "
           "<prompt> [negative] [output.png] [options]\n\n"
           "Persistent jobs:\n  sdxl_cuda_denoise <model> <width> <height> <steps> <cfg> <seed> "
           "--jobs jobs.tsv [options]\n\n"
           "Defaults: --sampler dpmpp_2m --scheduler normal, ComfyUI-compatible "
           "CPU noise + FP32 sampler state\n\n"
           "Options:\n"
           "  --memory low|balanced|high   balanced keeps FP8 UNet resident\n"
           "  --precision <profile>        fp16, FP8 profiles, int8-convrot, or int8-row\n"
           "  --int8-group-size N          ConvRot group size: 4, 16, 64, or 256\n"
           "  --int8-strict                reject upload fallback and require verified IMMA\n"
           "  --int8-tensor-cores-only     hard-error instead of using the DP4A fallback\n"
           "  --int8-prequantized-only     require I8 weights plus weight_scale metadata\n"
           "  --int8-unet-only             keep both SDXL text encoders in FP16\n"
           "  --attention <backend>         auto, cudnn-sdpa, flash-sm80, or warp-online\n"
           "  --sampler <name>              dpmpp_2m, dpmpp_sde, euler, euler_ancestral,\n"
           "                                dpmpp_2s_ancestral_cfg_pp, or ddim\n"
           "  --scheduler <name>            normal, karras, exponential, sgm_uniform, simple,\n"
           "                                ddim_uniform, ddim_trailing, beta, linear_quadratic,\n"
           "                                kl_optimal, or gits\n"
           "  --hyper-sdxl                  fixed-step Hyper-SDXL recipe: DDIM + trailing\n"
           "  --comfyui-parity              CPU noise, FP32 state, full-denoise noise scaling\n"
           "  --batch N                    duplicate a single prompt into an N-image batch\n"
           "  --cuda-graph                 capture/replay fixed-shape denoise loop\n"
           "  --force-cfg                  keep unconditional branch even when cfg <= 1\n"
           "  --profile                    full CUDA event trace and aggregate report\n"
           "  --profile-json file.json     Chrome trace output\n"
           "  --fp8-cache-dir path         packed UNet cache directory\n"
           "  --no-fp8-cache               disable packed FP8 cache\n"
           "  --arena-reserve-mib N        persistent coalescing device slab\n"
           "  --arena-cache-mib N          fallback exact-size allocation cache\n"
           "  --sync-png                   disable background PNG encoding\n"
           "  --raw-rgb                    write interleaved RGB bytes instead of PNG\n"
           "  --preload                    upload memory-mode resident components before jobs\n"
           "  --benchmark-csv file.csv     append comparable run metrics\n"
           "  --guidance-rescale F         SDXL CFG rescale, default 0\n"
           "  --ddim-eta F                 DDIM stochasticity, default 0\n"
           "  --ddim-set-alpha-to-one      use alpha=1 for DDIM final step\n"
           "  --eta F                      ancestral/SDE eta, default 1\n"
           "  --s-noise F                  sampler noise multiplier, default 1\n"
           "  --r F                        DPM++ SDE midpoint ratio, default 0.5\n"
           "  --noise-device cpu|gpu       initial latent and stochastic sampler noise, default cpu\n"
           "  --sampler-state fp32|fp16    sampler latent/history precision, default fp32\n"
           "  --initial-noise-scaling comfyui|sigma  full-denoise startup scaling\n"
           "  --s-churn F --s-tmin F --s-tmax F  Euler churn controls\n"
           "  --coeff F                    GITS coefficient (0.80..1.50, step 0.05)\n"
           "  --denoise F                  KSampler denoise fraction (0..1)\n"
           "  --alpha F --beta F           beta-scheduler shape values\n"
           "  --karras-rho F               Karras rho\n"
           "  --linear-quadratic-threshold F\n"
           "  --training-timesteps N --beta-start F --beta-end F\n"
           "  --tokenizer-dir path         override embedded SDXL tokenizers\n"
           "  --device N                   CUDA device\n\n"
           "Legacy commands may still place euler or ddim after <seed>.\n"
           "jobs.tsv: prompt<TAB>negative<TAB>output<TAB>seed<TAB>batch\n";
}

[[nodiscard]] std::vector<std::string> tabs(const std::string& line) {
    std::vector<std::string> fields;
    std::size_t begin = 0;
    for (;;) {
        const std::size_t end = line.find('\t', begin);
        fields.push_back(line.substr(begin, end == std::string::npos ? end : end - begin));
        if (end == std::string::npos) return fields;
        begin = end + 1;
    }
}

[[nodiscard]] std::vector<Job> read_jobs(const std::filesystem::path& path,
                                         std::uint64_t base_seed,
                                         std::size_t default_batch) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open jobs file: " + path.string());
    std::vector<Job> jobs;
    std::string line;
    while (std::getline(input, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line.front() == '#') continue;
        const auto fields = tabs(line);
        Job job;
        job.prompt = fields.at(0);
        if (fields.size() > 1) job.negative = fields[1];
        if (fields.size() > 2 && !fields[2].empty()) job.output = fields[2];
        const std::uint64_t seed = fields.size() > 3 && !fields[3].empty()
            ? std::stoull(fields[3]) : base_seed + jobs.size();
        const std::size_t batch = fields.size() > 4 && !fields[4].empty()
            ? std::stoull(fields[4]) : default_batch;
        if (batch == 0) throw std::runtime_error("job batch cannot be zero");
        for (std::size_t index = 0; index < batch; ++index) job.seeds.push_back(seed + index);
        jobs.push_back(std::move(job));
    }
    if (jobs.empty()) throw std::runtime_error("jobs file has no requests");
    return jobs;
}

[[nodiscard]] std::filesystem::path numbered_output(const std::filesystem::path& path,
                                                     std::size_t index,
                                                     std::size_t count) {
    if (count == 1) return path;
    const std::string extension = path.extension().empty() ? ".png" : path.extension().string();
    return path.parent_path() / (path.stem().string() + "_" + std::to_string(index) + extension);
}

[[nodiscard]] double profile_sum(const sdxl::cuda::GenerationResult& result,
                                 std::string_view label) {
    double total = 0.0;
    for (const auto& record : result.profile_records) {
        if (record.label == label) total += record.milliseconds;
    }
    return total;
}

[[nodiscard]] double profile_sum_prefix(
    const sdxl::cuda::GenerationResult& result, std::string_view prefix) {
    double total = 0.0;
    for (const auto& record : result.profile_records) {
        if (record.label.starts_with(prefix)) total += record.milliseconds;
    }
    return total;
}

void append_benchmark(const std::filesystem::path& path,
                      const sdxl::cuda::SDXLEngine& engine,
                      const sdxl::cuda::GenerationRequest& request,
                      const sdxl::cuda::GenerationResult& result,
                      double wall_ms,
                      double image_write_ms,
                      std::string_view precision) {
    const bool exists = std::filesystem::exists(path);
    std::ofstream output(path, std::ios::app);
    if (!output) throw std::runtime_error("cannot open benchmark CSV");
    if (!exists) {
        output << "width,height,steps,batch,sampler,scheduler,precision,memory,attention,cfg_bypassed,wall_ms,map_ms,clip_upload_ms,clip_encode_ms,unet_upload_ms,denoise_ms,attention_ms,cudnn_attention_ms,flash_attention_ms,warp_attention_ms,linear_ms,convolution_ms,vae_upload_ms,vae_decode_ms,rgb_download_ms,image_write_ms,graph_replay,fp8_cache_hit,temp_driver_alloc_delta,arena_hit_delta,slab_suballocation_delta,persistent_alloc_delta,persistent_live_bytes,int8_linear_calls,int8_imma_calls,int8_dp4a_fallbacks,int8_imma_plan_misses,int8_imma_execution_failures\n";
    }
    const auto delta = [](std::size_t after, std::size_t before) {
        return after >= before ? after - before : 0U;
    };
    output << request.width << ',' << request.height << ',' << request.steps << ','
           << request.prompts.size() << ','
           << sdxl::sampler_kind_name(request.sampler) << ','
           << sdxl::scheduler_kind_name(request.scheduler) << ','
           << precision << ',' << sdxl::cuda::memory_mode_name(engine.memory_mode()) << ','
           << sdxl::cuda::attention_backend_name(engine.runtime().options().attention_backend) << ','
           << (result.cfg_bypassed ? 1 : 0) << ','
           << wall_ms << ',' << engine.checkpoint_map_milliseconds() << ','
           << profile_sum(result, "stage/clip_weight_upload") << ','
           << profile_sum(result, "stage/clip_encode") << ','
           << profile_sum(result, "stage/unet_weight_upload_or_cache") << ','
           << profile_sum(result, "stage/complete_denoise_loop") +
              profile_sum(result, "stage/cuda_graph_denoise_replay") << ','
           << profile_sum_prefix(result, "ops/attention/") << ','
           << profile_sum_prefix(result, "ops/attention/cudnn-sdpa/") << ','
           << profile_sum_prefix(result, "ops/attention/flash-sm80/") << ','
           << profile_sum_prefix(result, "ops/attention/warp-online/") << ','
           << profile_sum_prefix(result, "ops/linear/") << ','
           << profile_sum_prefix(result, "ops/convolution/") << ','
           << profile_sum(result, "stage/vae_weight_upload") << ','
           << profile_sum(result, "stage/vae_decode") << ','
           << profile_sum(result, "stage/rgb8_conversion_download") << ','
           << image_write_ms << ','
           << (result.graph_replay ? 1 : 0) << ',' << (result.fp8_cache.hit ? 1 : 0) << ','
           << delta(result.arena_after.driver_allocations, result.arena_before.driver_allocations) << ','
           << delta(result.arena_after.cache_hits, result.arena_before.cache_hits) << ','
           << delta(result.arena_after.slab_suballocations, result.arena_before.slab_suballocations) << ','
           << delta(result.arena_after.persistent_allocations, result.arena_before.persistent_allocations) << ','
           << result.arena_after.persistent_live_bytes << ','
           << delta(result.int8_after.linear_calls, result.int8_before.linear_calls) << ','
           << delta(result.int8_after.cublaslt_imma_calls,
                    result.int8_before.cublaslt_imma_calls) << ','
           << delta(result.int8_after.dp4a_fallback_calls,
                    result.int8_before.dp4a_fallback_calls) << ','
           << delta(result.int8_after.tensor_core_plan_misses,
                    result.int8_before.tensor_core_plan_misses) << ','
           << delta(result.int8_after.tensor_core_execution_failures,
                    result.int8_before.tensor_core_execution_failures)
           << '\n';
}

struct PendingBenchmark {
    sdxl::cuda::GenerationRequest request;
    sdxl::cuda::GenerationResult result;
    double wall_ms = 0.0;
    std::vector<std::filesystem::path> outputs;
};

} // namespace

int main(int argc, char** argv) {
    try {
        Options options = parse(argc, argv);
#if !defined(SDXL_ENABLE_FINITE_CHECKS)
        if (options.finite_checks) throw std::runtime_error("finite checks require debug-finite build");
#endif

        std::filesystem::path model;
        std::size_t width = 0;
        std::size_t height = 0;
        std::size_t steps = 0;
        float cfg = 0.0F;
        std::uint64_t base_seed = 0;
        std::vector<Job> jobs;
        std::string sampler_value;
        std::string scheduler_value;


        const std::size_t minimum = options.jobs.has_value() ? 6U : 7U;
        if (options.positional.size() < minimum) {
            usage();
            return 0;
        }
        model = options.positional[0];
        width = std::stoull(options.positional[1]);
        height = std::stoull(options.positional[2]);
        steps = std::stoull(options.positional[3]);
        if (options.hyper_sdxl && steps != 2U && steps != 4U && steps != 8U) {
            throw std::runtime_error(
                "--hyper-sdxl is the fixed 2/4/8-step recipe; steps must be 2, 4, or 8");
        }
        cfg = std::stof(options.positional[4]);
        base_seed = std::stoull(options.positional[5]);
        sampler_value = options.hyper_sdxl ? "ddim" : options.sampler;
        scheduler_value = options.hyper_sdxl ? "ddim_trailing" : options.scheduler;
        std::size_t prompt_index = 6;
        if (!options.hyper_sdxl && !options.sampler_explicit &&
            options.positional.size() > prompt_index &&
            sdxl::is_sampler_kind_name(options.positional[prompt_index])) {
            sampler_value = options.positional[prompt_index++];
        }

        if (options.jobs.has_value()) {
            jobs = read_jobs(*options.jobs, base_seed, options.batch);
        } else {
            if (options.positional.size() <= prompt_index) {
                throw std::runtime_error("missing prompt");
            }
            Job job;
            job.prompt = options.positional[prompt_index];
            if (options.positional.size() > prompt_index + 1) {
                job.negative = options.positional[prompt_index + 1];
            }
            if (options.positional.size() > prompt_index + 2) {
                job.output = options.positional[prompt_index + 2];
            }
            for (std::size_t index = 0; index < options.batch; ++index) {
                job.seeds.push_back(base_seed + index);
            }
            jobs.push_back(std::move(job));
        }


        if (options.comfyui_parity) {
            if (!options.precision_explicit) options.precision = "fp16";
            if (!options.noise_device_explicit) options.noise_device = "cpu";
            if (!options.sampler_state_explicit) options.sampler_state = "fp32";
            if (!options.initial_noise_scaling_explicit) {
                options.initial_noise_scaling = "comfyui";
            }
        }

        const float effective_cfg = options.hyper_sdxl ? 0.0F : cfg;
        const float effective_guidance_rescale = options.hyper_sdxl
            ? 0.0F : options.guidance_rescale;
        const float effective_ddim_eta = options.hyper_sdxl ? 0.0F : options.ddim_eta;
        const sdxl::SamplerKind sampler = sdxl::parse_sampler_kind(sampler_value);
        const sdxl::SchedulerKind scheduler = sdxl::parse_scheduler_kind(scheduler_value);

        sdxl::cuda::EngineOptions engine_options;
        engine_options.model_path = model;
        engine_options.tokenizer_override = options.tokenizer_dir;
        engine_options.device = options.device;
        engine_options.memory_mode = sdxl::cuda::parse_memory_mode(options.memory);
        engine_options.precision = sdxl::cuda::parse_precision_profile(options.precision);
        if (engine_options.precision.int8) {
            engine_options.precision.int8_weights.convrot_group_size = options.int8_group_size;
            if (options.int8_strict) {
                engine_options.precision.int8_weights.strict = true;
                engine_options.precision.int8_weights.require_tensor_cores = true;
            }
            if (options.int8_tensor_cores_only) {
                engine_options.precision.int8_weights.require_tensor_cores = true;
            }
            if (options.int8_prequantized_only) {
                engine_options.precision.int8_weights.require_prequantized = true;
                engine_options.precision.int8_weights.quantize_floating_weights = false;
            }
            if (options.int8_unet_only) engine_options.precision.int8_clip = false;
        } else if (options.int8_strict || options.int8_tensor_cores_only ||
                   options.int8_prequantized_only || options.int8_unet_only ||
                   options.int8_group_size != 256U) {
            throw std::runtime_error("INT8 tuning flags require --precision int8-convrot or int8-row");
        }
        engine_options.attention_backend = sdxl::cuda::parse_attention_backend(options.attention);
        engine_options.finite_checks = options.finite_checks;
        engine_options.preload_components = options.preload;
        engine_options.enable_fp8_cache = options.fp8_cache;
        engine_options.fp8_cache_dir = options.fp8_cache_dir;
        const std::size_t default_reserve = engine_options.memory_mode == sdxl::cuda::MemoryMode::Balanced
            ? 1024U : 512U;
        engine_options.arena_reserve_bytes =
            (options.arena_reserve_mib == 0 ? default_reserve : options.arena_reserve_mib)
            * 1024ULL * 1024ULL;
        if (options.arena_cache_mib != static_cast<std::size_t>(-1)) {
            engine_options.arena_cache_limit_bytes = options.arena_cache_mib * 1024ULL * 1024ULL;
        }

        sdxl::cuda::SDXLEngine engine(engine_options);
        const auto& properties = engine.runtime().device_properties();
        std::cout << "Mapped " << engine.load_result().parameters_bound << " parameters in "
                  << engine.checkpoint_map_milliseconds() << " ms. CUDA device: "
                  << properties.name << " (SM " << properties.major << '.' << properties.minor
                  << "). Memory mode: " << sdxl::cuda::memory_mode_name(engine.memory_mode())
                  << ", precision: " << options.precision
                  << ", attention: " << sdxl::cuda::attention_backend_name(engine_options.attention_backend)
                  << ", sampler: " << sdxl::sampler_kind_name(sampler)
                  << ", scheduler: " << sdxl::scheduler_kind_name(scheduler)
                  << ", noise-device: " << sdxl::noise_device_name(
                         sdxl::parse_noise_device(options.noise_device))
                  << ", sampler-state: " << sdxl::sampler_state_precision_name(
                         sdxl::parse_sampler_state_precision(options.sampler_state))
                  << ", initial-noise: " << sdxl::initial_noise_scaling_name(
                         sdxl::parse_initial_noise_scaling(options.initial_noise_scaling));
        if (engine_options.precision.int8) {
            std::cout << ", int8-kernels: "
                      << (engine_options.precision.int8_weights.require_tensor_cores
                              ? "verified-cuBLASLt-IMMA-only"
                              : "verified-cuBLASLt-IMMA-preferred; DP4A allowed");
        }
        std::cout << ".\n";
        if (engine_options.precision.fp8 && options.fp8_cache) {
            std::cout << "FP8 cache: " << engine.fp8_cache_options().path.string() << '\n';
        }

        sdxl::cuda::AsyncImageWriter writer(options.async_png);
        std::vector<PendingBenchmark> pending_benchmarks;
        for (std::size_t job_index = 0; job_index < jobs.size(); ++job_index) {
            const Job& job = jobs[job_index];
            sdxl::cuda::GenerationRequest request;
            request.prompts.assign(job.seeds.size(), job.prompt);
            request.negative_prompts.assign(job.seeds.size(), job.negative);
            request.seeds = job.seeds;
            request.width = width;
            request.height = height;
            request.steps = steps;
            request.guidance = effective_cfg;
            request.guidance_rescale = effective_guidance_rescale;
            request.ddim_eta = effective_ddim_eta;
            request.sampler = sampler;
            request.scheduler = scheduler;
            request.sampler_config.eta = options.sampler_eta;
            request.sampler_config.s_noise = options.sampler_s_noise;
            request.sampler_config.r = options.sampler_r;
            request.sampler_config.s_churn = options.sampler_s_churn;
            request.sampler_config.s_tmin = options.sampler_s_tmin;
            request.sampler_config.s_tmax = options.sampler_s_tmax;
            request.sampler_config.noise_device = sdxl::parse_noise_device(options.noise_device);
            request.sampler_config.state_precision =
                sdxl::parse_sampler_state_precision(options.sampler_state);
            request.sampler_config.initial_noise_scaling =
                sdxl::parse_initial_noise_scaling(options.initial_noise_scaling);
            request.scheduler_config.training_timesteps = options.training_timesteps;
            request.scheduler_config.beta_start = options.beta_start;
            request.scheduler_config.beta_end = options.beta_end;
            request.scheduler_config.karras_rho = options.karras_rho;
            request.scheduler_config.beta_schedule_alpha = options.beta_alpha;
            request.scheduler_config.beta_schedule_beta = options.beta_beta;
            request.scheduler_config.linear_quadratic_threshold = options.linear_quadratic_threshold;
            request.scheduler_config.gits_coeff = options.gits_coeff;
            request.scheduler_config.denoise = options.scheduler_denoise;
            request.scheduler_config.set_alpha_to_one = options.hyper_sdxl
                ? true : options.ddim_set_alpha_to_one;
            request.cuda_graph = options.cuda_graph;
            request.force_cfg = options.hyper_sdxl ? false : options.force_cfg;
            request.profile = options.profile || options.profile_json.has_value() ||
                              options.benchmark_csv.has_value();
            if (options.profile_json.has_value()) {
                request.profile_json = jobs.size() == 1 ? *options.profile_json
                    : options.profile_json->parent_path() /
                      (options.profile_json->stem().string() + "_" + std::to_string(job_index) + ".json");
            }
            std::cout << "Job " << (job_index + 1) << '/' << jobs.size()
                      << ": batch " << request.prompts.size() << ", seed "
                      << request.seeds.front() << ", prompt: " << job.prompt << '\n';
            const auto job_arena_before = engine.runtime().memory_arena_stats();
            const auto int8_before = engine.runtime().int8_execution_stats();
            const auto begin = std::chrono::steady_clock::now();
            sdxl::cuda::GenerationResult generated = engine.generate(
                request, options.profile ? &std::cout : nullptr);
            const auto end = std::chrono::steady_clock::now();
            const double wall_ms = std::chrono::duration<double, std::milli>(end - begin).count();
            std::cout << "Stage timings: CLIP upload "
                      << profile_sum(generated, "stage/clip_weight_upload")
                      << " ms, CLIP encode "
                      << profile_sum(generated, "stage/clip_encode")
                      << " ms, UNet upload/cache "
                      << profile_sum(generated, "stage/unet_weight_upload_or_cache")
                      << " ms, denoise "
                      << (profile_sum(generated, "stage/complete_denoise_loop") +
                          profile_sum(generated, "stage/cuda_graph_denoise_replay"))
                      << " ms, VAE upload "
                      << profile_sum(generated, "stage/vae_weight_upload")
                      << " ms, VAE decode "
                      << profile_sum(generated, "stage/vae_decode") << " ms.\n";
            std::vector<std::filesystem::path> written_paths;
            written_paths.reserve(generated.images.size());
            for (std::size_t index = 0; index < generated.images.size(); ++index) {
                auto path = numbered_output(job.output, index, generated.images.size());
                if (options.raw_rgb) path.replace_extension(".rgb");
                writer.submit(path, std::move(generated.images[index]),
                    options.raw_rgb ? sdxl::cuda::ImageFileFormat::RawRGB
                                    : sdxl::cuda::ImageFileFormat::PNG);
                written_paths.push_back(path);
                std::cout << "Queued " << path.string() << '\n';
            }
            pending_benchmarks.push_back(PendingBenchmark{
                request, std::move(generated), wall_ms, std::move(written_paths)});
            if (options.trim_between_jobs) engine.trim_temporary_memory();
            const auto stats = engine.runtime().memory_arena_stats();
            const auto int8_after = engine.runtime().int8_execution_stats();
            const auto delta = [](std::size_t after, std::size_t before) {
                return after >= before ? after - before : 0U;
            };
            std::cout << "Wall " << wall_ms << " ms; CFG "
                      << (generated.cfg_bypassed ? "bypassed" : "enabled")
                      << "; temporary arena deltas: driver allocations "
                      << delta(stats.driver_allocations, job_arena_before.driver_allocations)
                      << ", exact-cache hits "
                      << delta(stats.cache_hits, job_arena_before.cache_hits)
                      << ", slab suballocations "
                      << delta(stats.slab_suballocations, job_arena_before.slab_suballocations)
                      << "; persistent weight allocations "
                      << delta(stats.persistent_allocations, job_arena_before.persistent_allocations)
                      << " (" << stats.persistent_live_bytes << " bytes resident)";
            if (engine_options.precision.int8) {
                std::cout << "; INT8 Linear calls "
                          << delta(int8_after.linear_calls, int8_before.linear_calls)
                          << ": verified cuBLASLt IMMA "
                          << delta(int8_after.cublaslt_imma_calls,
                                   int8_before.cublaslt_imma_calls)
                          << ", DP4A fallback "
                          << delta(int8_after.dp4a_fallback_calls,
                                   int8_before.dp4a_fallback_calls)
                          << ", IMMA plan misses "
                          << delta(int8_after.tensor_core_plan_misses,
                                   int8_before.tensor_core_plan_misses)
                          << ", IMMA execution failures "
                          << delta(int8_after.tensor_core_execution_failures,
                                   int8_before.tensor_core_execution_failures);
            }
            std::cout << ".\n";
        }
        writer.flush();
        const auto write_records = writer.take_records();
        std::unordered_map<std::string, double> write_times;
        for (const auto& record : write_records) {
            write_times[std::filesystem::weakly_canonical(record.path).generic_string()] +=
                record.milliseconds;
            if (options.profile) {
                std::cout << "Host image write " << record.path.string() << ": "
                          << record.milliseconds << " ms (" << record.bytes << " RGB bytes)\n";
            }
        }
        for (auto& pending : pending_benchmarks) {
            double image_write_ms = 0.0;
            for (const auto& output_path : pending.outputs) {
                const auto found = write_times.find(
                    std::filesystem::weakly_canonical(output_path).generic_string());
                if (found != write_times.end()) image_write_ms += found->second;
            }
            if (image_write_ms > 0.0) {
                float trace_end = 0.0F;
                std::size_t sequence = 0;
                for (const auto& record : pending.result.profile_records) {
                    trace_end = std::max(trace_end, record.start_milliseconds + record.milliseconds);
                    sequence = std::max(sequence, record.sequence + 1);
                }
                pending.result.profile_records.push_back(sdxl::cuda::ProfileRecord{
                    options.raw_rgb ? "host/raw_rgb_write" : "host/png_write",
                    static_cast<float>(image_write_ms), trace_end, sequence, true});
            }
            if (pending.request.profile_json.has_value()) {
                sdxl::cuda::write_chrome_trace(
                    *pending.request.profile_json, pending.result.profile_records);
            }
            if (options.benchmark_csv.has_value()) {
                append_benchmark(*options.benchmark_csv, engine, pending.request, pending.result,
                                 pending.wall_ms, image_write_ms, options.precision);
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
