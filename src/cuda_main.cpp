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
#include <sstream>
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
    std::string memory = "balanced";
    std::string attention = "auto";
    int device = 0;
    std::size_t batch = 1;
    std::size_t arena_reserve_mib = 0;
    std::size_t arena_cache_mib = static_cast<std::size_t>(-1);
    float guidance_rescale = 0.0F;
    float ddim_eta = 0.0F;
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

[[nodiscard]] std::string lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

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
        else if (argument == "--precision") options.precision = value_after(index, argc, argv, argument);
        else if (argument == "--memory") options.memory = value_after(index, argc, argv, argument);
        else if (argument == "--attention") options.attention = value_after(index, argc, argv, argument);
        else if (argument == "--device") options.device = std::stoi(value_after(index, argc, argv, argument));
        else if (argument == "--batch") options.batch = std::stoull(value_after(index, argc, argv, argument));
        else if (argument == "--arena-reserve-mib") options.arena_reserve_mib = std::stoull(value_after(index, argc, argv, argument));
        else if (argument == "--arena-cache-mib") options.arena_cache_mib = std::stoull(value_after(index, argc, argv, argument));
        else if (argument == "--guidance-rescale") options.guidance_rescale = std::stof(value_after(index, argc, argv, argument));
        else if (argument == "--ddim-eta") options.ddim_eta = std::stof(value_after(index, argc, argv, argument));
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
           "<euler|ddim> <prompt> [negative] [output.png] [options]\n\n"
           "Persistent jobs:\n  sdxl_cuda_denoise <model> <width> <height> <steps> <cfg> <seed> "
           "<euler|ddim> --jobs jobs.tsv [options]\n\n"
           "Options:\n"
           "  --memory low|balanced|high   balanced keeps FP8 UNet resident\n"
           "  --precision <profile>        fp8-auto, e4m3/e5m2 variants, or fp16\n"
           "  --attention <backend>         auto, cudnn-sdpa, flash-sm80, or warp-online\n"
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
           "  --tokenizer-dir path         override embedded SDXL tokenizers\n"
           "  --device N                   CUDA device\n\n"
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
        output << "width,height,steps,batch,scheduler,precision,memory,attention,cfg_bypassed,wall_ms,map_ms,clip_upload_ms,clip_encode_ms,unet_upload_ms,denoise_ms,attention_ms,cudnn_attention_ms,flash_attention_ms,warp_attention_ms,linear_ms,convolution_ms,vae_upload_ms,vae_decode_ms,rgb_download_ms,image_write_ms,graph_replay,fp8_cache_hit,temp_driver_alloc_delta,arena_hit_delta,slab_suballocation_delta,persistent_alloc_delta,persistent_live_bytes\n";
    }
    const auto delta = [](std::size_t after, std::size_t before) {
        return after >= before ? after - before : 0U;
    };
    output << request.width << ',' << request.height << ',' << request.steps << ','
           << request.prompts.size() << ','
           << (request.scheduler == sdxl::SchedulerKind::EulerDiscrete ? "euler" : "ddim") << ','
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
           << result.arena_after.persistent_live_bytes
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
        const Options options = parse(argc, argv);
        const std::size_t minimum = options.jobs.has_value() ? 7U : 8U;
        if (options.positional.size() < minimum) {
            usage();
            return 0;
        }
#if !defined(SDXL_ENABLE_FINITE_CHECKS)
        if (options.finite_checks) throw std::runtime_error("finite checks require debug-finite build");
#endif
        const std::filesystem::path model = options.positional[0];
        const std::size_t width = std::stoull(options.positional[1]);
        const std::size_t height = std::stoull(options.positional[2]);
        const std::size_t steps = std::stoull(options.positional[3]);
        const float cfg = std::stof(options.positional[4]);
        const std::uint64_t base_seed = std::stoull(options.positional[5]);
        const sdxl::SchedulerKind scheduler = lower(options.positional[6]) == "ddim"
            ? sdxl::SchedulerKind::DDIM : sdxl::SchedulerKind::EulerDiscrete;

        std::vector<Job> jobs;
        if (options.jobs.has_value()) {
            jobs = read_jobs(*options.jobs, base_seed, options.batch);
        } else {
            Job job;
            job.prompt = options.positional[7];
            if (options.positional.size() > 8) job.negative = options.positional[8];
            if (options.positional.size() > 9) job.output = options.positional[9];
            for (std::size_t index = 0; index < options.batch; ++index) {
                job.seeds.push_back(base_seed + index);
            }
            jobs.push_back(std::move(job));
        }

        sdxl::cuda::EngineOptions engine_options;
        engine_options.model_path = model;
        engine_options.tokenizer_override = options.tokenizer_dir;
        engine_options.device = options.device;
        engine_options.memory_mode = sdxl::cuda::parse_memory_mode(options.memory);
        engine_options.precision = sdxl::cuda::parse_precision_profile(options.precision);
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
                  << ".\n";
        if (options.fp8_cache) std::cout << "FP8 cache: " << engine.fp8_cache_options().path.string() << '\n';

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
            request.guidance = cfg;
            request.guidance_rescale = options.guidance_rescale;
            request.ddim_eta = options.ddim_eta;
            request.scheduler = scheduler;
            request.cuda_graph = options.cuda_graph;
            request.force_cfg = options.force_cfg;
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
                      << " (" << stats.persistent_live_bytes << " bytes resident).\n";
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
