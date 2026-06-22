#include "sdxl/cuda/async_image_writer.hpp"
#include "sdxl/cuda/engine.hpp"

#include <chrono>
#include <filesystem>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {

[[nodiscard]] std::vector<std::string> split_tabs(const std::string& line) {
    std::vector<std::string> result;
    std::size_t begin = 0;
    for (;;) {
        const std::size_t end = line.find('\t', begin);
        result.push_back(line.substr(begin, end == std::string::npos ? end : end - begin));
        if (end == std::string::npos) return result;
        begin = end + 1;
    }
}

[[nodiscard]] std::string escape_json(const std::string& value) {
    std::string result;
    for (const char c : value) {
        if (c == '\\' || c == '"') result.push_back('\\');
        if (c == '\n') result += "\\n";
        else if (c == '\r') result += "\\r";
        else result.push_back(c);
    }
    return result;
}

void usage() {
    std::cout
        << "Usage: sdxl_cuda_server <model> [--memory balanced] [--precision fp8-auto] "
           "[--device 0] [--attention auto] [--arena-reserve-mib 1024] [--no-fp8-cache] [--no-preload]\n\n"
           "stdin protocol (tab-separated):\n"
           "  generate<TAB>prompt<TAB>negative<TAB>output.png<TAB>seed<TAB>width<TAB>height<TAB>steps<TAB>cfg<TAB>euler|ddim<TAB>batch<TAB>graph<TAB>profile<TAB>force_cfg\n"
           "  stats\n  flush\n  trim\n  quit\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2) { usage(); return 0; }
        sdxl::cuda::EngineOptions options;
        options.model_path = argv[1];
        options.memory_mode = sdxl::cuda::MemoryMode::Balanced;
        options.precision = sdxl::cuda::parse_precision_profile("fp8-auto");
        options.preload_components = true;
        std::size_t arena_mib = 1024;
        for (int index = 2; index < argc; ++index) {
            const std::string argument = argv[index];
            auto value = [&]() -> std::string {
                if (index + 1 >= argc) throw std::runtime_error(argument + " requires a value");
                return argv[++index];
            };
            if (argument == "--memory") options.memory_mode = sdxl::cuda::parse_memory_mode(value());
            else if (argument == "--precision") options.precision = sdxl::cuda::parse_precision_profile(value());
            else if (argument == "--device") options.device = std::stoi(value());
            else if (argument == "--attention") options.attention_backend = sdxl::cuda::parse_attention_backend(value());
            else if (argument == "--arena-reserve-mib") arena_mib = std::stoull(value());
            else if (argument == "--fp8-cache-dir") options.fp8_cache_dir = value();
            else if (argument == "--tokenizer-dir") options.tokenizer_override = value();
            else if (argument == "--no-fp8-cache") options.enable_fp8_cache = false;
            else if (argument == "--no-preload") options.preload_components = false;
            else if (argument == "--finite-checks") options.finite_checks = true;
            else if (argument == "--help") { usage(); return 0; }
            else throw std::runtime_error("unknown server option: " + argument);
        }
        options.arena_reserve_bytes = arena_mib * 1024ULL * 1024ULL;
        sdxl::cuda::SDXLEngine engine(options);
        sdxl::cuda::AsyncImageWriter writer(true, 16);
        std::cout << "{\"status\":\"ready\",\"memory\":\""
                  << sdxl::cuda::memory_mode_name(engine.memory_mode())
                  << "\",\"attention\":\"" << sdxl::cuda::attention_backend_name(options.attention_backend)
                  << "\",\"map_ms\":" << engine.checkpoint_map_milliseconds()
                  << ",\"preloaded\":" << (options.preload_components ? "true" : "false") << "}\n";
        std::cout.flush();

        std::string line;
        while (std::getline(std::cin, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.empty()) continue;
            try {
                const auto fields = split_tabs(line);
                const std::string command = fields[0];
                if (command == "quit" || command == "exit") break;
                if (command == "trim") {
                    engine.trim_temporary_memory();
                    std::cout << "{\"status\":\"trimmed\"}\n";
                    std::cout.flush();
                    continue;
                }
                if (command == "flush") {
                    writer.flush();
                    const auto records = writer.take_records();
                    double milliseconds = 0.0;
                    for (const auto& record : records) milliseconds += record.milliseconds;
                    std::cout << "{\"status\":\"flushed\",\"images\":"
                              << records.size() << ",\"image_write_ms\":"
                              << milliseconds << "}\n";
                    std::cout.flush();
                    continue;
                }
                if (command == "stats") {
                    const auto stats = engine.runtime().memory_arena_stats();
                    std::cout << "{\"live_bytes\":" << stats.live_bytes
                              << ",\"cached_bytes\":" << stats.cached_bytes
                              << ",\"driver_allocations\":" << stats.driver_allocations
                              << ",\"cache_hits\":" << stats.cache_hits
                              << ",\"slab_suballocations\":" << stats.slab_suballocations
                              << ",\"persistent_allocations\":" << stats.persistent_allocations
                              << ",\"persistent_live_bytes\":" << stats.persistent_live_bytes
                              << "}\n";
                    std::cout.flush();
                    continue;
                }
                if (command != "generate" || fields.size() < 11) {
                    throw std::runtime_error("invalid command or missing generate fields");
                }
                const std::string prompt = fields[1];
                const std::string negative = fields[2];
                const std::filesystem::path output = fields[3];
                const std::uint64_t seed = std::stoull(fields[4]);
                const std::size_t width = std::stoull(fields[5]);
                const std::size_t height = std::stoull(fields[6]);
                const std::size_t steps = std::stoull(fields[7]);
                const float cfg = std::stof(fields[8]);
                const auto scheduler = fields[9] == "ddim"
                    ? sdxl::SchedulerKind::DDIM : sdxl::SchedulerKind::EulerDiscrete;
                const std::size_t batch = std::stoull(fields[10]);
                const bool graph = fields.size() > 11 ? fields[11] != "0" : true;
                const bool profile = fields.size() > 12 ? fields[12] != "0" : false;
                const bool force_cfg = fields.size() > 13 ? fields[13] != "0" : false;
                if (batch == 0) throw std::runtime_error("batch cannot be zero");

                sdxl::cuda::GenerationRequest request;
                request.prompts.assign(batch, prompt);
                request.negative_prompts.assign(batch, negative);
                for (std::size_t item = 0; item < batch; ++item) request.seeds.push_back(seed + item);
                request.width = width;
                request.height = height;
                request.steps = steps;
                request.guidance = cfg;
                request.scheduler = scheduler;
                request.cuda_graph = graph;
                request.profile = profile;
                request.force_cfg = force_cfg;
                const auto begin = std::chrono::steady_clock::now();
                auto generated = engine.generate(request, profile ? &std::cerr : nullptr);
                const auto end = std::chrono::steady_clock::now();
                for (std::size_t item = 0; item < generated.images.size(); ++item) {
                    std::filesystem::path path = output;
                    if (generated.images.size() > 1) {
                        path = output.parent_path() /
                            (output.stem().string() + "_" + std::to_string(item) +
                             (output.extension().empty() ? ".png" : output.extension().string()));
                    }
                    writer.submit(path, std::move(generated.images[item]));
                }
                const double wall_ms = std::chrono::duration<double, std::milli>(end - begin).count();
                std::cout << "{\"status\":\"ok\",\"images\":" << batch
                          << ",\"wall_ms\":" << wall_ms
                          << ",\"graph_replay\":" << (generated.graph_replay ? "true" : "false")
                          << ",\"cfg_bypassed\":" << (generated.cfg_bypassed ? "true" : "false")
                          << ",\"fp8_cache_hit\":" << (generated.fp8_cache.hit ? "true" : "false")
                          << "}\n";
            } catch (const std::exception& error) {
                std::cout << "{\"status\":\"error\",\"message\":\""
                          << escape_json(error.what()) << "\"}\n";
            }
            std::cout.flush();
        }
        writer.flush();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
