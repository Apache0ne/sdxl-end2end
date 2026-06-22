#include "sdxl/sdxl.hpp"

#include <algorithm>
#include <iostream>
#include <string>

namespace {

const char* layout_name(sdxl::CheckpointLayout layout) {
    switch (layout) {
    case sdxl::CheckpointLayout::DiffusersDirectory: return "Diffusers directory";
    case sdxl::CheckpointLayout::OriginalSingleFile: return "original single-file checkpoint";
    case sdxl::CheckpointLayout::Unknown: return "unknown";
    }
    return "unknown";
}

} // namespace

int main(int argc, char** argv) {
    try {
        sdxl::SDXLModel model;
        std::cout << "SDXL Base 1.0 architecture created\n";
        std::cout << "Expected parameter tensors: " << model.graph().parameter_index().size() << "\n";

        if (argc < 2) {
            std::cout << "Usage: sdxl_inspect <diffusers-model-directory|checkpoint.safetensors> [--non-strict]\n";
            return 0;
        }

        sdxl::LoadOptions options;
        if (argc >= 3 && std::string(argv[2]) == "--non-strict") options.strict = false;
        sdxl::SDXLWeightLoader loader(options);
        const sdxl::LoadResult result = loader.load(model, argv[1]);

        std::cout << "Layout: " << layout_name(result.layout) << "\n";
        std::cout << "Mapped files: " << result.files_mapped << "\n";
        std::cout << "Discovered tensors: " << result.tensors_discovered << "\n";
        std::cout << "Bound architecture parameters: " << result.parameters_bound << "\n";
        std::cout << "Missing: " << result.validation.missing << "\n";
        std::cout << "Shape mismatches: " << result.validation.shape_mismatches << "\n";

        for (const auto& note : result.notes) std::cout << "Note: " << note << "\n";

        const std::size_t issue_limit = std::min<std::size_t>(result.validation.issues.size(), 40);
        for (std::size_t i = 0; i < issue_limit; ++i) {
            const auto& issue = result.validation.issues[i];
            std::cout << "Issue: " << issue.logical_name << " - " << issue.detail << "\n";
        }
        if (result.validation.issues.size() > issue_limit) {
            std::cout << "... " << (result.validation.issues.size() - issue_limit)
                      << " additional issues omitted\n";
        }
        return result.validation.ok() ? 0 : 2;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << "\n";
        return 1;
    }
}
