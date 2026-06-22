#include "sdxl/tokenizer.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

namespace {

void write_file(const std::filesystem::path& path, const std::string& text) {
    std::ofstream output(path, std::ios::binary);
    output << text;
}

} // namespace

int main() {
    try {
        const auto directory = std::filesystem::temp_directory_path() / "sdxl_raw_tokenizer_test";
        std::filesystem::remove_all(directory);
        std::filesystem::create_directories(directory);
        write_file(directory / "vocab.json",
                   R"({"<|startoftext|>":14,"<|endoftext|>":15,"a</w>":2,"!</w>":3,"c":4,"a":5,"t</w>":6,"cat</w>":7})");
        write_file(directory / "merges.txt", "#version: 0.2\nc a\nca t</w>\n");

        sdxl::CLIPTokenizerOptions options;
        options.max_length = 7;
        sdxl::CLIPTokenizer tokenizer(directory, options);
        const sdxl::TokenizedBatch result = tokenizer.encode("A cat!");
        const std::vector<std::int32_t> expected{14, 2, 7, 3, 15, 15, 15};
        if (result.input_ids != expected) {
            std::cerr << "Unexpected token IDs:";
            for (const auto id : result.input_ids) std::cerr << ' ' << id;
            std::cerr << '\n';
            return 1;
        }
        const std::vector<std::uint8_t> expected_mask{1, 1, 1, 1, 1, 0, 0};
        if (result.attention_mask != expected_mask) return 2;
        if (result.any_truncated) return 3;
        std::filesystem::remove_all(directory);
        std::cout << "tokenizer test passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 4;
    }
}
