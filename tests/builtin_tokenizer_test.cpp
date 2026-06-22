#include "sdxl/resources/sdxl_tokenizer_data.hpp"
#include "sdxl/tokenizer.hpp"

#include <cstdint>
#include <iostream>
#include <string_view>
#include <vector>

namespace {

bool equal_prefix(const std::vector<std::int32_t>& values,
                  const std::vector<std::int32_t>& expected) {
    return values.size() >= expected.size() &&
           std::equal(expected.begin(), expected.end(), values.begin());
}

} // namespace

int main() {
    try {
        const sdxl::CLIPTokenizer clip_l = sdxl::CLIPTokenizer::sdxl_clip_l();
        const sdxl::CLIPTokenizer openclip = sdxl::CLIPTokenizer::sdxl_openclip_big_g();

        if (clip_l.vocab_size() != 49'408U || openclip.vocab_size() != 49'408U) return 1;
        if (clip_l.bos_token_id() != 49'406 || clip_l.eos_token_id() != 49'407) return 2;
        if (openclip.bos_token_id() != 49'406 || openclip.eos_token_id() != 49'407) return 3;
        if (clip_l.pad_token_id() != 49'407) return 4;
        if (openclip.pad_token_id() != 0) return 5;
        if (sdxl::resources::sdxl_clip_merges().size() != 524'619U) return 6;
        if (sdxl::resources::sdxl_clip_merges_sha256() !=
            std::string_view{"9fd691f7c8039210e0fced15865466c65820d09b63988b0174bfe25de299051a"}) {
            return 7;
        }

        const std::vector<std::int32_t> expected{
            49'406, 320, 1'125, 539, 320, 2'368, 49'407};
        const sdxl::TokenizedBatch first = clip_l.encode("a photo of a cat");
        const sdxl::TokenizedBatch second = openclip.encode("a photo of a cat");
        if (!equal_prefix(first.input_ids, expected) || !equal_prefix(second.input_ids, expected)) return 8;
        if (first.input_ids.at(7) != 49'407 || second.input_ids.at(7) != 0) return 9;
        if (first.sequence_length != 77 || second.sequence_length != 77) return 10;

        std::cout << "embedded SDXL tokenizer test passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 11;
    }
}
