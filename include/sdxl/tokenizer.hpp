#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace sdxl {

struct TokenizedBatch {
    std::size_t batch_size = 0;
    std::size_t sequence_length = 0;
    std::vector<std::int32_t> input_ids;
    std::vector<std::uint8_t> attention_mask;
    std::vector<std::size_t> unpadded_lengths;
    bool any_truncated = false;

    [[nodiscard]] const std::int32_t* ids(std::size_t batch) const;
    [[nodiscard]] const std::uint8_t* mask(std::size_t batch) const;
};

struct CLIPTokenizerOptions {
    std::size_t max_length = 77;
    bool add_special_tokens = true;
    bool pad_to_max_length = true;
    bool truncate = true;
};

enum class BuiltinCLIPTokenizer {
    SDXLClipL,
    SDXLOpenCLIPBigG
};

class CLIPTokenizer final {
public:
    CLIPTokenizer() = default;
    explicit CLIPTokenizer(const std::filesystem::path& tokenizer_directory,
                           CLIPTokenizerOptions options = {});

    [[nodiscard]] static CLIPTokenizer builtin(
        BuiltinCLIPTokenizer preset,
        CLIPTokenizerOptions options = {});
    [[nodiscard]] static CLIPTokenizer sdxl_clip_l(CLIPTokenizerOptions options = {});
    [[nodiscard]] static CLIPTokenizer sdxl_openclip_big_g(CLIPTokenizerOptions options = {});

    void load(const std::filesystem::path& tokenizer_directory,
              CLIPTokenizerOptions options = {});
    void load_from_memory(std::string_view vocab_json,
                          std::string_view merges_txt,
                          std::string_view tokenizer_config_json = {},
                          CLIPTokenizerOptions options = {});
    void load_builtin(BuiltinCLIPTokenizer preset,
                      CLIPTokenizerOptions options = {});

    [[nodiscard]] TokenizedBatch encode(std::string_view text) const;
    [[nodiscard]] TokenizedBatch encode_batch(const std::vector<std::string>& texts) const;
    [[nodiscard]] std::vector<std::int32_t> encode_tokens(std::string_view text) const;

    [[nodiscard]] std::size_t vocab_size() const noexcept { return encoder_.size(); }
    [[nodiscard]] std::size_t max_length() const noexcept { return options_.max_length; }
    [[nodiscard]] std::int32_t bos_token_id() const noexcept { return bos_token_id_; }
    [[nodiscard]] std::int32_t eos_token_id() const noexcept { return eos_token_id_; }
    [[nodiscard]] std::int32_t pad_token_id() const noexcept { return pad_token_id_; }
    [[nodiscard]] std::string_view bos_token() const noexcept { return bos_token_; }
    [[nodiscard]] std::string_view eos_token() const noexcept { return eos_token_; }

private:
    struct PairHash {
        [[nodiscard]] std::size_t operator()(const std::pair<std::string, std::string>& value) const noexcept;
    };

    [[nodiscard]] std::string normalize(std::string_view text) const;
    [[nodiscard]] std::vector<std::string> pre_tokenize(std::string_view normalized) const;
    [[nodiscard]] std::vector<std::string> bpe(std::string_view token) const;
    [[nodiscard]] std::int32_t token_to_id(std::string_view token) const;

    void reset(CLIPTokenizerOptions options);
    void build_byte_encoder();
    void load_vocab(const std::filesystem::path& path);
    void load_merges(const std::filesystem::path& path);
    void load_tokenizer_config(const std::filesystem::path& path);
    void load_vocab_json(std::string_view text, std::string_view source_name);
    void load_merges_text(std::string_view text, std::string_view source_name);
    void load_tokenizer_config_json(std::string_view text, std::string_view source_name);
    void build_standard_clip_vocab_from_merges(std::string_view merges_txt);
    void resolve_special_token_ids();

    CLIPTokenizerOptions options_{};
    std::unordered_map<std::string, std::int32_t> encoder_;
    std::vector<std::string> decoder_;
    std::unordered_map<std::pair<std::string, std::string>, std::size_t, PairHash> merge_ranks_;
    std::array<std::string, 256> byte_encoder_{};

    std::string unk_token_ = "<|endoftext|>";
    std::string bos_token_ = "<|startoftext|>";
    std::string eos_token_ = "<|endoftext|>";
    std::string pad_token_ = "<|endoftext|>";
    std::int32_t unk_token_id_ = -1;
    std::int32_t bos_token_id_ = -1;
    std::int32_t eos_token_id_ = -1;
    std::int32_t pad_token_id_ = -1;

    mutable std::unordered_map<std::string, std::vector<std::string>> bpe_cache_;
};

} // namespace sdxl
