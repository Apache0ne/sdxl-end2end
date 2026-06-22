#pragma once

#include "sdxl/cuda/ops.hpp"
#include "sdxl/tokenizer.hpp"

#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace sdxl::cuda {

struct TextEncoderOptions {
    bool check_finite_outputs = false;
    bool unload_weights_after_encode = true;
};

struct TextEncoderOutput {
    Tensor last_hidden_state;
    Tensor penultimate_hidden_state;
    Tensor pooled_output;
    Tensor text_embeds;
};

class TextEncoder final {
public:
    TextEncoder(const Runtime& runtime,
                const SDXLModel& model,
                const WeightStore& weights,
                std::string component,
                SDXLConfig::TextEncoder config,
                TextEncoderOptions options = {});

    [[nodiscard]] TextEncoderOutput forward(const TokenizedBatch& tokens) const;

private:
    const Runtime* runtime_ = nullptr;
    const SDXLModel* model_ = nullptr;
    const WeightStore* weights_ = nullptr;
    std::string component_;
    SDXLConfig::TextEncoder config_;
    TextEncoderOptions options_;
};

struct TokenizedClassifierFree {
    TokenizedBatch clip_l;
    TokenizedBatch openclip;
    std::size_t batch_size = 0;
    bool classifier_free = true;
};

struct ClassifierFreeConditioning {
    Tensor prompt_embeds;        // CFG: [2B,77,2048]; bypass: [B,77,2048]
    Tensor pooled_text_embeds;   // CFG: [2B,1280]; bypass: [B,1280]
    std::size_t batch_size = 0;
    bool classifier_free = true;
};

class TextConditioner final {
public:
    TextConditioner(const Runtime& runtime,
                    const SDXLModel& model,
                    WeightStore& weights,
                    CLIPTokenizer tokenizer,
                    CLIPTokenizer tokenizer_2,
                    TextEncoderOptions options = {});

    [[nodiscard]] static TextConditioner builtin_sdxl(
        const Runtime& runtime,
        const SDXLModel& model,
        WeightStore& weights,
        TextEncoderOptions options = {});

    [[nodiscard]] static TextConditioner from_model_directory(
        const Runtime& runtime,
        const SDXLModel& model,
        WeightStore& weights,
        const std::filesystem::path& model_directory,
        TextEncoderOptions options = {});

    [[nodiscard]] TokenizedClassifierFree tokenize_classifier_free(
        const std::vector<std::string>& prompts,
        const std::vector<std::string>& negative_prompts,
        bool classifier_free = true) const;

    [[nodiscard]] ClassifierFreeConditioning encode_tokenized(
        const TokenizedClassifierFree& tokens);

    [[nodiscard]] ClassifierFreeConditioning encode_classifier_free(
        const std::vector<std::string>& prompts,
        const std::vector<std::string>& negative_prompts,
        bool classifier_free = true);

private:
    const Runtime* runtime_ = nullptr;
    const SDXLModel* model_ = nullptr;
    WeightStore* weights_ = nullptr;
    CLIPTokenizer tokenizer_;
    CLIPTokenizer tokenizer_2_;
    TextEncoderOptions options_;
};

} // namespace sdxl::cuda
