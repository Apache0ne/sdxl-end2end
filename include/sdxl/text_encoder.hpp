#pragma once

#include "sdxl/sdxl.hpp"
#include "sdxl/tokenizer.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

namespace sdxl {

struct FloatTensor {
    std::vector<std::size_t> shape;
    std::vector<float> values;

    [[nodiscard]] std::size_t element_count() const noexcept { return values.size(); }
    [[nodiscard]] float* data() noexcept { return values.data(); }
    [[nodiscard]] const float* data() const noexcept { return values.data(); }
};

struct TextEncoderExecutionOptions {
    std::size_t thread_count = 0;
    std::size_t linear_output_block = 32;
    bool use_attention_mask = false;
    bool check_finite_outputs = true;
    std::function<void(std::string_view component, std::size_t layer, std::size_t layer_count)> progress;
};

struct CLIPTextEncoderOutput {
    FloatTensor last_hidden_state;       // [batch, sequence, hidden]
    FloatTensor penultimate_hidden_state; // [batch, sequence, hidden], before final layer norm
    FloatTensor pooled_output;           // [batch, hidden]
    FloatTensor text_embeds;             // [batch, projection_dim], empty for CLIPTextModel
};

class CLIPTextEncoder final {
public:
    CLIPTextEncoder(const SDXLModel& model,
                    std::string component_name,
                    SDXLConfig::TextEncoder config,
                    TextEncoderExecutionOptions options = {});

    [[nodiscard]] CLIPTextEncoderOutput forward(const TokenizedBatch& tokens) const;

    [[nodiscard]] std::string_view component_name() const noexcept { return component_name_; }
    [[nodiscard]] const SDXLConfig::TextEncoder& config() const noexcept { return config_; }

private:
    const SDXLModel* model_ = nullptr;
    std::string component_name_;
    SDXLConfig::TextEncoder config_;
    TextEncoderExecutionOptions options_;
};

struct SDXLPromptConditioning {
    TokenizedBatch clip_l_tokens;
    TokenizedBatch openclip_tokens;
    FloatTensor prompt_embeds;          // [batch, 77, 2048]
    FloatTensor pooled_prompt_embeds;   // [batch, 1280]
};

struct SDXLClassifierFreeConditioning {
    SDXLPromptConditioning positive;
    SDXLPromptConditioning negative;
};

class SDXLTextConditioner final {
public:
    SDXLTextConditioner(const SDXLModel& model,
                        CLIPTokenizer tokenizer,
                        CLIPTokenizer tokenizer_2,
                        TextEncoderExecutionOptions options = {});

    [[nodiscard]] static SDXLTextConditioner builtin_sdxl(
        const SDXLModel& model,
        TextEncoderExecutionOptions options = {});

    [[nodiscard]] static SDXLTextConditioner from_model_directory(
        const SDXLModel& model,
        const std::filesystem::path& model_directory,
        TextEncoderExecutionOptions options = {});

    [[nodiscard]] SDXLPromptConditioning encode(const std::vector<std::string>& prompts) const;
    [[nodiscard]] SDXLPromptConditioning encode(const std::vector<std::string>& prompts,
                                                const std::vector<std::string>& prompts_2) const;
    [[nodiscard]] SDXLPromptConditioning encode(std::string_view prompt) const;
    [[nodiscard]] SDXLPromptConditioning encode(std::string_view prompt,
                                                std::string_view prompt_2) const;
    [[nodiscard]] SDXLClassifierFreeConditioning encode_classifier_free(
        const std::vector<std::string>& prompts,
        const std::vector<std::string>& negative_prompts) const;

private:
    const SDXLModel* model_ = nullptr;
    CLIPTokenizer tokenizer_;
    CLIPTokenizer tokenizer_2_;
    CLIPTextEncoder clip_l_;
    CLIPTextEncoder openclip_;
};

} // namespace sdxl
