#include "sdxl/cuda/text_encoder.hpp"
#include "sdxl/cuda/profiler.hpp"

#include <algorithm>
#include <utility>

namespace sdxl::cuda {
namespace {

[[nodiscard]] Tensor ids_to_device(const Runtime& runtime, const TokenizedBatch& tokens) {
    return Tensor::from_host_i32(runtime,
        {tokens.batch_size, tokens.sequence_length}, tokens.input_ids);
}

[[nodiscard]] std::vector<std::string> normalize_negative_batch(
    const std::vector<std::string>& prompts,
    const std::vector<std::string>& negative_prompts) {
    if (prompts.empty()) throw CudaError("prompt batch cannot be empty");
    std::vector<std::string> negatives = negative_prompts;
    if (negatives.empty()) negatives.assign(prompts.size(), std::string{});
    if (negatives.size() == 1 && prompts.size() > 1) negatives.assign(prompts.size(), negatives.front());
    if (negatives.size() != prompts.size()) {
        throw CudaError("negative prompt batch must have one item or match the prompt batch");
    }
    return negatives;
}

} // namespace

TextEncoder::TextEncoder(const Runtime& runtime,
                         const SDXLModel& model,
                         const WeightStore& weights,
                         std::string component,
                         SDXLConfig::TextEncoder config,
                         TextEncoderOptions options)
    : runtime_(&runtime),
      model_(&model),
      weights_(&weights),
      component_(std::move(component)),
      config_(std::move(config)),
      options_(options) {}

TextEncoderOutput TextEncoder::forward(const TokenizedBatch& tokens) const {
    auto forward_profile = profile_scope("clip/" + component_ + "/forward");
    if (runtime_ == nullptr || model_ == nullptr || weights_ == nullptr) {
        throw CudaError("CUDA text encoder is not initialized");
    }
    if (tokens.batch_size == 0 || tokens.sequence_length == 0 ||
        tokens.sequence_length > config_.max_positions) {
        throw CudaError("invalid CUDA CLIP token batch");
    }
    Ops ops(*runtime_);
    const std::string prefix = component_ + ".text_model.";
    const Tensor token_ids = ids_to_device(*runtime_, tokens);
    Tensor hidden = ops.embedding(
        token_ids,
        weights_->get(prefix + "embeddings.token_embedding.weight"),
        weights_->get(prefix + "embeddings.position_embedding.weight"));

    Tensor penultimate;
    if (config_.layers == 1) penultimate = hidden;

    for (std::size_t layer = 0; layer < static_cast<std::size_t>(config_.layers); ++layer) {
        auto layer_profile = profile_scope("clip/" + component_ + "/layer/" + std::to_string(layer));
        const std::string layer_prefix = prefix + "encoder.layers." + std::to_string(layer) + ".";
        Tensor normalized = ops.layer_norm(
            hidden,
            weights_->get(layer_prefix + "layer_norm1.weight"),
            weights_->get(layer_prefix + "layer_norm1.bias"),
            1.0e-5F);
        Tensor query = ops.linear(
            normalized,
            weights_->get(layer_prefix + "self_attn.q_proj.weight"),
            &weights_->get(layer_prefix + "self_attn.q_proj.bias"));
        Tensor key = ops.linear(
            normalized,
            weights_->get(layer_prefix + "self_attn.k_proj.weight"),
            &weights_->get(layer_prefix + "self_attn.k_proj.bias"));
        Tensor value = ops.linear(
            normalized,
            weights_->get(layer_prefix + "self_attn.v_proj.weight"),
            &weights_->get(layer_prefix + "self_attn.v_proj.bias"));
        Tensor attention = ops.attention(query, key, value,
                                         static_cast<std::size_t>(config_.heads), true, nullptr);
        Tensor attention_output = ops.linear(
            attention,
            weights_->get(layer_prefix + "self_attn.out_proj.weight"),
            &weights_->get(layer_prefix + "self_attn.out_proj.bias"));
        ops.add_in_place(hidden, attention_output);

        normalized = ops.layer_norm(
            hidden,
            weights_->get(layer_prefix + "layer_norm2.weight"),
            weights_->get(layer_prefix + "layer_norm2.bias"),
            1.0e-5F);
        Tensor intermediate = ops.linear_activation(
            normalized,
            weights_->get(layer_prefix + "mlp.fc1.weight"),
            weights_->get(layer_prefix + "mlp.fc1.bias"),
            config_.activation == "quick_gelu"
                ? LinearActivation::QuickGELU : LinearActivation::GELU);
        Tensor mlp_output = ops.linear(
            intermediate,
            weights_->get(layer_prefix + "mlp.fc2.weight"),
            &weights_->get(layer_prefix + "mlp.fc2.bias"));
        ops.add_in_place(hidden, mlp_output);

        if (layer + 2 == static_cast<std::size_t>(config_.layers)) penultimate = hidden;
    }

    if (!penultimate.defined()) throw CudaError("CUDA CLIP penultimate state was not captured");
    Tensor last_hidden = ops.layer_norm(
        hidden,
        weights_->get(prefix + "final_layer_norm.weight"),
        weights_->get(prefix + "final_layer_norm.bias"),
        1.0e-5F);
    Tensor pooled = ops.pool_eos(last_hidden, token_ids);
    Tensor projected;
    if (config_.with_projection) {
        projected = ops.linear(pooled, weights_->get(component_ + ".text_projection.weight"), nullptr);
    }

    if (options_.check_finite_outputs) {
        ops.check_finite(last_hidden, component_ + " last hidden state");
        ops.check_finite(penultimate, component_ + " penultimate hidden state");
        ops.check_finite(pooled, component_ + " pooled output");
        if (projected.defined()) ops.check_finite(projected, component_ + " projected output");
    }
    return TextEncoderOutput{
        std::move(last_hidden), std::move(penultimate), std::move(pooled), std::move(projected)};
}

TextConditioner::TextConditioner(const Runtime& runtime,
                                 const SDXLModel& model,
                                 WeightStore& weights,
                                 CLIPTokenizer tokenizer,
                                 CLIPTokenizer tokenizer_2,
                                 TextEncoderOptions options)
    : runtime_(&runtime),
      model_(&model),
      weights_(&weights),
      tokenizer_(std::move(tokenizer)),
      tokenizer_2_(std::move(tokenizer_2)),
      options_(options) {}

TextConditioner TextConditioner::builtin_sdxl(
    const Runtime& runtime,
    const SDXLModel& model,
    WeightStore& weights,
    TextEncoderOptions options) {
    return TextConditioner(runtime, model, weights,
                           CLIPTokenizer::sdxl_clip_l(),
                           CLIPTokenizer::sdxl_openclip_big_g(),
                           options);
}

TextConditioner TextConditioner::from_model_directory(
    const Runtime& runtime,
    const SDXLModel& model,
    WeightStore& weights,
    const std::filesystem::path& model_directory,
    TextEncoderOptions options) {
    const std::filesystem::path first = model_directory / "tokenizer";
    const std::filesystem::path second_candidate = model_directory / "tokenizer_2";
    const std::filesystem::path second = std::filesystem::is_directory(second_candidate)
                                             ? second_candidate
                                             : first;
    return TextConditioner(runtime, model, weights,
                           CLIPTokenizer(first), CLIPTokenizer(second), options);
}

TokenizedClassifierFree TextConditioner::tokenize_classifier_free(
    const std::vector<std::string>& prompts,
    const std::vector<std::string>& negative_prompts,
    bool classifier_free) const {
    if (runtime_ == nullptr || model_ == nullptr || weights_ == nullptr) {
        throw CudaError("CUDA text conditioner is not initialized");
    }
    std::vector<std::string> combined;
    if (classifier_free) {
        const std::vector<std::string> negatives =
            normalize_negative_batch(prompts, negative_prompts);
        combined.reserve(prompts.size() * 2);
        combined.insert(combined.end(), negatives.begin(), negatives.end());
        combined.insert(combined.end(), prompts.begin(), prompts.end());
    } else {
        combined = prompts;
    }

    TokenizedClassifierFree result;
    result.clip_l = tokenizer_.encode_batch(combined);
    result.openclip = tokenizer_2_.encode_batch(combined);
    result.batch_size = prompts.size();
    result.classifier_free = classifier_free;
    if (result.clip_l.batch_size != result.openclip.batch_size ||
        result.clip_l.sequence_length != result.openclip.sequence_length) {
        throw CudaError("the two CLIP tokenizers produced incompatible shapes");
    }
    return result;
}

ClassifierFreeConditioning TextConditioner::encode_tokenized(
    const TokenizedClassifierFree& tokens) {
    if (runtime_ == nullptr || model_ == nullptr || weights_ == nullptr) {
        throw CudaError("CUDA text conditioner is not initialized");
    }
    const std::size_t conditioning_multiplier = tokens.classifier_free ? 2 : 1;
    if (tokens.batch_size == 0 ||
        tokens.clip_l.batch_size != tokens.batch_size * conditioning_multiplier ||
        tokens.openclip.batch_size != tokens.batch_size * conditioning_multiplier) {
        throw CudaError("invalid tokenized text-conditioning batch");
    }
    if (!weights_->contains("text_encoder.text_model.embeddings.token_embedding.weight") ||
        !weights_->contains("text_encoder_2.text_model.embeddings.token_embedding.weight")) {
        throw CudaError("required CUDA text-encoder weights are not resident");
    }
    TextEncoder clip_l(*runtime_, *model_, *weights_, "text_encoder",
                       model_->config().clip_l, options_);
    TextEncoder openclip(*runtime_, *model_, *weights_, "text_encoder_2",
                         model_->config().openclip_big_g, options_);
    TextEncoderOutput first = clip_l.forward(tokens.clip_l);
    TextEncoderOutput second = openclip.forward(tokens.openclip);
    if (!second.text_embeds.defined()) {
        throw CudaError("the second CUDA text encoder did not produce projected embeddings");
    }
    Ops ops(*runtime_);
    Tensor prompt_embeds = ops.concat_last_dim(
        first.penultimate_hidden_state, second.penultimate_hidden_state);
    Tensor pooled = second.text_embeds;

    if (options_.unload_weights_after_encode) {
        weights_->unload_prefix("text_encoder.");
        weights_->unload_prefix("text_encoder_2.");
    }
    return ClassifierFreeConditioning{
        std::move(prompt_embeds), std::move(pooled), tokens.batch_size,
        tokens.classifier_free};
}

ClassifierFreeConditioning TextConditioner::encode_classifier_free(
    const std::vector<std::string>& prompts,
    const std::vector<std::string>& negative_prompts,
    bool classifier_free) {
    const TokenizedClassifierFree tokens =
        tokenize_classifier_free(prompts, negative_prompts, classifier_free);
    [[maybe_unused]] const WeightLoadStats text_weight_stats =
        weights_->load_prefixes({"text_encoder.", "text_encoder_2."});
    return encode_tokenized(tokens);
}

} // namespace sdxl::cuda
