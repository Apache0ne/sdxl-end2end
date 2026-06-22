#include "sdxl/tokenizer.hpp"

#include "sdxl/json.hpp"
#include "sdxl/safetensors.hpp"
#include "sdxl/resources/sdxl_tokenizer_data.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <climits>
#include <cstdint>
#include <fstream>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

namespace sdxl {
namespace {

[[nodiscard]] std::string read_text_file(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw Error("cannot open tokenizer file: " + path.string());
    input.seekg(0, std::ios::end);
    const std::streamoff size = input.tellg();
    if (size < 0) throw Error("cannot determine tokenizer file size: " + path.string());
    input.seekg(0, std::ios::beg);
    std::string text(static_cast<std::size_t>(size), '\0');
    if (!text.empty()) input.read(text.data(), size);
    if (!input && !text.empty()) throw Error("cannot read tokenizer file: " + path.string());
    return text;
}

void append_utf8(std::string& output, std::uint32_t codepoint) {
    if (codepoint <= 0x7FU) {
        output.push_back(static_cast<char>(codepoint));
    } else if (codepoint <= 0x7FFU) {
        output.push_back(static_cast<char>(0xC0U | (codepoint >> 6U)));
        output.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
    } else if (codepoint <= 0xFFFFU) {
        output.push_back(static_cast<char>(0xE0U | (codepoint >> 12U)));
        output.push_back(static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3FU)));
        output.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
    } else if (codepoint <= 0x10FFFFU) {
        output.push_back(static_cast<char>(0xF0U | (codepoint >> 18U)));
        output.push_back(static_cast<char>(0x80U | ((codepoint >> 12U) & 0x3FU)));
        output.push_back(static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3FU)));
        output.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
    } else {
        append_utf8(output, 0xFFFDU);
    }
}

struct DecodedCodepoint {
    std::uint32_t value = 0xFFFDU;
    std::size_t bytes = 1;
};

[[nodiscard]] DecodedCodepoint decode_utf8(std::string_view text, std::size_t position) noexcept {
    if (position >= text.size()) return {};
    const auto first = static_cast<unsigned char>(text[position]);
    if (first < 0x80U) return {first, 1};

    std::size_t length = 0;
    std::uint32_t value = 0;
    if ((first & 0xE0U) == 0xC0U) {
        length = 2;
        value = first & 0x1FU;
    } else if ((first & 0xF0U) == 0xE0U) {
        length = 3;
        value = first & 0x0FU;
    } else if ((first & 0xF8U) == 0xF0U) {
        length = 4;
        value = first & 0x07U;
    } else {
        return {};
    }
    if (position + length > text.size()) return {};
    for (std::size_t i = 1; i < length; ++i) {
        const auto next = static_cast<unsigned char>(text[position + i]);
        if ((next & 0xC0U) != 0x80U) return {};
        value = static_cast<std::uint32_t>((value << 6U) | (next & 0x3FU));
    }
    const bool overlong = (length == 2 && value < 0x80U) ||
                          (length == 3 && value < 0x800U) ||
                          (length == 4 && value < 0x10000U);
    if (overlong || value > 0x10FFFFU || (value >= 0xD800U && value <= 0xDFFFU)) return {};
    return {value, length};
}

[[nodiscard]] bool is_unicode_space(std::uint32_t cp) noexcept {
    if (cp <= 0x7FU) return cp == 0x20U || (cp >= 0x09U && cp <= 0x0DU);
    return cp == 0x0085U || cp == 0x00A0U || cp == 0x1680U ||
           (cp >= 0x2000U && cp <= 0x200AU) || cp == 0x2028U ||
           cp == 0x2029U || cp == 0x202FU || cp == 0x205FU || cp == 0x3000U;
}

#ifdef _WIN32
[[nodiscard]] std::wstring utf8_to_wide(std::string_view text) {
    if (text.empty()) return {};
    if (text.size() > static_cast<std::size_t>(INT_MAX)) throw Error("text is too long for Windows Unicode conversion");
    const int input_size = static_cast<int>(text.size());
    const int count = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, text.data(), input_size, nullptr, 0);
    if (count <= 0) throw Error("prompt is not valid UTF-8");
    std::wstring output(static_cast<std::size_t>(count), L'\0');
    if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, text.data(), input_size,
                            output.data(), count) != count) {
        throw Error("UTF-8 to UTF-16 conversion failed");
    }
    return output;
}

[[nodiscard]] std::string wide_to_utf8(std::wstring_view text) {
    if (text.empty()) return {};
    if (text.size() > static_cast<std::size_t>(INT_MAX)) throw Error("text is too long for Windows Unicode conversion");
    const int input_size = static_cast<int>(text.size());
    const int count = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, text.data(), input_size,
                                          nullptr, 0, nullptr, nullptr);
    if (count <= 0) throw Error("UTF-16 to UTF-8 conversion failed");
    std::string output(static_cast<std::size_t>(count), '\0');
    if (WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, text.data(), input_size,
                            output.data(), count, nullptr, nullptr) != count) {
        throw Error("UTF-16 to UTF-8 conversion failed");
    }
    return output;
}

[[nodiscard]] std::wstring normalize_lower_windows(std::wstring_view input) {
    if (input.empty()) return {};
    if (input.size() > static_cast<std::size_t>(INT_MAX)) throw Error("prompt is too long");
    const int input_size = static_cast<int>(input.size());
    int normalized_size = NormalizeString(NormalizationC, input.data(), input_size, nullptr, 0);
    if (normalized_size <= 0) throw Error("Windows NFC normalization failed");
    std::wstring normalized(static_cast<std::size_t>(normalized_size), L'\0');
    normalized_size = NormalizeString(NormalizationC, input.data(), input_size,
                                      normalized.data(), normalized_size);
    if (normalized_size <= 0) throw Error("Windows NFC normalization failed");
    normalized.resize(static_cast<std::size_t>(normalized_size));

    const int lower_size = LCMapStringEx(LOCALE_NAME_INVARIANT, LCMAP_LOWERCASE,
                                         normalized.data(), static_cast<int>(normalized.size()),
                                         nullptr, 0, nullptr, nullptr, 0);
    if (lower_size <= 0) throw Error("Windows Unicode lowercase conversion failed");
    std::wstring lowered(static_cast<std::size_t>(lower_size), L'\0');
    if (LCMapStringEx(LOCALE_NAME_INVARIANT, LCMAP_LOWERCASE,
                      normalized.data(), static_cast<int>(normalized.size()),
                      lowered.data(), lower_size, nullptr, nullptr, 0) <= 0) {
        throw Error("Windows Unicode lowercase conversion failed");
    }
    return lowered;
}

[[nodiscard]] bool windows_ctype(std::uint32_t cp, WORD mask) noexcept {
    wchar_t buffer[2]{};
    int length = 1;
    if (cp <= 0xFFFFU) {
        buffer[0] = static_cast<wchar_t>(cp);
    } else {
        const std::uint32_t value = cp - 0x10000U;
        buffer[0] = static_cast<wchar_t>(0xD800U + (value >> 10U));
        buffer[1] = static_cast<wchar_t>(0xDC00U + (value & 0x3FFU));
        length = 2;
    }
    WORD types[2]{};
    if (!GetStringTypeW(CT_CTYPE1, buffer, length, types)) return false;
    return (types[0] & mask) != 0;
}
#endif

[[nodiscard]] bool is_unicode_letter(std::uint32_t cp) noexcept {
#ifdef _WIN32
    return windows_ctype(cp, C1_ALPHA);
#else
    if ((cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z')) return true;
    // Broad Unicode letter ranges used by common prompts. Windows builds use the
    // operating system's complete Unicode character database above.
    return (cp >= 0x00C0U && cp <= 0x02AFU) ||
           (cp >= 0x0370U && cp <= 0x052FU) ||
           (cp >= 0x0531U && cp <= 0x1FFFU) ||
           (cp >= 0x2C00U && cp <= 0xD7FFU) ||
           (cp >= 0xF900U && cp <= 0xFDCFU) ||
           (cp >= 0x10000U && cp <= 0x2FA1FU);
#endif
}

[[nodiscard]] bool is_unicode_number(std::uint32_t cp) noexcept {
#ifdef _WIN32
    return windows_ctype(cp, C1_DIGIT);
#else
    if (cp >= '0' && cp <= '9') return true;
    return (cp >= 0x0660U && cp <= 0x0669U) || (cp >= 0x06F0U && cp <= 0x06F9U) ||
           (cp >= 0x0966U && cp <= 0x096FU) || (cp >= 0xFF10U && cp <= 0xFF19U);
#endif
}

[[nodiscard]] std::string extract_added_token_string(const json::Value& value,
                                                      std::string fallback) {
    if (value.is_string()) return value.as_string();
    if (value.is_object()) {
        if (const json::Value* content = value.find("content"); content && content->is_string()) {
            return content->as_string();
        }
    }
    return fallback;
}

} // namespace

const std::int32_t* TokenizedBatch::ids(std::size_t batch) const {
    if (batch >= batch_size) throw Error("token batch index is out of range");
    return input_ids.data() + batch * sequence_length;
}

const std::uint8_t* TokenizedBatch::mask(std::size_t batch) const {
    if (batch >= batch_size) throw Error("token batch index is out of range");
    return attention_mask.data() + batch * sequence_length;
}

std::size_t CLIPTokenizer::PairHash::operator()(
    const std::pair<std::string, std::string>& value) const noexcept {
    const std::size_t first = std::hash<std::string>{}(value.first);
    const std::size_t second = std::hash<std::string>{}(value.second);
    return first ^ (second + 0x9E3779B97F4A7C15ULL + (first << 6U) + (first >> 2U));
}

CLIPTokenizer::CLIPTokenizer(const std::filesystem::path& tokenizer_directory,
                             CLIPTokenizerOptions options) {
    load(tokenizer_directory, options);
}

CLIPTokenizer CLIPTokenizer::builtin(BuiltinCLIPTokenizer preset,
                                     CLIPTokenizerOptions options) {
    CLIPTokenizer tokenizer;
    tokenizer.load_builtin(preset, options);
    return tokenizer;
}

CLIPTokenizer CLIPTokenizer::sdxl_clip_l(CLIPTokenizerOptions options) {
    return builtin(BuiltinCLIPTokenizer::SDXLClipL, options);
}

CLIPTokenizer CLIPTokenizer::sdxl_openclip_big_g(CLIPTokenizerOptions options) {
    return builtin(BuiltinCLIPTokenizer::SDXLOpenCLIPBigG, options);
}

void CLIPTokenizer::reset(CLIPTokenizerOptions options) {
    options_ = options;
    encoder_.clear();
    decoder_.clear();
    merge_ranks_.clear();
    bpe_cache_.clear();
    unk_token_ = "<|endoftext|>";
    bos_token_ = "<|startoftext|>";
    eos_token_ = "<|endoftext|>";
    pad_token_ = "<|endoftext|>";
    unk_token_id_ = -1;
    bos_token_id_ = -1;
    eos_token_id_ = -1;
    pad_token_id_ = -1;
}

void CLIPTokenizer::load(const std::filesystem::path& tokenizer_directory,
                         CLIPTokenizerOptions options) {
    reset(options);
    const auto config_path = tokenizer_directory / "tokenizer_config.json";
    if (std::filesystem::is_regular_file(config_path)) load_tokenizer_config(config_path);
    build_byte_encoder();
    load_vocab(tokenizer_directory / "vocab.json");
    load_merges(tokenizer_directory / "merges.txt");
    resolve_special_token_ids();
}

void CLIPTokenizer::load_from_memory(std::string_view vocab_json,
                                     std::string_view merges_txt,
                                     std::string_view tokenizer_config_json,
                                     CLIPTokenizerOptions options) {
    reset(options);
    if (!tokenizer_config_json.empty()) {
        load_tokenizer_config_json(tokenizer_config_json, "in-memory tokenizer_config.json");
    }
    build_byte_encoder();
    load_vocab_json(vocab_json, "in-memory vocab.json");
    load_merges_text(merges_txt, "in-memory merges.txt");
    resolve_special_token_ids();
}

void CLIPTokenizer::load_builtin(BuiltinCLIPTokenizer preset,
                                 CLIPTokenizerOptions options) {
    reset(options);
    if (preset == BuiltinCLIPTokenizer::SDXLOpenCLIPBigG) {
        // This matches stabilityai/stable-diffusion-xl-base-1.0/tokenizer_2:
        // the OpenCLIP-bigG tokenizer uses token ID 0 ("!") for padding.
        pad_token_ = "!";
    }
    build_byte_encoder();
    build_standard_clip_vocab_from_merges(resources::sdxl_clip_merges());
    resolve_special_token_ids();
    if (encoder_.size() != 49'408U || bos_token_id_ != 49'406 || eos_token_id_ != 49'407) {
        throw Error("embedded SDXL CLIP tokenizer failed its vocabulary integrity check: size=" +
                    std::to_string(encoder_.size()) + " bos=" + std::to_string(bos_token_id_) +
                    " eos=" + std::to_string(eos_token_id_));
    }
}

void CLIPTokenizer::load_tokenizer_config(const std::filesystem::path& path) {
    load_tokenizer_config_json(read_text_file(path), path.string());
}

void CLIPTokenizer::load_tokenizer_config_json(std::string_view text,
                                               std::string_view source_name) {
    try {
        const json::Value config = json::parse(text);
        if (const json::Value* value = config.find("unk_token")) {
            unk_token_ = extract_added_token_string(*value, unk_token_);
        }
        if (const json::Value* value = config.find("bos_token")) {
            bos_token_ = extract_added_token_string(*value, bos_token_);
        }
        if (const json::Value* value = config.find("eos_token")) {
            eos_token_ = extract_added_token_string(*value, eos_token_);
        }
        if (const json::Value* value = config.find("pad_token")) {
            pad_token_ = extract_added_token_string(*value, pad_token_);
        }
        if (const json::Value* value = config.find("model_max_length"); value && value->is_number()) {
            const std::uint64_t max_length = value->as_u64();
            if (max_length > 0 && max_length < 1'000'000ULL && options_.max_length == 77) {
                options_.max_length = static_cast<std::size_t>(max_length);
            }
        }
    } catch (const std::exception& error) {
        throw Error("cannot parse tokenizer configuration " + std::string(source_name) + ": " + error.what());
    }
}

void CLIPTokenizer::build_byte_encoder() {
    std::vector<unsigned> bytes;
    bytes.reserve(256);
    for (unsigned value = 33; value <= 126; ++value) bytes.push_back(value);
    for (unsigned value = 161; value <= 172; ++value) bytes.push_back(value);
    for (unsigned value = 174; value <= 255; ++value) bytes.push_back(value);

    std::vector<std::uint32_t> codepoints(bytes.begin(), bytes.end());
    std::array<bool, 256> used{};
    for (const unsigned value : bytes) used[value] = true;
    std::uint32_t extra = 0;
    for (unsigned value = 0; value < 256; ++value) {
        if (!used[value]) {
            bytes.push_back(value);
            codepoints.push_back(256U + extra);
            ++extra;
        }
    }
    for (std::size_t i = 0; i < bytes.size(); ++i) {
        std::string encoded;
        append_utf8(encoded, codepoints[i]);
        byte_encoder_[bytes[i]] = std::move(encoded);
    }
}

void CLIPTokenizer::load_vocab(const std::filesystem::path& path) {
    load_vocab_json(read_text_file(path), path.string());
}

void CLIPTokenizer::load_vocab_json(std::string_view text,
                                    std::string_view source_name) {
    try {
        const json::Value root = json::parse(text);
        const auto& object = root.as_object();
        std::size_t highest = 0;
        for (const auto& [token, value] : object) {
            const std::uint64_t raw_id = value.as_u64();
            if (raw_id > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
                throw Error("token ID exceeds int32 range in " + std::string(source_name));
            }
            const auto id = static_cast<std::int32_t>(raw_id);
            if (!encoder_.emplace(token, id).second) throw Error("duplicate token in vocab: " + token);
            highest = std::max(highest, static_cast<std::size_t>(id));
        }
        decoder_.assign(highest + 1, {});
        for (const auto& [token, id] : encoder_) {
            decoder_[static_cast<std::size_t>(id)] = token;
        }
    } catch (const std::exception& error) {
        throw Error("cannot parse tokenizer vocabulary " + std::string(source_name) + ": " + error.what());
    }
}

void CLIPTokenizer::load_merges(const std::filesystem::path& path) {
    load_merges_text(read_text_file(path), path.string());
}

void CLIPTokenizer::load_merges_text(std::string_view text,
                                     std::string_view source_name) {
    std::istringstream input{std::string(text)};
    std::string line;
    std::size_t rank = 0;
    while (std::getline(input, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line.rfind("#version:", 0) == 0) continue;
        std::istringstream fields(line);
        std::string first;
        std::string second;
        fields >> first >> second;
        if (first.empty() || second.empty()) continue;
        if (!merge_ranks_.emplace(std::make_pair(first, second), rank++).second) {
            throw Error("duplicate BPE merge in " + std::string(source_name));
        }
    }
    if (rank == 0) throw Error("tokenizer merge table is empty: " + std::string(source_name));
}

void CLIPTokenizer::build_standard_clip_vocab_from_merges(std::string_view merges_txt) {
    std::vector<unsigned> ordered_bytes;
    ordered_bytes.reserve(256);
    for (unsigned value = 33; value <= 126; ++value) ordered_bytes.push_back(value);
    for (unsigned value = 161; value <= 172; ++value) ordered_bytes.push_back(value);
    for (unsigned value = 174; value <= 255; ++value) ordered_bytes.push_back(value);
    std::array<bool, 256> used{};
    for (const unsigned value : ordered_bytes) used[value] = true;
    for (unsigned value = 0; value < 256; ++value) {
        if (!used[value]) ordered_bytes.push_back(value);
    }

    std::vector<std::string> vocab;
    vocab.reserve(49'408);
    for (const unsigned byte : ordered_bytes) vocab.push_back(byte_encoder_[byte]);
    for (const unsigned byte : ordered_bytes) vocab.push_back(byte_encoder_[byte] + "</w>");

    std::istringstream input{std::string(merges_txt)};
    std::string line;
    std::size_t rank = 0;
    while (std::getline(input, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line.rfind("#version:", 0) == 0) continue;
        std::istringstream fields(line);
        std::string first;
        std::string second;
        fields >> first >> second;
        if (first.empty() || second.empty()) continue;
        if (!merge_ranks_.emplace(std::make_pair(first, second), rank++).second) {
            throw Error("embedded SDXL merge table contains a duplicate pair");
        }
        vocab.push_back(first + second);
    }
    vocab.push_back("<|startoftext|>");
    vocab.push_back("<|endoftext|>");

    encoder_.reserve(vocab.size());
    decoder_ = vocab;
    for (std::size_t id = 0; id < vocab.size(); ++id) {
        if (!encoder_.emplace(vocab[id], static_cast<std::int32_t>(id)).second) {
            throw Error("embedded SDXL vocabulary contains a duplicate token: " + vocab[id]);
        }
    }
}

void CLIPTokenizer::resolve_special_token_ids() {
    auto require = [&](const std::string& token) -> std::int32_t {
        const auto it = encoder_.find(token);
        if (it == encoder_.end()) throw Error("special token is missing from CLIP vocabulary: " + token);
        return it->second;
    };
    unk_token_id_ = require(unk_token_);
    bos_token_id_ = require(bos_token_);
    eos_token_id_ = require(eos_token_);
    pad_token_id_ = require(pad_token_);
}

std::string CLIPTokenizer::normalize(std::string_view text) const {
#ifdef _WIN32
    std::string lowered = wide_to_utf8(normalize_lower_windows(utf8_to_wide(text)));
#else
    // The standard library has no Unicode normalization API. This fallback
    // preserves valid UTF-8 and performs ASCII lowercase. Windows uses NFC and
    // invariant Unicode lowercase through NormalizeString/LCMapStringEx.
    std::string lowered(text);
    for (char& c : lowered) {
        const auto byte = static_cast<unsigned char>(c);
        if (byte >= 'A' && byte <= 'Z') c = static_cast<char>(byte - 'A' + 'a');
    }
#endif

    std::string collapsed;
    collapsed.reserve(lowered.size());
    bool pending_space = false;
    std::size_t position = 0;
    while (position < lowered.size()) {
        const DecodedCodepoint decoded = decode_utf8(lowered, position);
        if (is_unicode_space(decoded.value)) {
            pending_space = !collapsed.empty();
        } else {
            if (pending_space) collapsed.push_back(' ');
            pending_space = false;
            collapsed.append(lowered, position, decoded.bytes);
        }
        position += decoded.bytes;
    }
    return collapsed;
}

std::vector<std::string> CLIPTokenizer::pre_tokenize(std::string_view normalized) const {
    static constexpr std::array<std::string_view, 7> contractions{
        "'s", "'t", "'re", "'ve", "'m", "'ll", "'d"
    };
    std::vector<std::string> tokens;
    std::size_t position = 0;
    while (position < normalized.size()) {
        const DecodedCodepoint current = decode_utf8(normalized, position);
        if (is_unicode_space(current.value)) {
            position += current.bytes;
            continue;
        }

        if (normalized.substr(position, bos_token_.size()) == bos_token_) {
            tokens.push_back(bos_token_);
            position += bos_token_.size();
            continue;
        }
        if (normalized.substr(position, eos_token_.size()) == eos_token_) {
            tokens.push_back(eos_token_);
            position += eos_token_.size();
            continue;
        }

        bool matched_contraction = false;
        if (normalized[position] == '\'') {
            for (const std::string_view contraction : contractions) {
                if (normalized.substr(position, contraction.size()) == contraction) {
                    tokens.emplace_back(contraction);
                    position += contraction.size();
                    matched_contraction = true;
                    break;
                }
            }
        }
        if (matched_contraction) continue;

        if (is_unicode_letter(current.value)) {
            const std::size_t begin = position;
            position += current.bytes;
            while (position < normalized.size()) {
                const DecodedCodepoint next = decode_utf8(normalized, position);
                if (!is_unicode_letter(next.value)) break;
                position += next.bytes;
            }
            tokens.emplace_back(normalized.substr(begin, position - begin));
            continue;
        }

        if (is_unicode_number(current.value)) {
            tokens.emplace_back(normalized.substr(position, current.bytes));
            position += current.bytes;
            continue;
        }

        const std::size_t begin = position;
        position += current.bytes;
        while (position < normalized.size()) {
            const DecodedCodepoint next = decode_utf8(normalized, position);
            if (is_unicode_space(next.value) || is_unicode_letter(next.value) || is_unicode_number(next.value)) break;
            bool next_is_contraction = false;
            if (normalized[position] == '\'') {
                for (const std::string_view contraction : contractions) {
                    if (normalized.substr(position, contraction.size()) == contraction) {
                        next_is_contraction = true;
                        break;
                    }
                }
            }
            if (next_is_contraction || normalized.substr(position, bos_token_.size()) == bos_token_ ||
                normalized.substr(position, eos_token_.size()) == eos_token_) {
                break;
            }
            position += next.bytes;
        }
        tokens.emplace_back(normalized.substr(begin, position - begin));
    }
    return tokens;
}

std::vector<std::string> CLIPTokenizer::bpe(std::string_view token) const {
    if (token == bos_token_ || token == eos_token_) return {std::string(token)};
    const auto cached = bpe_cache_.find(std::string(token));
    if (cached != bpe_cache_.end()) return cached->second;

    std::vector<std::string> word;
    word.reserve(token.size());
    for (const unsigned char byte : token) word.push_back(byte_encoder_[byte]);
    if (word.empty()) return {};
    word.back() += "</w>";

    while (word.size() > 1) {
        std::size_t best_rank = std::numeric_limits<std::size_t>::max();
        std::pair<std::string, std::string> best_pair;
        for (std::size_t i = 0; i + 1 < word.size(); ++i) {
            const auto pair = std::make_pair(word[i], word[i + 1]);
            const auto found = merge_ranks_.find(pair);
            if (found != merge_ranks_.end() && found->second < best_rank) {
                best_rank = found->second;
                best_pair = pair;
            }
        }
        if (best_rank == std::numeric_limits<std::size_t>::max()) break;

        std::vector<std::string> merged;
        merged.reserve(word.size());
        for (std::size_t i = 0; i < word.size();) {
            if (i + 1 < word.size() && word[i] == best_pair.first && word[i + 1] == best_pair.second) {
                merged.push_back(word[i] + word[i + 1]);
                i += 2;
            } else {
                merged.push_back(word[i]);
                ++i;
            }
        }
        word = std::move(merged);
    }

    bpe_cache_.emplace(std::string(token), word);
    return word;
}

std::int32_t CLIPTokenizer::token_to_id(std::string_view token) const {
    const auto found = encoder_.find(std::string(token));
    return found == encoder_.end() ? unk_token_id_ : found->second;
}

std::vector<std::int32_t> CLIPTokenizer::encode_tokens(std::string_view text) const {
    const std::string normalized = normalize(text);
    const std::vector<std::string> pieces = pre_tokenize(normalized);
    std::vector<std::int32_t> ids;
    for (const std::string& piece : pieces) {
        for (const std::string& token : bpe(piece)) ids.push_back(token_to_id(token));
    }
    return ids;
}

TokenizedBatch CLIPTokenizer::encode(std::string_view text) const {
    return encode_batch({std::string(text)});
}

TokenizedBatch CLIPTokenizer::encode_batch(const std::vector<std::string>& texts) const {
    if (texts.empty()) throw Error("CLIP tokenizer batch cannot be empty");
    if (options_.max_length == 0) throw Error("CLIP tokenizer max_length cannot be zero");

    std::vector<std::vector<std::int32_t>> all_ids;
    all_ids.reserve(texts.size());
    std::size_t output_length = options_.pad_to_max_length ? options_.max_length : 0;
    bool any_truncated = false;

    for (const std::string& text : texts) {
        std::vector<std::int32_t> ids = encode_tokens(text);
        if (options_.add_special_tokens) {
            ids.insert(ids.begin(), bos_token_id_);
            ids.push_back(eos_token_id_);
        }
        if (ids.size() > options_.max_length) {
            if (!options_.truncate) throw Error("CLIP token sequence exceeds max_length");
            ids.resize(options_.max_length);
            if (options_.add_special_tokens) ids.back() = eos_token_id_;
            any_truncated = true;
        }
        output_length = std::max(output_length, ids.size());
        all_ids.push_back(std::move(ids));
    }

    TokenizedBatch result;
    result.batch_size = texts.size();
    result.sequence_length = output_length;
    result.input_ids.assign(result.batch_size * output_length, pad_token_id_);
    result.attention_mask.assign(result.batch_size * output_length, 0);
    result.unpadded_lengths.resize(result.batch_size);
    result.any_truncated = any_truncated;

    for (std::size_t batch = 0; batch < all_ids.size(); ++batch) {
        const auto& ids = all_ids[batch];
        result.unpadded_lengths[batch] = ids.size();
        std::copy(ids.begin(), ids.end(), result.input_ids.begin() + static_cast<std::ptrdiff_t>(batch * output_length));
        std::fill_n(result.attention_mask.begin() + static_cast<std::ptrdiff_t>(batch * output_length),
                    ids.size(), static_cast<std::uint8_t>(1));
    }
    return result;
}

} // namespace sdxl
