#include "sdxl/json.hpp"

#include <charconv>
#include <cstdlib>
#include <cmath>
#include <limits>
#include <sstream>

namespace sdxl::json {

namespace {

[[noreturn]] void type_error(const char* expected) {
    throw Error(std::string("JSON value is not ") + expected);
}

void append_utf8(std::string& out, std::uint32_t codepoint) {
    if (codepoint <= 0x7F) {
        out.push_back(static_cast<char>(codepoint));
    } else if (codepoint <= 0x7FF) {
        out.push_back(static_cast<char>(0xC0 | (codepoint >> 6)));
        out.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
    } else if (codepoint <= 0xFFFF) {
        out.push_back(static_cast<char>(0xE0 | (codepoint >> 12)));
        out.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
    } else if (codepoint <= 0x10FFFF) {
        out.push_back(static_cast<char>(0xF0 | (codepoint >> 18)));
        out.push_back(static_cast<char>(0x80 | ((codepoint >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
    } else {
        throw Error("JSON unicode codepoint is out of range");
    }
}

class Parser final {
public:
    explicit Parser(std::string_view text) : text_(text) {}

    Value run() {
        skip_ws();
        Value value = parse_value();
        skip_ws();
        if (pos_ != text_.size()) {
            fail("unexpected trailing data");
        }
        return value;
    }

private:
    [[noreturn]] void fail(const std::string& message) const {
        std::ostringstream stream;
        stream << "JSON parse error at byte " << pos_ << ": " << message;
        throw Error(stream.str());
    }

    [[nodiscard]] bool eof() const noexcept { return pos_ >= text_.size(); }
    [[nodiscard]] char peek() const {
        if (eof()) {
            fail("unexpected end of input");
        }
        return text_[pos_];
    }

    char take() {
        const char c = peek();
        ++pos_;
        return c;
    }

    void expect(char expected) {
        const char actual = take();
        if (actual != expected) {
            fail(std::string("expected '") + expected + "'");
        }
    }

    void skip_ws() noexcept {
        while (pos_ < text_.size()) {
            const char c = text_[pos_];
            if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
                ++pos_;
            } else {
                break;
            }
        }
    }

    Value parse_value() {
        if (eof()) {
            fail("expected value");
        }

        switch (peek()) {
        case 'n': return parse_literal("null", Value(nullptr));
        case 't': return parse_literal("true", Value(true));
        case 'f': return parse_literal("false", Value(false));
        case '"': return Value(parse_string());
        case '[': return parse_array();
        case '{': return parse_object();
        default:
            if (peek() == '-' || (peek() >= '0' && peek() <= '9')) {
                return Value(parse_number());
            }
            fail("invalid value");
        }
    }

    Value parse_literal(std::string_view literal, Value value) {
        if (text_.substr(pos_, literal.size()) != literal) {
            fail("invalid literal");
        }
        pos_ += literal.size();
        return value;
    }

    static std::uint32_t hex_value(char c) {
        if (c >= '0' && c <= '9') return static_cast<std::uint32_t>(c - '0');
        if (c >= 'a' && c <= 'f') return static_cast<std::uint32_t>(10 + c - 'a');
        if (c >= 'A' && c <= 'F') return static_cast<std::uint32_t>(10 + c - 'A');
        throw Error("invalid hex digit in JSON unicode escape");
    }

    std::uint32_t parse_u16_escape() {
        std::uint32_t value = 0;
        for (int i = 0; i < 4; ++i) {
            if (eof()) {
                fail("truncated unicode escape");
            }
            value = static_cast<std::uint32_t>((value << 4) | hex_value(take()));
        }
        return value;
    }

    std::string parse_string() {
        expect('"');
        std::string out;
        while (!eof()) {
            const char c = take();
            if (c == '"') {
                return out;
            }
            if (static_cast<unsigned char>(c) < 0x20) {
                fail("control character in string");
            }
            if (c != '\\') {
                out.push_back(c);
                continue;
            }

            if (eof()) {
                fail("truncated escape sequence");
            }
            switch (take()) {
            case '"': out.push_back('"'); break;
            case '\\': out.push_back('\\'); break;
            case '/': out.push_back('/'); break;
            case 'b': out.push_back('\b'); break;
            case 'f': out.push_back('\f'); break;
            case 'n': out.push_back('\n'); break;
            case 'r': out.push_back('\r'); break;
            case 't': out.push_back('\t'); break;
            case 'u': {
                std::uint32_t cp = parse_u16_escape();
                if (cp >= 0xD800 && cp <= 0xDBFF) {
                    if (pos_ + 2 > text_.size() || text_[pos_] != '\\' || text_[pos_ + 1] != 'u') {
                        fail("high surrogate without low surrogate");
                    }
                    pos_ += 2;
                    const std::uint32_t low = parse_u16_escape();
                    if (low < 0xDC00 || low > 0xDFFF) {
                        fail("invalid low surrogate");
                    }
                    cp = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00);
                } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
                    fail("unexpected low surrogate");
                }
                append_utf8(out, cp);
                break;
            }
            default:
                fail("invalid escape sequence");
            }
        }
        fail("unterminated string");
    }

    double parse_number() {
        const std::size_t begin = pos_;
        if (peek() == '-') ++pos_;

        if (eof()) fail("truncated number");
        if (text_[pos_] == '0') {
            ++pos_;
        } else {
            if (text_[pos_] < '1' || text_[pos_] > '9') fail("invalid number");
            while (pos_ < text_.size() && text_[pos_] >= '0' && text_[pos_] <= '9') ++pos_;
        }

        if (pos_ < text_.size() && text_[pos_] == '.') {
            ++pos_;
            if (pos_ >= text_.size() || text_[pos_] < '0' || text_[pos_] > '9') {
                fail("invalid fraction");
            }
            while (pos_ < text_.size() && text_[pos_] >= '0' && text_[pos_] <= '9') ++pos_;
        }

        if (pos_ < text_.size() && (text_[pos_] == 'e' || text_[pos_] == 'E')) {
            ++pos_;
            if (pos_ < text_.size() && (text_[pos_] == '+' || text_[pos_] == '-')) ++pos_;
            if (pos_ >= text_.size() || text_[pos_] < '0' || text_[pos_] > '9') {
                fail("invalid exponent");
            }
            while (pos_ < text_.size() && text_[pos_] >= '0' && text_[pos_] <= '9') ++pos_;
        }

        const std::string token(text_.substr(begin, pos_ - begin));
        char* end = nullptr;
        const double value = std::strtod(token.c_str(), &end);
        if (end == nullptr || *end != '\0' || !std::isfinite(value)) {
            fail("invalid or non-finite number");
        }
        return value;
    }

    Value parse_array() {
        expect('[');
        skip_ws();
        Value::Array values;
        if (!eof() && peek() == ']') {
            ++pos_;
            return Value(std::move(values));
        }
        while (true) {
            skip_ws();
            values.push_back(parse_value());
            skip_ws();
            const char delimiter = take();
            if (delimiter == ']') break;
            if (delimiter != ',') fail("expected ',' or ']'");
        }
        return Value(std::move(values));
    }

    Value parse_object() {
        expect('{');
        skip_ws();
        Value::Object object;
        if (!eof() && peek() == '}') {
            ++pos_;
            return Value(std::move(object));
        }
        while (true) {
            skip_ws();
            if (peek() != '"') fail("object key must be a string");
            std::string key = parse_string();
            skip_ws();
            expect(':');
            skip_ws();
            Value value = parse_value();
            const auto [_, inserted] = object.emplace(std::move(key), std::move(value));
            if (!inserted) fail("duplicate object key");
            skip_ws();
            const char delimiter = take();
            if (delimiter == '}') break;
            if (delimiter != ',') fail("expected ',' or '}'");
        }
        return Value(std::move(object));
    }

    std::string_view text_;
    std::size_t pos_ = 0;
};

} // namespace

bool Value::is_null() const noexcept { return std::holds_alternative<std::nullptr_t>(storage_); }
bool Value::is_bool() const noexcept { return std::holds_alternative<bool>(storage_); }
bool Value::is_number() const noexcept { return std::holds_alternative<double>(storage_); }
bool Value::is_string() const noexcept { return std::holds_alternative<std::string>(storage_); }
bool Value::is_array() const noexcept { return std::holds_alternative<Array>(storage_); }
bool Value::is_object() const noexcept { return std::holds_alternative<Object>(storage_); }

bool Value::as_bool() const {
    if (!is_bool()) type_error("a boolean");
    return std::get<bool>(storage_);
}

double Value::as_number() const {
    if (!is_number()) type_error("a number");
    return std::get<double>(storage_);
}

std::uint64_t Value::as_u64() const {
    const double value = as_number();
    if (value < 0.0 || value > static_cast<double>(std::numeric_limits<std::uint64_t>::max()) ||
        std::floor(value) != value) {
        throw Error("JSON number cannot be represented as uint64");
    }
    return static_cast<std::uint64_t>(value);
}

const std::string& Value::as_string() const {
    if (!is_string()) type_error("a string");
    return std::get<std::string>(storage_);
}

const Value::Array& Value::as_array() const {
    if (!is_array()) type_error("an array");
    return std::get<Array>(storage_);
}

const Value::Object& Value::as_object() const {
    if (!is_object()) type_error("an object");
    return std::get<Object>(storage_);
}

Value::Array& Value::as_array() {
    if (!is_array()) type_error("an array");
    return std::get<Array>(storage_);
}

Value::Object& Value::as_object() {
    if (!is_object()) type_error("an object");
    return std::get<Object>(storage_);
}

const Value& Value::at(std::string_view key) const {
    const auto& object = as_object();
    const auto it = object.find(key);
    if (it == object.end()) {
        throw Error("missing JSON key: " + std::string(key));
    }
    return it->second;
}

const Value* Value::find(std::string_view key) const noexcept {
    if (!is_object()) return nullptr;
    const auto& object = std::get<Object>(storage_);
    const auto it = object.find(key);
    return it == object.end() ? nullptr : &it->second;
}

Value parse(std::string_view text) {
    return Parser(text).run();
}

} // namespace sdxl::json
