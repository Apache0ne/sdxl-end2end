#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace sdxl {

class Error : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

enum class DType {
    Bool,
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    U64,
    I64,
    F16,
    BF16,
    F32,
    F64,
    F8E4M3FN,
    F8E5M2,
    Unknown
};

[[nodiscard]] std::string_view dtype_name(DType dtype) noexcept;
[[nodiscard]] std::size_t dtype_size(DType dtype);
[[nodiscard]] DType parse_dtype(std::string_view text);

class MappedFile final {
public:
    explicit MappedFile(const std::filesystem::path& path);
    ~MappedFile();

    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;
    MappedFile(MappedFile&&) = delete;
    MappedFile& operator=(MappedFile&&) = delete;

    [[nodiscard]] const std::byte* data() const noexcept { return data_; }
    [[nodiscard]] std::uint64_t size() const noexcept { return size_; }
    [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }

private:
    std::filesystem::path path_;
    const std::byte* data_ = nullptr;
    std::uint64_t size_ = 0;
#ifdef _WIN32
    void* file_handle_ = nullptr;
    void* mapping_handle_ = nullptr;
#else
    int fd_ = -1;
#endif
};

struct TensorView {
    std::shared_ptr<const MappedFile> owner;
    const std::byte* data = nullptr;
    DType dtype = DType::Unknown;
    std::vector<std::uint64_t> shape;
    std::vector<std::int64_t> strides_bytes;
    std::uint64_t storage_bytes = 0;
    std::string source_key;
    std::filesystem::path source_file;

    [[nodiscard]] std::uint64_t element_count() const;
    [[nodiscard]] std::uint64_t logical_bytes() const;
    [[nodiscard]] bool contiguous() const noexcept;
    [[nodiscard]] TensorView slice(std::size_t dim, std::uint64_t start, std::uint64_t count) const;
    [[nodiscard]] TensorView transpose(std::size_t dim_a, std::size_t dim_b) const;
    [[nodiscard]] TensorView squeeze_trailing_ones() const;
};

struct SafeTensorEntry {
    DType dtype = DType::Unknown;
    std::vector<std::uint64_t> shape;
    std::uint64_t begin = 0;
    std::uint64_t end = 0;
};

class SafeTensorFile final {
public:
    explicit SafeTensorFile(const std::filesystem::path& path);

    [[nodiscard]] const std::filesystem::path& path() const noexcept;
    [[nodiscard]] const std::map<std::string, SafeTensorEntry, std::less<>>& entries() const noexcept;
    [[nodiscard]] bool contains(std::string_view key) const noexcept;
    [[nodiscard]] TensorView tensor(std::string_view key) const;

private:
    std::shared_ptr<MappedFile> mapping_;
    std::uint64_t data_base_ = 0;
    std::map<std::string, SafeTensorEntry, std::less<>> entries_;
};

class WeightStore final {
public:
    void add_file(const std::filesystem::path& path);
    void add_directory(const std::filesystem::path& directory, bool recursive = false);

    [[nodiscard]] bool contains(std::string_view key) const noexcept;
    [[nodiscard]] const TensorView* find(std::string_view key) const noexcept;
    [[nodiscard]] const TensorView& at(std::string_view key) const;
    [[nodiscard]] std::size_t tensor_count() const noexcept { return tensors_.size(); }
    [[nodiscard]] const std::unordered_map<std::string, TensorView>& tensors() const noexcept { return tensors_; }
    [[nodiscard]] const std::vector<std::shared_ptr<SafeTensorFile>>& files() const noexcept { return files_; }

private:
    std::vector<std::shared_ptr<SafeTensorFile>> files_;
    std::unordered_map<std::string, TensorView> tensors_;
};

} // namespace sdxl
