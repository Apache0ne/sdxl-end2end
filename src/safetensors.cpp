#include "sdxl/safetensors.hpp"
#include "sdxl/json.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <numeric>
#include <set>
#include <sstream>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace sdxl {

namespace {

constexpr std::uint64_t kMaxHeaderBytes = 100ULL * 1024ULL * 1024ULL;

[[nodiscard]] std::uint64_t checked_mul(std::uint64_t a, std::uint64_t b, const char* what) {
    if (a != 0 && b > std::numeric_limits<std::uint64_t>::max() / a) {
        throw Error(std::string("integer overflow while computing ") + what);
    }
    return a * b;
}

[[nodiscard]] std::uint64_t read_le_u64(const std::byte* bytes) noexcept {
    std::uint64_t value = 0;
    for (unsigned i = 0; i < 8; ++i) {
        value |= static_cast<std::uint64_t>(std::to_integer<unsigned char>(bytes[i])) << (8U * i);
    }
    return value;
}

[[nodiscard]] std::vector<std::int64_t> contiguous_strides(
    const std::vector<std::uint64_t>& shape,
    std::size_t item_size) {
    std::vector<std::int64_t> strides(shape.size(), 0);
    std::uint64_t stride = item_size;
    for (std::size_t i = shape.size(); i-- > 0;) {
        if (stride > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
            throw Error("tensor stride exceeds int64 range");
        }
        strides[i] = static_cast<std::int64_t>(stride);
        stride = checked_mul(stride, shape[i], "tensor stride");
    }
    return strides;
}

#ifdef _WIN32
[[nodiscard]] std::string windows_error_message(DWORD code) {
    LPSTR buffer = nullptr;
    const DWORD length = FormatMessageA(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
        nullptr,
        code,
        MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        reinterpret_cast<LPSTR>(&buffer),
        0,
        nullptr);
    std::string message = length && buffer ? std::string(buffer, length) : "unknown Windows error";
    if (buffer) LocalFree(buffer);
    return message;
}
#endif

} // namespace

std::string_view dtype_name(DType dtype) noexcept {
    switch (dtype) {
    case DType::Bool: return "BOOL";
    case DType::U8: return "U8";
    case DType::I8: return "I8";
    case DType::U16: return "U16";
    case DType::I16: return "I16";
    case DType::U32: return "U32";
    case DType::I32: return "I32";
    case DType::U64: return "U64";
    case DType::I64: return "I64";
    case DType::F16: return "F16";
    case DType::BF16: return "BF16";
    case DType::F32: return "F32";
    case DType::F64: return "F64";
    case DType::F8E4M3FN: return "F8_E4M3";
    case DType::F8E5M2: return "F8_E5M2";
    case DType::Unknown: return "UNKNOWN";
    }
    return "UNKNOWN";
}

std::size_t dtype_size(DType dtype) {
    switch (dtype) {
    case DType::Bool:
    case DType::U8:
    case DType::I8:
    case DType::F8E4M3FN:
    case DType::F8E5M2:
        return 1;
    case DType::U16:
    case DType::I16:
    case DType::F16:
    case DType::BF16:
        return 2;
    case DType::U32:
    case DType::I32:
    case DType::F32:
        return 4;
    case DType::U64:
    case DType::I64:
    case DType::F64:
        return 8;
    case DType::Unknown:
        break;
    }
    throw Error("unsupported or unknown tensor dtype");
}

DType parse_dtype(std::string_view text) {
    if (text == "BOOL") return DType::Bool;
    if (text == "U8") return DType::U8;
    if (text == "I8") return DType::I8;
    if (text == "U16") return DType::U16;
    if (text == "I16") return DType::I16;
    if (text == "U32") return DType::U32;
    if (text == "I32") return DType::I32;
    if (text == "U64") return DType::U64;
    if (text == "I64") return DType::I64;
    if (text == "F16") return DType::F16;
    if (text == "BF16") return DType::BF16;
    if (text == "F32") return DType::F32;
    if (text == "F64") return DType::F64;
    if (text == "F8_E4M3" || text == "F8_E4M3FN") return DType::F8E4M3FN;
    if (text == "F8_E5M2") return DType::F8E5M2;
    return DType::Unknown;
}

MappedFile::MappedFile(const std::filesystem::path& path) : path_(path) {
#ifdef _WIN32
    const std::wstring wide = path.wstring();
    HANDLE file = CreateFileW(wide.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                              FILE_ATTRIBUTE_NORMAL | FILE_FLAG_RANDOM_ACCESS, nullptr);
    if (file == INVALID_HANDLE_VALUE) {
        throw Error("CreateFileW failed for " + path.string() + ": " +
                    windows_error_message(GetLastError()));
    }
    file_handle_ = file;

    LARGE_INTEGER size{};
    if (!GetFileSizeEx(file, &size) || size.QuadPart < 0) {
        const DWORD code = GetLastError();
        CloseHandle(file);
        file_handle_ = nullptr;
        throw Error("GetFileSizeEx failed for " + path.string() + ": " +
                    windows_error_message(code));
    }
    size_ = static_cast<std::uint64_t>(size.QuadPart);
    if (size_ == 0) {
        CloseHandle(file);
        file_handle_ = nullptr;
        throw Error("cannot memory-map empty file: " + path.string());
    }

    HANDLE mapping = CreateFileMappingW(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!mapping) {
        const DWORD code = GetLastError();
        CloseHandle(file);
        file_handle_ = nullptr;
        throw Error("CreateFileMappingW failed for " + path.string() + ": " +
                    windows_error_message(code));
    }
    mapping_handle_ = mapping;

    const void* view = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
    if (!view) {
        const DWORD code = GetLastError();
        CloseHandle(mapping);
        CloseHandle(file);
        mapping_handle_ = nullptr;
        file_handle_ = nullptr;
        throw Error("MapViewOfFile failed for " + path.string() + ": " +
                    windows_error_message(code));
    }
    data_ = static_cast<const std::byte*>(view);
#else
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0) {
        throw Error("open failed for " + path.string() + ": " + std::strerror(errno));
    }
    struct stat st{};
    if (::fstat(fd_, &st) != 0 || st.st_size <= 0) {
        const int code = errno;
        ::close(fd_);
        fd_ = -1;
        throw Error("fstat failed or file is empty for " + path.string() + ": " +
                    std::strerror(code));
    }
    size_ = static_cast<std::uint64_t>(st.st_size);
    void* view = ::mmap(nullptr, static_cast<std::size_t>(size_), PROT_READ, MAP_PRIVATE, fd_, 0);
    if (view == MAP_FAILED) {
        const int code = errno;
        ::close(fd_);
        fd_ = -1;
        throw Error("mmap failed for " + path.string() + ": " + std::strerror(code));
    }
    data_ = static_cast<const std::byte*>(view);
#endif
}

MappedFile::~MappedFile() {
#ifdef _WIN32
    if (data_) UnmapViewOfFile(data_);
    if (mapping_handle_) CloseHandle(static_cast<HANDLE>(mapping_handle_));
    if (file_handle_) CloseHandle(static_cast<HANDLE>(file_handle_));
#else
    if (data_) ::munmap(const_cast<std::byte*>(data_), static_cast<std::size_t>(size_));
    if (fd_ >= 0) ::close(fd_);
#endif
}

std::uint64_t TensorView::element_count() const {
    std::uint64_t count = 1;
    for (const std::uint64_t dim : shape) {
        count = checked_mul(count, dim, "tensor element count");
    }
    return count;
}

std::uint64_t TensorView::logical_bytes() const {
    return checked_mul(element_count(), static_cast<std::uint64_t>(dtype_size(dtype)),
                       "tensor byte size");
}

bool TensorView::contiguous() const noexcept {
    try {
        return strides_bytes == contiguous_strides(shape, dtype_size(dtype));
    } catch (...) {
        return false;
    }
}

TensorView TensorView::slice(std::size_t dim, std::uint64_t start, std::uint64_t count) const {
    if (dim >= shape.size()) throw Error("slice dimension is out of range");
    if (start > shape[dim] || count > shape[dim] - start) throw Error("slice is out of bounds");
    if (strides_bytes.size() != shape.size()) throw Error("tensor view has invalid stride rank");

    TensorView result = *this;
    const std::uint64_t stride = static_cast<std::uint64_t>(strides_bytes[dim]);
    const std::uint64_t offset = checked_mul(start, stride, "slice byte offset");
    result.data += offset;
    result.shape[dim] = count;
    result.storage_bytes = result.logical_bytes();
    result.source_key += "[slice]";
    return result;
}

TensorView TensorView::transpose(std::size_t dim_a, std::size_t dim_b) const {
    if (dim_a >= shape.size() || dim_b >= shape.size()) throw Error("transpose dimension is out of range");
    TensorView result = *this;
    std::swap(result.shape[dim_a], result.shape[dim_b]);
    std::swap(result.strides_bytes[dim_a], result.strides_bytes[dim_b]);
    result.source_key += "[transpose]";
    return result;
}

TensorView TensorView::squeeze_trailing_ones() const {
    TensorView result = *this;
    while (!result.shape.empty() && result.shape.back() == 1) {
        result.shape.pop_back();
        result.strides_bytes.pop_back();
    }
    result.source_key += "[squeeze]";
    return result;
}

SafeTensorFile::SafeTensorFile(const std::filesystem::path& path)
    : mapping_(std::make_shared<MappedFile>(path)) {
    if (mapping_->size() < 8) throw Error("safetensors file is smaller than its 8-byte header length");

    const std::uint64_t header_size = read_le_u64(mapping_->data());
    if (header_size == 0 || header_size > kMaxHeaderBytes) {
        throw Error("invalid safetensors header length in " + path.string());
    }
    if (header_size > mapping_->size() - 8) {
        throw Error("safetensors header extends beyond end of file: " + path.string());
    }

    data_base_ = 8 + header_size;
    const char* header_data = reinterpret_cast<const char*>(mapping_->data() + 8);
    if (header_data[0] != '{') {
        throw Error("safetensors header does not begin with '{': " + path.string());
    }

    const json::Value root = json::parse(std::string_view(header_data, static_cast<std::size_t>(header_size)));
    const auto& object = root.as_object();
    const std::uint64_t data_size = mapping_->size() - data_base_;

    struct Interval { std::uint64_t begin; std::uint64_t end; std::string key; };
    std::vector<Interval> intervals;

    for (const auto& [key, value] : object) {
        if (key == "__metadata__") continue;
        (void)value.as_object();

        const DType dtype = parse_dtype(value.at("dtype").as_string());
        if (dtype == DType::Unknown) {
            throw Error("unsupported dtype for tensor '" + key + "': " + value.at("dtype").as_string());
        }

        std::vector<std::uint64_t> shape;
        for (const json::Value& dim : value.at("shape").as_array()) {
            shape.push_back(dim.as_u64());
        }

        const auto& offsets = value.at("data_offsets").as_array();
        if (offsets.size() != 2) throw Error("tensor '" + key + "' has invalid data_offsets");
        const std::uint64_t begin = offsets[0].as_u64();
        const std::uint64_t end = offsets[1].as_u64();
        if (begin > end || end > data_size) {
            throw Error("tensor '" + key + "' has out-of-range data offsets");
        }

        std::uint64_t elements = 1;
        for (const auto dim : shape) elements = checked_mul(elements, dim, "tensor element count");
        const std::uint64_t expected_bytes = checked_mul(
            elements, static_cast<std::uint64_t>(dtype_size(dtype)), "tensor byte size");
        if (expected_bytes != end - begin) {
            std::ostringstream stream;
            stream << "tensor '" << key << "' byte size mismatch: shape/dtype require "
                   << expected_bytes << " bytes but offsets contain " << (end - begin);
            throw Error(stream.str());
        }

        entries_.emplace(key, SafeTensorEntry{dtype, std::move(shape), begin, end});
        if (begin != end) intervals.push_back({begin, end, key});
    }

    std::sort(intervals.begin(), intervals.end(), [](const Interval& a, const Interval& b) {
        return a.begin < b.begin;
    });
    std::uint64_t cursor = 0;
    for (const Interval& interval : intervals) {
        if (interval.begin != cursor) {
            throw Error("safetensors data buffer has a hole or overlap before tensor '" + interval.key + "'");
        }
        cursor = interval.end;
    }
    if (!intervals.empty() && cursor != data_size) {
        throw Error("safetensors data buffer is not fully indexed");
    }
}

const std::filesystem::path& SafeTensorFile::path() const noexcept { return mapping_->path(); }
const std::map<std::string, SafeTensorEntry, std::less<>>& SafeTensorFile::entries() const noexcept { return entries_; }

bool SafeTensorFile::contains(std::string_view key) const noexcept {
    return entries_.find(key) != entries_.end();
}

TensorView SafeTensorFile::tensor(std::string_view key) const {
    const auto it = entries_.find(key);
    if (it == entries_.end()) throw Error("tensor not found: " + std::string(key));
    const SafeTensorEntry& entry = it->second;
    TensorView view;
    view.owner = mapping_;
    view.data = mapping_->data() + data_base_ + entry.begin;
    view.dtype = entry.dtype;
    view.shape = entry.shape;
    view.strides_bytes = contiguous_strides(entry.shape, dtype_size(entry.dtype));
    view.storage_bytes = entry.end - entry.begin;
    view.source_key = std::string(key);
    view.source_file = mapping_->path();
    return view;
}

void WeightStore::add_file(const std::filesystem::path& path) {
    auto file = std::make_shared<SafeTensorFile>(path);
    for (const auto& [key, _] : file->entries()) {
        if (tensors_.contains(key)) {
            throw Error("duplicate tensor key across safetensors files: " + key);
        }
        tensors_.emplace(key, file->tensor(key));
    }
    files_.push_back(std::move(file));
}

void WeightStore::add_directory(const std::filesystem::path& directory, bool recursive) {
    if (!std::filesystem::is_directory(directory)) {
        throw Error("not a directory: " + directory.string());
    }
    std::vector<std::filesystem::path> files;
    if (recursive) {
        for (const auto& entry : std::filesystem::recursive_directory_iterator(directory)) {
            if (entry.is_regular_file() && entry.path().extension() == ".safetensors") files.push_back(entry.path());
        }
    } else {
        for (const auto& entry : std::filesystem::directory_iterator(directory)) {
            if (entry.is_regular_file() && entry.path().extension() == ".safetensors") files.push_back(entry.path());
        }
    }
    std::sort(files.begin(), files.end());
    if (files.empty()) throw Error("no .safetensors files found in " + directory.string());
    for (const auto& file : files) add_file(file);
}

bool WeightStore::contains(std::string_view key) const noexcept {
    return tensors_.find(std::string(key)) != tensors_.end();
}

const TensorView* WeightStore::find(std::string_view key) const noexcept {
    const auto it = tensors_.find(std::string(key));
    return it == tensors_.end() ? nullptr : &it->second;
}

const TensorView& WeightStore::at(std::string_view key) const {
    const auto* view = find(key);
    if (!view) throw Error("tensor not found in weight store: " + std::string(key));
    return *view;
}

} // namespace sdxl
