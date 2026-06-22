#include "sdxl/cuda/image.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace sdxl::cuda {
namespace {

void append_u32_be(std::vector<std::uint8_t>& output, std::uint32_t value) {
    output.push_back(static_cast<std::uint8_t>((value >> 24U) & 0xFFU));
    output.push_back(static_cast<std::uint8_t>((value >> 16U) & 0xFFU));
    output.push_back(static_cast<std::uint8_t>((value >> 8U) & 0xFFU));
    output.push_back(static_cast<std::uint8_t>(value & 0xFFU));
}

[[nodiscard]] std::uint32_t crc32(const std::uint8_t* data, std::size_t size) {
    std::uint32_t crc = 0xFFFFFFFFU;
    for (std::size_t index = 0; index < size; ++index) {
        crc ^= data[index];
        for (int bit = 0; bit < 8; ++bit) {
            const std::uint32_t mask = 0U - (crc & 1U);
            crc = (crc >> 1U) ^ (0xEDB88320U & mask);
        }
    }
    return crc ^ 0xFFFFFFFFU;
}

[[nodiscard]] std::uint32_t adler32(const std::vector<std::uint8_t>& data) {
    constexpr std::uint32_t modulus = 65521U;
    std::uint32_t a = 1U;
    std::uint32_t b = 0U;
    for (const std::uint8_t value : data) {
        a = (a + value) % modulus;
        b = (b + a) % modulus;
    }
    return (b << 16U) | a;
}

void append_chunk(std::vector<std::uint8_t>& png,
                  const std::array<char, 4>& type,
                  const std::vector<std::uint8_t>& payload) {
    if (payload.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("PNG chunk exceeds 32-bit length");
    }
    append_u32_be(png, static_cast<std::uint32_t>(payload.size()));
    const std::size_t crc_begin = png.size();
    for (const char value : type) png.push_back(static_cast<std::uint8_t>(value));
    png.insert(png.end(), payload.begin(), payload.end());
    append_u32_be(png, crc32(png.data() + crc_begin, png.size() - crc_begin));
}

[[nodiscard]] std::vector<std::uint8_t> zlib_store(const std::vector<std::uint8_t>& raw) {
    std::vector<std::uint8_t> output;
    output.reserve(raw.size() + raw.size() / 65535U * 5U + 16U);
    output.push_back(0x78U); // deflate, 32K window
    output.push_back(0x01U); // fastest/no compression; header divisible by 31

    std::size_t offset = 0;
    do {
        const std::size_t remaining = raw.size() - offset;
        const std::size_t length = remaining > 65535U ? 65535U : remaining;
        const bool final = offset + length == raw.size();
        output.push_back(final ? 0x01U : 0x00U); // BFINAL + stored BTYPE
        const std::uint16_t len = static_cast<std::uint16_t>(length);
        const std::uint16_t complement = static_cast<std::uint16_t>(~len);
        output.push_back(static_cast<std::uint8_t>(len & 0xFFU));
        output.push_back(static_cast<std::uint8_t>((len >> 8U) & 0xFFU));
        output.push_back(static_cast<std::uint8_t>(complement & 0xFFU));
        output.push_back(static_cast<std::uint8_t>((complement >> 8U) & 0xFFU));
        output.insert(output.end(), raw.begin() + static_cast<std::ptrdiff_t>(offset),
                      raw.begin() + static_cast<std::ptrdiff_t>(offset + length));
        offset += length;
    } while (offset < raw.size());

    append_u32_be(output, adler32(raw));
    return output;
}

} // namespace

void write_png(const std::filesystem::path& path, const RGBImage& image) {
    if (image.width == 0 || image.height == 0) {
        throw std::runtime_error("cannot write an empty PNG image");
    }
    if (image.width > std::numeric_limits<std::uint32_t>::max() ||
        image.height > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("PNG dimensions exceed 32-bit limits");
    }
    const std::size_t expected = image.width * image.height * 3;
    if (image.pixels.size() != expected) {
        throw std::runtime_error("RGB image storage does not match its dimensions");
    }

    std::vector<std::uint8_t> raw;
    raw.reserve(image.height * (image.width * 3 + 1));
    const std::size_t row_bytes = image.width * 3;
    for (std::size_t y = 0; y < image.height; ++y) {
        raw.push_back(0U); // PNG filter type None
        const auto begin = image.pixels.begin() + static_cast<std::ptrdiff_t>(y * row_bytes);
        raw.insert(raw.end(), begin, begin + static_cast<std::ptrdiff_t>(row_bytes));
    }

    std::vector<std::uint8_t> png{
        0x89U, 0x50U, 0x4EU, 0x47U, 0x0DU, 0x0AU, 0x1AU, 0x0AU};
    std::vector<std::uint8_t> ihdr;
    ihdr.reserve(13);
    append_u32_be(ihdr, static_cast<std::uint32_t>(image.width));
    append_u32_be(ihdr, static_cast<std::uint32_t>(image.height));
    ihdr.push_back(8U); // bit depth
    ihdr.push_back(2U); // truecolor RGB
    ihdr.push_back(0U); // compression
    ihdr.push_back(0U); // filter
    ihdr.push_back(0U); // no interlace
    append_chunk(png, {'I', 'H', 'D', 'R'}, ihdr);
    append_chunk(png, {'I', 'D', 'A', 'T'}, zlib_store(raw));
    append_chunk(png, {'I', 'E', 'N', 'D'}, {});

    std::ofstream output(path, std::ios::binary);
    if (!output) throw std::runtime_error("cannot open PNG output: " + path.string());
    output.write(reinterpret_cast<const char*>(png.data()),
                 static_cast<std::streamsize>(png.size()));
    if (!output) throw std::runtime_error("failed while writing PNG output: " + path.string());
}

} // namespace sdxl::cuda
