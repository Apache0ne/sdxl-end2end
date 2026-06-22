#pragma once

#include "sdxl/cuda/image.hpp"

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <exception>
#include <filesystem>
#include <mutex>
#include <thread>
#include <vector>

namespace sdxl::cuda {

enum class ImageFileFormat : std::uint8_t { PNG, RawRGB };

struct ImageWriteRecord {
    std::filesystem::path path;
    double milliseconds = 0.0;
    std::size_t bytes = 0;
    ImageFileFormat format = ImageFileFormat::PNG;
};

class AsyncImageWriter final {
public:
    explicit AsyncImageWriter(bool enabled = true, std::size_t queue_limit = 8);
    ~AsyncImageWriter();

    AsyncImageWriter(const AsyncImageWriter&) = delete;
    AsyncImageWriter& operator=(const AsyncImageWriter&) = delete;

    void submit(std::filesystem::path path, RGBImage image,
                ImageFileFormat format = ImageFileFormat::PNG);
    void flush();
    [[nodiscard]] std::vector<ImageWriteRecord> take_records();

private:
    struct Job {
        std::filesystem::path path;
        RGBImage image;
        ImageFileFormat format = ImageFileFormat::PNG;
    };
    void worker_loop();
    void rethrow_worker_error();

    bool enabled_ = true;
    std::size_t queue_limit_ = 8;
    bool stopping_ = false;
    std::size_t active_jobs_ = 0;
    std::deque<Job> queue_;
    std::mutex mutex_;
    std::condition_variable ready_;
    std::condition_variable space_;
    std::condition_variable drained_;
    std::exception_ptr error_;
    std::vector<ImageWriteRecord> completed_;
    std::thread worker_;
};

} // namespace sdxl::cuda
