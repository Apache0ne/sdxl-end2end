#include "sdxl/cuda/async_image_writer.hpp"

#include <chrono>
#include <fstream>
#include <utility>

namespace sdxl::cuda {
namespace {

void write_raw_rgb(const std::filesystem::path& path, const RGBImage& image) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) throw CudaError("cannot create raw RGB file: " + path.string());
    output.write(reinterpret_cast<const char*>(image.pixels.data()),
                 static_cast<std::streamsize>(image.pixels.size()));
    if (!output) throw CudaError("failed while writing raw RGB file: " + path.string());
}

void write_image(const std::filesystem::path& path, const RGBImage& image,
                 ImageFileFormat format) {
    if (format == ImageFileFormat::RawRGB) write_raw_rgb(path, image);
    else write_png(path, image);
}

} // namespace

AsyncImageWriter::AsyncImageWriter(bool enabled, std::size_t queue_limit)
    : enabled_(enabled), queue_limit_(queue_limit == 0 ? 1 : queue_limit) {
    if (enabled_) worker_ = std::thread([this] { worker_loop(); });
}

AsyncImageWriter::~AsyncImageWriter() {
    try { flush(); } catch (...) {}
    if (enabled_) {
        {
            std::lock_guard lock(mutex_);
            stopping_ = true;
        }
        ready_.notify_all();
        if (worker_.joinable()) worker_.join();
    }
}

void AsyncImageWriter::rethrow_worker_error() {
    if (error_) std::rethrow_exception(error_);
}

void AsyncImageWriter::submit(std::filesystem::path path, RGBImage image,
                              ImageFileFormat format) {
    if (!enabled_) {
        const auto begin = std::chrono::steady_clock::now();
        write_image(path, image, format);
        const auto end = std::chrono::steady_clock::now();
        completed_.push_back(ImageWriteRecord{
            std::move(path), std::chrono::duration<double, std::milli>(end - begin).count(),
            image.pixels.size(), format});
        return;
    }
    std::unique_lock lock(mutex_);
    space_.wait(lock, [&] { return queue_.size() < queue_limit_ || error_ || stopping_; });
    rethrow_worker_error();
    if (stopping_) throw CudaError("image writer is stopping");
    queue_.push_back(Job{std::move(path), std::move(image), format});
    lock.unlock();
    ready_.notify_one();
}

void AsyncImageWriter::flush() {
    if (!enabled_) return;
    std::unique_lock lock(mutex_);
    drained_.wait(lock, [&] { return (queue_.empty() && active_jobs_ == 0) || error_; });
    rethrow_worker_error();
}

std::vector<ImageWriteRecord> AsyncImageWriter::take_records() {
    if (enabled_) flush();
    std::lock_guard lock(mutex_);
    std::vector<ImageWriteRecord> result;
    result.swap(completed_);
    return result;
}

void AsyncImageWriter::worker_loop() {
    for (;;) {
        Job job;
        {
            std::unique_lock lock(mutex_);
            ready_.wait(lock, [&] { return stopping_ || !queue_.empty(); });
            if (stopping_ && queue_.empty()) break;
            job = std::move(queue_.front());
            queue_.pop_front();
            ++active_jobs_;
            space_.notify_one();
        }
        try {
            const auto begin = std::chrono::steady_clock::now();
            write_image(job.path, job.image, job.format);
            const auto end = std::chrono::steady_clock::now();
            const ImageWriteRecord record{
                job.path, std::chrono::duration<double, std::milli>(end - begin).count(),
                job.image.pixels.size(), job.format};
            std::lock_guard lock(mutex_);
            completed_.push_back(record);
        } catch (...) {
            std::lock_guard lock(mutex_);
            if (!error_) error_ = std::current_exception();
        }
        {
            std::lock_guard lock(mutex_);
            --active_jobs_;
            if (queue_.empty() && active_jobs_ == 0) drained_.notify_all();
        }
    }
}

} // namespace sdxl::cuda
