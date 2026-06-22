#pragma once

#include <string_view>

namespace sdxl::resources {

[[nodiscard]] std::string_view sdxl_clip_merges();
[[nodiscard]] std::string_view sdxl_clip_merges_sha256() noexcept;

} // namespace sdxl::resources
