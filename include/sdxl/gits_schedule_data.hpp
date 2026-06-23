#pragma once
#include <cstddef>
#include <vector>
namespace sdxl {
[[nodiscard]] std::vector<float> make_gits_sigmas(std::size_t steps, float coeff, float denoise);
}
