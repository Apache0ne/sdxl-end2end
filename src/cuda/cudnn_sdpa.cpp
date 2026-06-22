#include "cudnn_sdpa.hpp"

#include "runtime_internal.hpp"

#include <cmath>
#include <cstdint>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>

#if defined(SDXL_HAS_CUDNN_FRONTEND_SDPA)
#include <cudnn_frontend.h>
#endif

namespace sdxl::cuda {

#if defined(SDXL_HAS_CUDNN_FRONTEND_SDPA)
namespace {
namespace fe = cudnn_frontend;

constexpr std::int64_t kQueryUid = 1;
constexpr std::int64_t kKeyUid = 2;
constexpr std::int64_t kValueUid = 3;
constexpr std::int64_t kOutputUid = 4;

struct CudnnSdpaPlan final {
    std::shared_ptr<fe::graph::Graph> graph;
    std::size_t workspace_bytes = 0;
};

[[nodiscard]] std::string plan_key(std::size_t batch,
                                   std::size_t query_sequence,
                                   std::size_t key_sequence,
                                   std::size_t heads,
                                   std::size_t head_dimension,
                                   bool causal) {
    std::ostringstream stream;
    stream << batch << ':' << query_sequence << ':' << key_sequence << ':'
           << heads << ':' << head_dimension << ':' << (causal ? 1 : 0);
    return stream.str();
}

[[nodiscard]] std::shared_ptr<CudnnSdpaPlan> build_plan(
    const Runtime& runtime,
    std::size_t batch,
    std::size_t query_sequence,
    std::size_t key_sequence,
    std::size_t heads,
    std::size_t head_dimension,
    bool causal) {
    const auto b = static_cast<std::int64_t>(batch);
    const auto h = static_cast<std::int64_t>(heads);
    const auto sq = static_cast<std::int64_t>(query_sequence);
    const auto sk = static_cast<std::int64_t>(key_sequence);
    const auto d = static_cast<std::int64_t>(head_dimension);

    auto plan = std::make_shared<CudnnSdpaPlan>();
    plan->graph = std::make_shared<fe::graph::Graph>();
    plan->graph->set_io_data_type(fe::DataType_t::HALF)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    // The engine stores [B,S,H,D]. Present it as logical [B,H,S,D] with
    // explicit BSHD strides so no layout conversion kernel is needed.
    const std::vector<std::int64_t> q_dimensions{b, h, sq, d};
    const std::vector<std::int64_t> q_strides{sq * h * d, d, h * d, 1};
    const std::vector<std::int64_t> kv_dimensions{b, h, sk, d};
    const std::vector<std::int64_t> kv_strides{sk * h * d, d, h * d, 1};

    auto q = plan->graph->tensor(fe::graph::Tensor_attributes()
        .set_name("Q").set_uid(kQueryUid)
        .set_dim(q_dimensions).set_stride(q_strides));
    auto k = plan->graph->tensor(fe::graph::Tensor_attributes()
        .set_name("K").set_uid(kKeyUid)
        .set_dim(kv_dimensions).set_stride(kv_strides));
    auto v = plan->graph->tensor(fe::graph::Tensor_attributes()
        .set_name("V").set_uid(kValueUid)
        .set_dim(kv_dimensions).set_stride(kv_strides));

    auto attributes = fe::graph::SDPA_attributes()
        .set_name("sdxl_sdpa")
        .set_generate_stats(false)
        .set_attn_scale(1.0F / std::sqrt(static_cast<float>(head_dimension)));
    if (causal) {
        attributes.set_diagonal_alignment(fe::DiagonalAlignment_t::TOP_LEFT)
            .set_diagonal_band_right_bound(0);
    }
    auto [o, stats] = plan->graph->sdpa(q, k, v, attributes);
    (void)stats;
    o->set_output(true).set_uid(kOutputUid)
        .set_dim(q_dimensions).set_stride(q_strides);

    auto status = plan->graph->build(runtime.cudnn(), {fe::HeurMode_t::A});
    if (status.is_bad()) {
        throw CudaError("cuDNN Frontend SDPA graph build failed: " + status.get_message());
    }
    std::int64_t workspace = 0;
    status = plan->graph->get_workspace_size(workspace);
    if (status.is_bad() || workspace < 0) {
        throw CudaError("cuDNN Frontend SDPA workspace query failed: " + status.get_message());
    }
    plan->workspace_bytes = static_cast<std::size_t>(workspace);
    return plan;
}

} // namespace
#endif

bool cudnn_frontend_sdpa_compiled() noexcept {
#if defined(SDXL_HAS_CUDNN_FRONTEND_SDPA)
    return true;
#else
    return false;
#endif
}

void launch_cudnn_frontend_sdpa(const Runtime& runtime,
                                const Tensor& query,
                                const Tensor& key,
                                const Tensor& value,
                                Tensor& output,
                                std::size_t heads,
                                bool causal) {
#if defined(SDXL_HAS_CUDNN_FRONTEND_SDPA)
    if (query.type() != ScalarType::Float16 || key.type() != ScalarType::Float16 ||
        value.type() != ScalarType::Float16 || output.type() != ScalarType::Float16 ||
        query.rank() != 3 || key.rank() != 3 || value.shape() != key.shape() ||
        output.shape() != query.shape() || query.size(0) != key.size(0) ||
        query.size(2) != key.size(2) || heads == 0 || query.size(2) % heads != 0) {
        throw CudaError("cuDNN Frontend SDPA received incompatible tensors");
    }
    const std::size_t head_dimension = query.size(2) / heads;
    const std::string key_string = plan_key(
        query.size(0), query.size(1), key.size(1), heads, head_dimension, causal);

    std::shared_ptr<CudnnSdpaPlan> plan;
    {
        std::lock_guard lock(runtime.state()->sdpa_mutex);
        const auto found = runtime.state()->sdpa_plans.find(key_string);
        if (found != runtime.state()->sdpa_plans.end()) {
            plan = std::static_pointer_cast<CudnnSdpaPlan>(found->second);
        } else {
            plan = build_plan(runtime, query.size(0), query.size(1), key.size(1),
                              heads, head_dimension, causal);
            runtime.state()->sdpa_plans.emplace(key_string, plan);
        }
    }

    std::unordered_map<fe::graph::Tensor_attributes::uid_t, void*> variant_pack{
        {kQueryUid, const_cast<void*>(query.data())},
        {kKeyUid, const_cast<void*>(key.data())},
        {kValueUid, const_cast<void*>(value.data())},
        {kOutputUid, output.data()}
    };
    std::lock_guard lock(runtime.state()->cudnn_mutex);
    void* workspace = runtime.state()->ensure_cudnn_workspace(plan->workspace_bytes);
    auto status = plan->graph->execute(runtime.cudnn(), variant_pack, workspace);
    if (status.is_bad()) {
        throw CudaError("cuDNN Frontend SDPA execution failed: " + status.get_message());
    }
#else
    (void)runtime; (void)query; (void)key; (void)value; (void)output;
    (void)heads; (void)causal;
    throw CudaError(
        "cudnn-sdpa was requested, but cudnn_frontend.h was not available when the project was built");
#endif
}

} // namespace sdxl::cuda
