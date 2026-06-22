#pragma once

#include "sdxl/safetensors.hpp"

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace sdxl {

struct ParameterSlot {
    std::string logical_name;
    std::vector<std::uint64_t> expected_shape;
    bool required = true;
    std::optional<TensorView> tensor;

    [[nodiscard]] bool bound() const noexcept { return tensor.has_value(); }
};

class ModuleNode final {
public:
    ModuleNode(std::string name, std::string type);

    ModuleNode& child(std::string name, std::string type);
    ParameterSlot& parameter(std::string local_name,
                             std::vector<std::uint64_t> expected_shape,
                             bool required = true);
    void attribute(std::string key, std::string value);

    [[nodiscard]] const std::string& name() const noexcept { return name_; }
    [[nodiscard]] const std::string& type() const noexcept { return type_; }
    [[nodiscard]] const std::string& full_name() const noexcept { return full_name_; }
    [[nodiscard]] const std::vector<std::unique_ptr<ModuleNode>>& children() const noexcept { return children_; }
    [[nodiscard]] std::vector<std::unique_ptr<ModuleNode>>& children() noexcept { return children_; }
    [[nodiscard]] const std::vector<ParameterSlot>& parameters() const noexcept { return parameters_; }
    [[nodiscard]] std::vector<ParameterSlot>& parameters() noexcept { return parameters_; }
    [[nodiscard]] const std::map<std::string, std::string>& attributes() const noexcept { return attributes_; }

private:
    friend class ModelGraph;
    void set_parent(ModuleNode* parent);
    void refresh_full_names_recursive();

    std::string name_;
    std::string type_;
    std::string full_name_;
    ModuleNode* parent_ = nullptr;
    std::vector<std::unique_ptr<ModuleNode>> children_;
    std::vector<ParameterSlot> parameters_;
    std::map<std::string, std::string> attributes_;
};

struct ValidationIssue {
    enum class Kind { Missing, ShapeMismatch, Unexpected };
    Kind kind = Kind::Missing;
    std::string logical_name;
    std::string detail;
};

struct ValidationReport {
    std::size_t expected = 0;
    std::size_t bound = 0;
    std::size_t missing = 0;
    std::size_t shape_mismatches = 0;
    std::vector<ValidationIssue> issues;

    [[nodiscard]] bool ok() const noexcept { return missing == 0 && shape_mismatches == 0; }
};

class ModelGraph final {
public:
    ModelGraph();

    [[nodiscard]] ModuleNode& root() noexcept { return root_; }
    [[nodiscard]] const ModuleNode& root() const noexcept { return root_; }

    void rebuild_index();
    [[nodiscard]] ParameterSlot* find_parameter(std::string_view logical_name) noexcept;
    [[nodiscard]] const ParameterSlot* find_parameter(std::string_view logical_name) const noexcept;
    [[nodiscard]] const std::unordered_map<std::string, ParameterSlot*>& parameter_index() const noexcept { return index_; }
    [[nodiscard]] ValidationReport validate(bool include_issue_details = true) const;

private:
    static void index_recursive(ModuleNode& node,
                                std::unordered_map<std::string, ParameterSlot*>& index);
    static void validate_recursive(const ModuleNode& node,
                                   ValidationReport& report,
                                   bool include_issue_details);

    ModuleNode root_;
    std::unordered_map<std::string, ParameterSlot*> index_;
};

[[nodiscard]] std::string shape_to_string(const std::vector<std::uint64_t>& shape);

} // namespace sdxl
