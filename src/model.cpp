#include "sdxl/model.hpp"

#include <sstream>

namespace sdxl {

ModuleNode::ModuleNode(std::string name, std::string type)
    : name_(std::move(name)), type_(std::move(type)), full_name_(name_) {}

ModuleNode& ModuleNode::child(std::string name, std::string type) {
    auto node = std::make_unique<ModuleNode>(std::move(name), std::move(type));
    node->set_parent(this);
    children_.push_back(std::move(node));
    return *children_.back();
}

ParameterSlot& ModuleNode::parameter(std::string local_name,
                                     std::vector<std::uint64_t> expected_shape,
                                     bool required) {
    std::string logical = full_name_.empty() ? std::move(local_name)
                                             : full_name_ + "." + local_name;
    parameters_.push_back(ParameterSlot{std::move(logical), std::move(expected_shape), required, std::nullopt, std::nullopt});
    return parameters_.back();
}

void ModuleNode::attribute(std::string key, std::string value) {
    attributes_.insert_or_assign(std::move(key), std::move(value));
}

void ModuleNode::set_parent(ModuleNode* parent) {
    parent_ = parent;
    refresh_full_names_recursive();
}

void ModuleNode::refresh_full_names_recursive() {
    if (parent_ == nullptr || parent_->full_name_.empty()) {
        full_name_ = name_;
    } else if (name_.empty()) {
        full_name_ = parent_->full_name_;
    } else {
        full_name_ = parent_->full_name_ + "." + name_;
    }

    for (auto& child : children_) {
        child->parent_ = this;
        child->refresh_full_names_recursive();
    }
}

ModelGraph::ModelGraph() : root_("", "SDXLPipeline") {}

void ModelGraph::rebuild_index() {
    index_.clear();
    root_.refresh_full_names_recursive();
    index_recursive(root_, index_);
}

ParameterSlot* ModelGraph::find_parameter(std::string_view logical_name) noexcept {
    const auto it = index_.find(std::string(logical_name));
    return it == index_.end() ? nullptr : it->second;
}

const ParameterSlot* ModelGraph::find_parameter(std::string_view logical_name) const noexcept {
    const auto it = index_.find(std::string(logical_name));
    return it == index_.end() ? nullptr : it->second;
}

ValidationReport ModelGraph::validate(bool include_issue_details) const {
    ValidationReport report;
    validate_recursive(root_, report, include_issue_details);
    return report;
}

void ModelGraph::index_recursive(ModuleNode& node,
                                 std::unordered_map<std::string, ParameterSlot*>& index) {
    for (auto& parameter : node.parameters_) {
        const auto [_, inserted] = index.emplace(parameter.logical_name, &parameter);
        if (!inserted) {
            throw Error("duplicate logical parameter in model graph: " + parameter.logical_name);
        }
    }
    for (auto& child : node.children_) index_recursive(*child, index);
}

void ModelGraph::validate_recursive(const ModuleNode& node,
                                    ValidationReport& report,
                                    bool include_issue_details) {
    for (const auto& parameter : node.parameters_) {
        ++report.expected;
        if (!parameter.tensor.has_value()) {
            if (parameter.required) {
                ++report.missing;
                if (include_issue_details) {
                    report.issues.push_back({ValidationIssue::Kind::Missing,
                                             parameter.logical_name,
                                             "required tensor is not bound"});
                }
            }
            continue;
        }
        ++report.bound;
        if (parameter.tensor->shape != parameter.expected_shape) {
            ++report.shape_mismatches;
            if (include_issue_details) {
                report.issues.push_back({
                    ValidationIssue::Kind::ShapeMismatch,
                    parameter.logical_name,
                    "expected " + shape_to_string(parameter.expected_shape) +
                        ", loaded " + shape_to_string(parameter.tensor->shape) +
                        " from " + parameter.tensor->source_key
                });
            }
        }
    }
    for (const auto& child : node.children_) {
        validate_recursive(*child, report, include_issue_details);
    }
}

std::string shape_to_string(const std::vector<std::uint64_t>& shape) {
    std::ostringstream stream;
    stream << '[';
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (i) stream << ',';
        stream << shape[i];
    }
    stream << ']';
    return stream.str();
}

} // namespace sdxl
