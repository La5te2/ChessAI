#pragma once

// Gadus canonical chess-structured policy/value network.

#include "gadus/game.hpp"
#include <cstdint>
#include <torch/nn.h>
#include <utility>

namespace gadus {

struct SquareEmbeddingImpl : torch::nn::Module {
	explicit SquareEmbeddingImpl(int channels);
	torch::Tensor forward(torch::Tensor x);

	torch::Tensor values;
};
TORCH_MODULE(SquareEmbedding);

struct ResidualBlockImpl : torch::nn::Module {
	ResidualBlockImpl(int channels, int sequence_depth, bool zero_output_scale = false);
	torch::Tensor forward(torch::Tensor x);
	void fuse_for_inference();
	void project_relation_residual();
	torch::Tensor relation_matrices() const;
	void save(torch::serialize::OutputArchive &archive) const override;
	void load(torch::serialize::InputArchive &archive) override;

private:
	torch::nn::Conv2d down_conv{nullptr};
	torch::nn::BatchNorm2d down_norm{nullptr};
	torch::nn::Conv2d local_conv3{nullptr};
	torch::nn::BatchNorm2d local_norm3{nullptr};
	torch::nn::Conv2d local_conv1{nullptr};
	torch::nn::BatchNorm2d local_norm1{nullptr};
	torch::nn::BatchNorm2d local_identity_norm{nullptr};
	torch::nn::Conv2d fused_local{nullptr};
	torch::nn::BatchNorm2d relation_norm{nullptr};
	torch::nn::Conv2d up_conv{nullptr};
	torch::nn::BatchNorm2d up_norm{nullptr};
	torch::Tensor displacement_index;
	torch::Tensor displacement_scale;
	torch::Tensor displacement_support;
	torch::Tensor relation_coefficients;
	torch::Tensor relation_residual;
	torch::Tensor fused_relation;
	torch::Tensor fused_relation_scale;
	torch::Tensor path_balance;
	bool inference_fused_ = false;
	int channels_ = 0;
	int groups_ = 0;
	int group_width_ = 0;
};
TORCH_MODULE(ResidualBlock);

struct ModelImpl : torch::nn::Module {
	ModelImpl(int channels = 128, int blocks = 12);

	std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor x);
	std::pair<torch::Tensor, torch::Tensor> forward_legal(torch::Tensor x, torch::Tensor legal_indices);
	void initialize_output_priors(const torch::Tensor &action_counts, double mean_value, double smoothing_count = 1.0, double output_scale = 0.1);
	void project_relation_residuals();
	void fuse_for_inference();
	int channels() const noexcept;
	int blocks() const noexcept;
	void save(torch::serialize::OutputArchive &archive) const override;
	void load(torch::serialize::InputArchive &archive) override;

	torch::nn::Sequential backbone{nullptr};
	torch::nn::Sequential value_head{nullptr};

private:
	torch::Tensor trunk_features(torch::Tensor x);
	torch::Tensor policy_logits(torch::Tensor features);
	torch::Tensor policy_logits_for_indices(const torch::Tensor &features, const torch::Tensor &compact_indices);
	torch::Tensor centered_policy_bias() const;
	torch::Tensor policy_features(torch::Tensor features);
	torch::Tensor value(torch::Tensor features);

	torch::nn::Conv2d backbone_conv{nullptr};
	torch::nn::BatchNorm2d backbone_norm{nullptr};
	SquareEmbedding square_embedding{nullptr};
	torch::nn::Conv2d policy_conv{nullptr};
	torch::nn::BatchNorm2d policy_norm{nullptr};
	torch::nn::Sequential policy_blocks{nullptr};
	torch::nn::Linear policy_source{nullptr};
	torch::Tensor policy_action_bias;
	ResidualBlock value_block{nullptr};
	ResidualBlock value_block_2{nullptr};
	torch::nn::Conv2d value_conv{nullptr};
	torch::nn::BatchNorm2d value_norm{nullptr};
	torch::nn::Linear value_hidden{nullptr};
	torch::nn::Linear value_output{nullptr};
	torch::Tensor compact_action_sources;
	torch::Tensor compact_action_patterns;
	bool inference_fused_ = false;
	int channels_ = 0;
	int blocks_ = 0;
};
TORCH_MODULE(Model);

std::int64_t parameter_count(const Model &model);

} // namespace gadus
