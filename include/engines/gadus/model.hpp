#pragma once

// Gadus canonical chess-structured policy/value network.

#include <cstdint>
#include <utility>
#include <torch/nn.h>
#include "gadus/game.hpp"

namespace gadus {

struct SquareEmbeddingImpl : torch::nn::Module {
	explicit SquareEmbeddingImpl(int channels);
	torch::Tensor forward(torch::Tensor x);

	torch::Tensor values;
};
TORCH_MODULE(SquareEmbedding);

struct ResidualBlockImpl : torch::nn::Module {
	ResidualBlockImpl(int channels, int sequence_depth);
	torch::Tensor forward(torch::Tensor x);
	torch::Tensor forward_with_relation(torch::Tensor x, torch::Tensor coefficient_corrections);
	void fuse_for_inference();
	torch::Tensor relation_matrices() const;
	int relation_groups() const noexcept;

private:
	torch::Tensor forward_impl(torch::Tensor x, const torch::Tensor &coefficient_corrections);

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
	torch::Tensor relation_basis;
	torch::Tensor relation_coefficients;
	torch::Tensor relation_residual;
	torch::Tensor fused_relation;
	torch::Tensor path_balance;
	bool inference_fused_ = false;
	int channels_ = 0;
	int groups_ = 0;
	int group_width_ = 0;
};
TORCH_MODULE(ResidualBlock);

struct ValueHeadImpl : torch::nn::Module {
	ValueHeadImpl(int channels);
	torch::Tensor forward(torch::Tensor features);
	void initialize_output_prior(double mean_value, double output_scale);
	void fuse_for_inference();

private:
	torch::nn::Conv2d dynamic_conv{nullptr};
	torch::nn::Linear dynamic_coefficients{nullptr};
	ResidualBlock block{nullptr};
	torch::nn::Conv2d output_conv{nullptr};
	torch::nn::BatchNorm2d output_norm{nullptr};
	torch::nn::Linear hidden{nullptr};
	torch::nn::Linear output{nullptr};
	bool inference_fused_ = false;
	int relation_groups_ = 0;
};
TORCH_MODULE(ValueHead);

struct ModelImpl : torch::nn::Module {
	ModelImpl(int channels = 128, int blocks = 10);

	std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor x);
	std::pair<torch::Tensor, torch::Tensor> forward_legal(torch::Tensor x,
										 torch::Tensor legal_indices);
	void initialize_output_priors(const torch::Tensor &action_counts, double mean_value,
								  double smoothing_count = 1.0, double output_scale = 0.1);
	void fuse_for_inference();
	int channels() const noexcept;
	int blocks() const noexcept;

	torch::nn::Sequential backbone{nullptr};
	ValueHead value_head{nullptr};

private:
	torch::Tensor policy_logits(torch::Tensor features);
	torch::Tensor policy_features(torch::Tensor features);
	torch::Tensor policy_contexts(torch::Tensor features);
	torch::Tensor policy_action_vectors();
	torch::Tensor value(torch::Tensor features);

	torch::nn::Conv2d backbone_conv{nullptr};
	torch::nn::BatchNorm2d backbone_norm{nullptr};
	SquareEmbedding square_embedding{nullptr};
	torch::nn::Conv2d policy_conv{nullptr};
	torch::nn::BatchNorm2d policy_norm{nullptr};
	torch::nn::Sequential policy_blocks{nullptr};
	torch::nn::Linear policy_source{nullptr};
	torch::nn::Linear policy_global{nullptr};
	torch::Tensor policy_motion_vectors;
	torch::Tensor policy_action_corrections;
	torch::Tensor policy_action_bias;
	bool inference_fused_ = false;
	int channels_ = 0;
	int blocks_ = 0;
};
TORCH_MODULE(Model);

std::int64_t parameter_count(const Model &model);

} // namespace gadus
