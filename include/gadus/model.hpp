#pragma once

// Gadus residual convolutional policy/value network.

#include <utility>
#include <torch/nn.h>
#include "gadus/game.hpp"

namespace gadus {

struct ResidualBlockImpl : torch::nn::Module {
	/// Builds a two-convolution residual transform that preserves board shape and channels.
	explicit ResidualBlockImpl(int channels);
	/// Applies ReLU(x + F(x)) so gradients can bypass the learned residual branch.
	torch::Tensor forward(torch::Tensor x);
	/// Folds both evaluation BatchNorm transforms into their preceding convolutions.
	void fuse_for_inference();

	torch::nn::Sequential block{nullptr};

private:
	torch::nn::Conv2d conv1{nullptr};
	torch::nn::BatchNorm2d norm1{nullptr};
	torch::nn::Conv2d conv2{nullptr};
	torch::nn::BatchNorm2d norm2{nullptr};
	bool inference_fused_ = false;
};
TORCH_MODULE(ResidualBlock);

struct ModelImpl : torch::nn::Module {
	/// Builds the 18-plane residual backbone and independent policy/value heads.
	ModelImpl(int channels = 128, int blocks = 10);

	/// Returns policy logits [N, 4672] and side-to-move value [N, 1].
	std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor x);
	/// Returns logits only for legal_indices [N, L] and the same side-to-move value [N, 1].
	std::pair<torch::Tensor, torch::Tensor> forward_legal(torch::Tensor x,
													 torch::Tensor legal_indices);
	/// Replaces Conv-BN pairs with equivalent evaluation-only fused convolutions.
	void fuse_for_inference();
	/// Returns the residual trunk width stored in the checkpoint architecture descriptor.
	int channels() const noexcept;
	/// Returns the number of residual blocks stored in the checkpoint descriptor.
	int blocks() const noexcept;

	torch::nn::Sequential backbone{nullptr};
	torch::nn::Sequential value_head{nullptr};

private:
	torch::Tensor policy_logits(torch::Tensor features);
	torch::Tensor policy_planes(torch::Tensor features);
	torch::Tensor policy_features(torch::Tensor features);
	torch::Tensor value(torch::Tensor features);

	// Shared handles expose layers that remain registered through their Sequential containers.
	torch::nn::Conv2d backbone_conv{nullptr};
	torch::nn::BatchNorm2d backbone_norm{nullptr};
	torch::nn::Conv2d policy_conv{nullptr};
	torch::nn::BatchNorm2d policy_norm{nullptr};
	torch::nn::Sequential policy_blocks{nullptr};
	torch::nn::Conv2d policy_output{nullptr};
	torch::Tensor policy_position;
	torch::Tensor policy_action_bias;
	ResidualBlock value_block{nullptr};
	torch::nn::Conv2d value_conv{nullptr};
	torch::nn::BatchNorm2d value_norm{nullptr};
	torch::nn::Linear value_hidden{nullptr};
	torch::nn::Linear value_output{nullptr};
	bool inference_fused_ = false;
	int channels_;
	int blocks_;
};
TORCH_MODULE(Model);

/// Counts all trainable and non-trainable model parameter elements.
std::int64_t parameter_count(const Model &model);

} // namespace gadus
