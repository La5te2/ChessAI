#pragma once

// Gadus residual convolutional policy/value network.

#include <utility>
#include <torch/torch.h>
#include "gadus/game.hpp"

namespace gadus {

struct ResidualBlockImpl : torch::nn::Module {
	/// Builds a two-convolution residual transform that preserves board shape and channels.
	explicit ResidualBlockImpl(int channels);
	/// Applies ReLU(x + F(x)) so gradients can bypass the learned residual branch.
	torch::Tensor forward(torch::Tensor x);

	torch::nn::Sequential block{nullptr};
};
TORCH_MODULE(ResidualBlock);

struct ModelImpl : torch::nn::Module {
	/// Builds the 18-plane residual backbone and independent policy/value heads.
	ModelImpl(int channels = 128, int blocks = 10);

	/// Returns policy logits [N, 4672] and side-to-move value [N, 1].
	std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor x);
	/// Returns the residual trunk width stored in the checkpoint architecture descriptor.
	int channels() const noexcept;
	/// Returns the number of residual blocks stored in the checkpoint descriptor.
	int blocks() const noexcept;

	torch::nn::Sequential backbone{nullptr};
	torch::nn::Sequential policy_head{nullptr};
	torch::nn::Sequential value_head{nullptr};

	private:
	int channels_;
	int blocks_;
};
TORCH_MODULE(Model);

/// Counts all trainable and non-trainable model parameter elements.
std::int64_t parameter_count(const Model &model);

} // namespace gadus
