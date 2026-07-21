#pragma once

#include <utility>

#include <torch/torch.h>

#include "gadus/game.hpp"

namespace gadus {

struct ResidualBlockImpl : torch::nn::Module {
	explicit ResidualBlockImpl(int channels);
	torch::Tensor forward(torch::Tensor x);

	torch::nn::Sequential block{nullptr};
};
TORCH_MODULE(ResidualBlock);

struct ModelImpl : torch::nn::Module {
	ModelImpl(int channels = 128, int blocks = 10);

	std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor x);
	int channels() const noexcept;
	int blocks() const noexcept;

	torch::nn::Sequential backbone{nullptr};
	torch::nn::Sequential policy_head{nullptr};
	torch::nn::Sequential value_head{nullptr};

	private:
	int channels_;
	int blocks_;
};
TORCH_MODULE(Model);

std::int64_t parameter_count(const Model &model);

} // namespace gadus
