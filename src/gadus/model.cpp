#include "gadus/model.hpp"

namespace gadus {

ResidualBlockImpl::ResidualBlockImpl(int channels) {
	block = register_module(
		"block",
		torch::nn::Sequential(
			torch::nn::Conv2d(
				torch::nn::Conv2dOptions(channels, channels, 3).padding(1).bias(false)),
			torch::nn::BatchNorm2d(channels), torch::nn::ReLU(torch::nn::ReLUOptions(true)),
			torch::nn::Conv2d(
				torch::nn::Conv2dOptions(channels, channels, 3).padding(1).bias(false)),
			torch::nn::BatchNorm2d(channels)));
}

torch::Tensor ResidualBlockImpl::forward(torch::Tensor x) {
	return torch::relu(x + block->forward(x));
}

ModelImpl::ModelImpl(int channels, int blocks) : channels_(channels), blocks_(blocks) {
	backbone = register_module("backbone", torch::nn::Sequential());
	backbone->push_back(torch::nn::Conv2d(
		torch::nn::Conv2dOptions(kStatePlanes, channels_, 3).padding(1).bias(false)));
	backbone->push_back(torch::nn::BatchNorm2d(channels_));
	backbone->push_back(torch::nn::ReLU(torch::nn::ReLUOptions(true)));
	for (int index = 0; index < blocks_; ++index) {
		backbone->push_back(ResidualBlock(channels_));
	}

	policy_head = register_module(
		"policy_head",
		torch::nn::Sequential(
			torch::nn::Conv2d(torch::nn::Conv2dOptions(channels_, 32, 1).bias(false)),
			torch::nn::BatchNorm2d(32), torch::nn::ReLU(torch::nn::ReLUOptions(true)),
			torch::nn::Flatten(), torch::nn::Linear(32 * 8 * 8, kActionSize)));

	value_head = register_module(
		"value_head", torch::nn::Sequential(
						  torch::nn::Conv2d(torch::nn::Conv2dOptions(channels_, 32, 1).bias(false)),
						  torch::nn::BatchNorm2d(32), torch::nn::ReLU(torch::nn::ReLUOptions(true)),
						  torch::nn::Flatten(), torch::nn::Linear(32 * 8 * 8, 256),
						  torch::nn::ReLU(torch::nn::ReLUOptions(true)), torch::nn::Linear(256, 1),
						  torch::nn::Tanh()));
}

std::pair<torch::Tensor, torch::Tensor> ModelImpl::forward(torch::Tensor x) {
	auto features = backbone->forward(x);
	return {policy_head->forward(features), value_head->forward(features)};
}

int ModelImpl::channels() const noexcept { return channels_; }
int ModelImpl::blocks() const noexcept { return blocks_; }

std::int64_t parameter_count(const Model &model) {
	std::int64_t count = 0;
	for (const auto &parameter : model->parameters()) {
		count += parameter.numel();
	}
	return count;
}

} // namespace gadus
