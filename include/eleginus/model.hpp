#pragma once

// Trainable NNUE-style Value network plus custom incremental CPU inference.

#include <cstdint>
#include <vector>

#include <torch/nn.h>

#include "eleginus/nnue.hpp"

namespace eleginus {

/// Packs encoded positions as [N, 2, 34] indices and [N] side-to-move flags.
std::pair<torch::Tensor, torch::Tensor>
encode_feature_batch(const std::vector<EncodedFeatures> &positions, const torch::Device &device);

struct SparseEncoderImpl : torch::nn::Module {
	explicit SparseEncoderImpl(int width);
	torch::Tensor forward(torch::Tensor features, torch::Tensor white_to_move);

	int width;
	torch::nn::Embedding table{nullptr};
};
TORCH_MODULE(SparseEncoder);

struct ValueNetworkImpl : torch::nn::Module {
	ValueNetworkImpl();
	torch::Tensor hidden_state(torch::Tensor features, torch::Tensor white_to_move);
	torch::Tensor forward(torch::Tensor features, torch::Tensor white_to_move);

	SparseEncoder encoder{nullptr};
	torch::nn::Linear hidden{nullptr};
	torch::nn::Linear bottleneck{nullptr};
	torch::nn::Linear output{nullptr};
};
TORCH_MODULE(ValueNetwork);

/// Copies trainable tensors into a Torch-free immutable CPU evaluator.
CpuValue snapshot_value(const ValueNetwork &model);
/// Restores runtime weights into a trainable LibTorch module.
void restore_value(const ValueNetwork &model, const ValueWeights &weights);

/// Counts scalar parameter elements in one module.
std::int64_t parameter_count(const torch::nn::Module &model);

} // namespace eleginus
