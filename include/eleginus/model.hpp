#pragma once

// Trainable incremental Value network.

#include <cstdint>
#include <vector>

#include <torch/nn.h>

#include "eleginus/nnue.hpp"

namespace eleginus {

struct NetworkBatch {
	torch::Tensor features;
	torch::Tensor white_to_move;
	torch::Tensor edge_source;
	torch::Tensor edge_target;
	torch::Tensor edge_geometry;
	torch::Tensor edge_destination;
	torch::Tensor occupancy;
	torch::Tensor material;
};

/// Packs sparse features and the rule-derived control graph for one neural batch.
NetworkBatch encode_feature_batch(
	const std::vector<EncodedFeatures> &positions, const torch::Device &device);

struct SparseEncoderImpl : torch::nn::Module {
	explicit SparseEncoderImpl(int width);
	torch::Tensor accumulate(torch::Tensor features);
	torch::Tensor forward(torch::Tensor features, torch::Tensor white_to_move);

	int width;
	torch::nn::Embedding table{nullptr};
	torch::Tensor bias;
};
TORCH_MODULE(SparseEncoder);

struct ValueNetworkImpl : torch::nn::Module {
	ValueNetworkImpl();
	torch::Tensor control_state(const NetworkBatch &batch, torch::Tensor buckets);
	torch::Tensor forward(const NetworkBatch &batch);

	SparseEncoder encoder{nullptr};
	torch::nn::Embedding control_source{nullptr};
	torch::nn::Embedding control_target{nullptr};
	torch::nn::Embedding control_geometry{nullptr};
	torch::nn::Embedding control_occupancy{nullptr};
	torch::nn::Embedding control_count{nullptr};
	torch::nn::Embedding control_square{nullptr};
	torch::nn::Linear control_local{nullptr};
	torch::nn::Linear attention_key{nullptr};
	torch::nn::Linear attention_value{nullptr};
	torch::nn::Embedding attention_query{nullptr};
	torch::nn::Linear material{nullptr};
	torch::nn::Linear hidden{nullptr};
	torch::nn::Linear bottleneck{nullptr};
	torch::nn::Linear output{nullptr};
};
TORCH_MODULE(ValueNetwork);

struct ModelImpl : torch::nn::Module {
	ModelImpl();

	ValueNetwork value{nullptr};
};
TORCH_MODULE(Model);

/// Copies trainable tensors into a Torch-free immutable CPU evaluator.
CpuValue snapshot_value(const ValueNetwork &model);
/// Restores runtime weights into a trainable LibTorch module.
void restore_value(const ValueNetwork &model, const ValueWeights &weights);

/// Counts scalar parameter elements in one module.
std::int64_t parameter_count(const torch::nn::Module &model);

} // namespace eleginus
