// Implements Gadus's residual convolutional policy/value network.

#include "gadus/model.hpp"
#include "gadus/precision.hpp"
#include <stdexcept>

namespace gadus {

namespace {

constexpr int kPolicyChannels = 128;
constexpr int kPolicyBlocks = 2;
constexpr int kValueChannels = 48;
constexpr int kValueHidden = 512;

// Fold y = gamma * (conv(x) - mean) / sqrt(var + eps) + beta into one biased convolution.
torch::nn::Conv2d fuse_conv_bn(const torch::nn::Conv2d &conv,
							   const torch::nn::BatchNorm2d &norm) {
	if (!conv || !norm || norm->is_training()) {
		throw std::logic_error("Conv-BN fusion requires initialized evaluation modules");
	}
	auto options = torch::nn::Conv2dOptions(
		conv->options.in_channels(), conv->options.out_channels(),
		conv->options.kernel_size())
			   .stride(conv->options.stride())
			   .padding(conv->options.padding())
			   .dilation(conv->options.dilation())
			   .groups(conv->options.groups())
			   .bias(true)
			   .padding_mode(conv->options.padding_mode());
	auto fused = torch::nn::Conv2d(options);
	fused->to(conv->weight.device(), conv->weight.scalar_type());

	torch::NoGradGuard guard;
	const auto scale = norm->weight / torch::sqrt(norm->running_var + norm->options.eps());
	const auto source_bias = conv->bias.defined()
							 ? conv->bias
							 : torch::zeros_like(norm->running_mean);
	fused->weight.copy_(conv->weight * scale.reshape({-1, 1, 1, 1}));
	fused->bias.copy_(norm->bias + (source_bias - norm->running_mean) * scale);
	fused->eval();
	return fused;
}

} // namespace

// Construct F(x) as Conv-BN-ReLU-Conv-BN while leaving the skip path untouched.
ResidualBlockImpl::ResidualBlockImpl(int channels) {
	conv1 = torch::nn::Conv2d(
		torch::nn::Conv2dOptions(channels, channels, 3).padding(1).bias(false));
	norm1 = torch::nn::BatchNorm2d(channels);
	conv2 = torch::nn::Conv2d(
		torch::nn::Conv2dOptions(channels, channels, 3).padding(1).bias(false));
	norm2 = torch::nn::BatchNorm2d(channels);
	block = register_module(
		"block", torch::nn::Sequential(conv1, norm1,
										 torch::nn::ReLU(torch::nn::ReLUOptions(true)),
										 conv2, norm2));
}

// Residual addition gives y = ReLU(x + F(x)) and preserves the board tensor shape.
torch::Tensor ResidualBlockImpl::forward(torch::Tensor x) {
	return torch::relu(x + block->forward(x));
}

// Remove four normalization tensors and two normalization kernels from inference.
void ResidualBlockImpl::fuse_for_inference() {
	if (inference_fused_) {
		return;
	}
	conv1 = fuse_conv_bn(conv1, norm1);
	conv2 = fuse_conv_bn(conv2, norm2);
	block = replace_module(
		"block", torch::nn::Sequential(conv1,
										 torch::nn::ReLU(torch::nn::ReLUOptions(true)), conv2));
	norm1 = nullptr;
	norm2 = nullptr;
	inference_fused_ = true;
}

// Build one shared residual trunk and separate policy/value readouts.
ModelImpl::ModelImpl(int channels, int blocks) : channels_(channels), blocks_(blocks) {
	backbone = register_module("backbone", torch::nn::Sequential());
	backbone_conv = torch::nn::Conv2d(
		torch::nn::Conv2dOptions(kStatePlanes, channels_, 3).padding(1).bias(false));
	backbone_norm = torch::nn::BatchNorm2d(channels_);
	backbone->push_back(backbone_conv);
	backbone->push_back(backbone_norm);
	backbone->push_back(torch::nn::ReLU(torch::nn::ReLUOptions(true)));
	for (int index = 0; index < blocks_; ++index) {
		backbone->push_back(ResidualBlock(channels_));
	}

	policy_conv =
		torch::nn::Conv2d(torch::nn::Conv2dOptions(channels_, kPolicyChannels, 1).bias(false));
	policy_norm = torch::nn::BatchNorm2d(kPolicyChannels);
	register_module("policy_conv", policy_conv);
	register_module("policy_norm", policy_norm);
	policy_blocks = register_module("policy_blocks", torch::nn::Sequential());
	for (int index = 0; index < kPolicyBlocks; ++index) {
		policy_blocks->push_back(ResidualBlock(kPolicyChannels));
	}
	policy_output = register_module(
		"policy_output",
		torch::nn::Conv2d(torch::nn::Conv2dOptions(kPolicyChannels, kPolicyPlanes, 1)
							  .bias(false)));
	policy_position = register_parameter(
		"policy_position", torch::empty({1, kPolicyChannels, 8, 8}));
	policy_action_bias = register_parameter(
		"policy_action_bias", torch::zeros({1, kPolicyPlanes, 8, 8}));
	torch::nn::init::normal_(policy_position, 0.0, 0.02);

	value_block = ResidualBlock(channels_);
	value_conv =
		torch::nn::Conv2d(torch::nn::Conv2dOptions(channels_, kValueChannels, 1).bias(false));
	value_norm = torch::nn::BatchNorm2d(kValueChannels);
	value_hidden = torch::nn::Linear(kValueChannels * 8 * 8, kValueHidden);
	value_output = torch::nn::Linear(kValueHidden, 1);
	{
		torch::NoGradGuard guard;
		value_output->weight.zero_();
		value_output->bias.zero_();
	}
	value_head = register_module(
		"value_head", torch::nn::Sequential(
						  value_block, value_conv, value_norm,
						  torch::nn::ReLU(torch::nn::ReLUOptions(true)),
						  torch::nn::Flatten(), value_hidden,
						  torch::nn::ReLU(torch::nn::ReLUOptions(true)), value_output,
						  torch::nn::Tanh()));
}

// Evaluate the shared features once, then return action logits and bounded V(s).
std::pair<torch::Tensor, torch::Tensor> ModelImpl::forward(torch::Tensor x) {
	auto features = backbone->forward(x);
	return {policy_logits(features), value(features)};
}

// Convert source-major action indices q * 73 + p to plane-major NCHW offsets p * 64 + q.
std::pair<torch::Tensor, torch::Tensor>
ModelImpl::forward_legal(torch::Tensor x, torch::Tensor legal_indices) {
	if (legal_indices.dim() != 2 || legal_indices.size(0) != x.size(0)) {
		throw std::invalid_argument(
			"legal_indices must have shape [batch, legal_width]");
	}
	if (legal_indices.scalar_type() != torch::kInt64 || legal_indices.device() != x.device()) {
		throw std::invalid_argument(
			"legal_indices must be int64 and reside on the input device");
	}

	auto features = backbone->forward(x);
	auto source_squares = torch::floor_divide(legal_indices, kPolicyPlanes);
	auto patterns = torch::remainder(legal_indices, kPolicyPlanes);
	auto compact_features = policy_features(features);
	auto source_features = compact_features.flatten(2)
						   .transpose(1, 2)
						   .gather(1, source_squares.unsqueeze(2).expand(
									  {-1, -1, kPolicyChannels}));
	auto selected_weights = policy_output->weight.flatten(1)
							.index_select(0, patterns.reshape({-1}))
							.view({legal_indices.size(0), legal_indices.size(1),
								   kPolicyChannels});
	auto legal_logits = (source_features * selected_weights).sum(2);
	auto plane_offsets = patterns * 64 + source_squares;
	legal_logits = legal_logits + policy_action_bias.flatten(1)
								 .expand({legal_indices.size(0), -1})
								 .gather(1, plane_offsets);
	return {legal_logits, value(features)};
}

// Keep the bounded Value readout in FP32 even when the shared trunk uses CUDA BF16.
torch::Tensor ModelImpl::value(torch::Tensor features) {
	AutocastGuard fp32(ComputePrecision::Fp32, features.device());
	return value_head->forward(features.to(torch::kFloat32));
}

// Produce position-aware Policy features before the final action-pattern projection.
torch::Tensor ModelImpl::policy_features(torch::Tensor features) {
	auto policy_features = policy_conv->forward(features);
	if (!inference_fused_) {
		policy_features = policy_norm->forward(policy_features);
	}
	policy_features = torch::relu(policy_features + policy_position);
	return policy_blocks->forward(policy_features);
}

// Produce one action-pattern channel per source square in NCHW board order.
torch::Tensor ModelImpl::policy_planes(torch::Tensor features) {
	return policy_output->forward(policy_features(features)) + policy_action_bias;
}

// Reorder [pattern, rank, file] to the source-major action index 73 * square + pattern.
torch::Tensor ModelImpl::policy_logits(torch::Tensor features) {
	auto planes = policy_planes(features);
	return planes.permute({0, 2, 3, 1}).contiguous().view({planes.size(0), kActionSize});
}

// Rebuild only the in-memory inference graph; checkpoint construction and training stay unfused.
void ModelImpl::fuse_for_inference() {
	if (inference_fused_) {
		return;
	}
	if (is_training()) {
		throw std::logic_error("Gadus model must be in evaluation mode before fusion");
	}

	backbone_conv = fuse_conv_bn(backbone_conv, backbone_norm);
	auto fused_backbone = torch::nn::Sequential(
		backbone_conv, torch::nn::ReLU(torch::nn::ReLUOptions(true)));
	for (int index = 0; index < blocks_; ++index) {
		auto residual = backbone->ptr<ResidualBlockImpl>(static_cast<std::size_t>(index + 3));
		residual->fuse_for_inference();
		fused_backbone->push_back(ResidualBlock(residual));
	}
	backbone = replace_module("backbone", fused_backbone);
	backbone_norm = nullptr;

	policy_conv = fuse_conv_bn(policy_conv, policy_norm);
	replace_module("policy_conv", policy_conv);
	policy_norm = nullptr;
	for (int index = 0; index < kPolicyBlocks; ++index) {
		policy_blocks->ptr<ResidualBlockImpl>(static_cast<std::size_t>(index))
			->fuse_for_inference();
	}

	value_block->fuse_for_inference();
	value_conv = fuse_conv_bn(value_conv, value_norm);
	value_head = replace_module(
		"value_head", torch::nn::Sequential(
						  value_block, value_conv,
						  torch::nn::ReLU(torch::nn::ReLUOptions(true)),
						  torch::nn::Flatten(), value_hidden,
						  torch::nn::ReLU(torch::nn::ReLUOptions(true)), value_output,
						  torch::nn::Tanh()));
	value_norm = nullptr;
	inference_fused_ = true;
}

// Expose the checkpoint-defining channel width.
int ModelImpl::channels() const noexcept { return channels_; }
// Expose the checkpoint-defining residual depth.
int ModelImpl::blocks() const noexcept { return blocks_; }

// Sum tensor element counts rather than serialized bytes or optimizer state.
std::int64_t parameter_count(const Model &model) {
	std::int64_t count = 0;
	for (const auto &parameter : model->parameters()) {
		count += parameter.numel();
	}
	return count;
}

} // namespace gadus
