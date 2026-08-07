// Implements Gadus's residual convolutional policy/value network.

#include "gadus/model.hpp"
#include <stdexcept>

namespace gadus {

namespace {

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
		torch::nn::Conv2d(torch::nn::Conv2dOptions(channels_, 32, 1).bias(false));
	policy_norm = torch::nn::BatchNorm2d(32);
	policy_projection = torch::nn::Linear(32 * 8 * 8, kActionSize);
	policy_head = register_module(
		"policy_head",
		torch::nn::Sequential(policy_conv, policy_norm,
							  torch::nn::ReLU(torch::nn::ReLUOptions(true)),
							  torch::nn::Flatten(), policy_projection));

	value_conv =
		torch::nn::Conv2d(torch::nn::Conv2dOptions(channels_, 32, 1).bias(false));
	value_norm = torch::nn::BatchNorm2d(32);
	value_hidden = torch::nn::Linear(32 * 8 * 8, 256);
	value_output = torch::nn::Linear(256, 1);
	value_head = register_module(
		"value_head", torch::nn::Sequential(
						  value_conv, value_norm, torch::nn::ReLU(torch::nn::ReLUOptions(true)),
						  torch::nn::Flatten(), value_hidden,
						  torch::nn::ReLU(torch::nn::ReLUOptions(true)), value_output,
						  torch::nn::Tanh()));
}

// Evaluate the shared features once, then return action logits and bounded V(s).
std::pair<torch::Tensor, torch::Tensor> ModelImpl::forward(torch::Tensor x) {
	auto features = backbone->forward(x);
	return {policy_head->forward(features), value_head->forward(features)};
}

// Apply the existing policy projection only to requested action rows, avoiding the 4672-way
// matrix product during legal-move inference while preserving checkpoint parameters exactly.
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
	auto policy_features = policy_conv->forward(features);
	if (!inference_fused_) {
		policy_features = policy_norm->forward(policy_features);
	}
	policy_features = torch::relu(policy_features).flatten(1);

	auto flat_indices = legal_indices.reshape({-1});
	auto selected_weights = policy_projection->weight.index_select(0, flat_indices)
								.reshape({legal_indices.size(0), legal_indices.size(1),
										  policy_features.size(1)});
	auto legal_logits =
		torch::bmm(selected_weights, policy_features.unsqueeze(2)).squeeze(2);
	if (policy_projection->bias.defined()) {
		legal_logits = legal_logits +
			policy_projection->bias.index_select(0, flat_indices).reshape_as(legal_indices);
	}
	return {legal_logits, value_head->forward(features)};
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
	policy_head = replace_module(
		"policy_head", torch::nn::Sequential(
							   policy_conv, torch::nn::ReLU(torch::nn::ReLUOptions(true)),
							   torch::nn::Flatten(), policy_projection));
	policy_norm = nullptr;

	value_conv = fuse_conv_bn(value_conv, value_norm);
	value_head = replace_module(
		"value_head", torch::nn::Sequential(
						  value_conv, torch::nn::ReLU(torch::nn::ReLUOptions(true)),
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
