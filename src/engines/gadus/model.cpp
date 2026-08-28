// Implements Gadus's canonical chess-structured policy/value network.

#include "gadus/model.hpp"
#include "gadus/precision.hpp"
#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <numeric>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace gadus {

namespace {

using torch::indexing::Slice;

constexpr int kMaximumRelationGroups = 8;
constexpr int kPolicyChannels = 128;
constexpr int kPolicyBlocks = 2;
constexpr int kValueChannels = 48;
constexpr int kValueHidden = 512;
constexpr double kValueClipEpsilon = 1e-4;
constexpr int kDisplacementWidth = 15;
constexpr int kDisplacementCount = kDisplacementWidth * kDisplacementWidth;

bool contains_name(std::initializer_list<std::string_view> names, const std::string &name) {
	return std::find(names.begin(), names.end(), name) != names.end();
}

void save_without_buffers(const torch::nn::Module &module, torch::serialize::OutputArchive &archive, std::initializer_list<std::string_view> omitted) {
	for (const auto &parameter : module.named_parameters(false)) {
		archive.write(parameter.key(), *parameter, false);
	}
	for (const auto &buffer : module.named_buffers(false)) {
		if (!contains_name(omitted, buffer.key())) {
			archive.write(buffer.key(), *buffer, true);
		}
	}
	for (const auto &child : module.named_children()) {
		torch::serialize::OutputArchive child_archive;
		child.value()->save(child_archive);
		archive.write(child.key(), child_archive);
	}
}

void load_without_buffers(torch::nn::Module &module, torch::serialize::InputArchive &archive, std::initializer_list<std::string_view> omitted) {
	for (auto &parameter : module.named_parameters(false)) {
		archive.read(parameter.key(), *parameter, false);
	}
	for (auto &buffer : module.named_buffers(false)) {
		if (!contains_name(omitted, buffer.key())) {
			archive.read(buffer.key(), *buffer, true);
		}
	}
	for (const auto &child : module.named_children()) {
		torch::serialize::InputArchive child_archive;
		archive.read(child.key(), child_archive);
		child.value()->load(child_archive);
	}
}

int displacement_slot(int rank_delta, int file_delta) {
	return (rank_delta + 7) * kDisplacementWidth + file_delta + 7;
}

std::vector<std::pair<int, int>> initial_relation_offsets(int group) {
	switch (group) {
	case 0:
		return {{0, 0}};
	case 1:
		return {{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 1}, {1, -1}, {1, 0}, {1, 1}};
	case 2:
		return {{-2, -1}, {-2, 1}, {-1, -2}, {-1, 2}, {1, -2}, {1, 2}, {2, -1}, {2, 1}};
	case 3:
		return {{1, 0}};
	case 4:
		return {{1, -1}, {1, 1}};
	default:
		break;
	}

	std::vector<std::pair<int, int>> offsets;
	for (int rank_delta = -7; rank_delta <= 7; ++rank_delta) {
		for (int file_delta = -7; file_delta <= 7; ++file_delta) {
			const bool global = group == 5;
			const bool rook = group == 6 && (rank_delta == 0 || file_delta == 0) && (rank_delta != 0 || file_delta != 0);
			const bool bishop = group == 7 && std::abs(rank_delta) == std::abs(file_delta) && rank_delta != 0;
			if (global || rook || bishop) {
				offsets.emplace_back(rank_delta, file_delta);
			}
		}
	}
	if (offsets.empty()) {
		throw std::out_of_range("Gadus initial relation group");
	}
	return offsets;
}

struct RelationGeometry {
	torch::Tensor displacement_index;
	torch::Tensor displacement_scale;
	torch::Tensor displacement_support;
};

RelationGeometry build_relation_geometry() {
	auto index = torch::empty({kBoardSquares, kBoardSquares}, torch::kInt64);
	auto scale = torch::empty({kBoardSquares, kBoardSquares}, torch::kFloat32);
	auto support = torch::empty({kDisplacementCount}, torch::kFloat32);
	auto index_data = index.accessor<std::int64_t, 2>();
	auto scale_data = scale.accessor<float, 2>();
	auto support_data = support.accessor<float, 1>();

	for (int rank_delta = -7; rank_delta <= 7; ++rank_delta) {
		for (int file_delta = -7; file_delta <= 7; ++file_delta) {
			const auto count = (8 - std::abs(rank_delta)) * (8 - std::abs(file_delta));
			support_data[displacement_slot(rank_delta, file_delta)] = static_cast<float>(count);
		}
	}
	for (int target = 0; target < kBoardSquares; ++target) {
		const int target_rank = target / 8;
		const int target_file = target % 8;
		for (int source = 0; source < kBoardSquares; ++source) {
			const int source_rank = source / 8;
			const int source_file = source % 8;
			const int rank_delta = target_rank - source_rank;
			const int file_delta = target_file - source_file;
			const int slot = displacement_slot(rank_delta, file_delta);
			index_data[target][source] = slot;
			scale_data[target][source] = 1.0F / std::sqrt(support_data[slot]);
		}
	}
	return {index, scale, support};
}

const RelationGeometry &relation_geometry() {
	static const RelationGeometry geometry = build_relation_geometry();
	return geometry;
}

torch::nn::Conv2d fuse_conv_bn(const torch::nn::Conv2d &conv, const torch::nn::BatchNorm2d &norm) {
	if (!conv || !norm || norm->is_training()) {
		throw std::logic_error("Conv-BN fusion requires initialized evaluation modules");
	}
	auto options = torch::nn::Conv2dOptions(conv->options.in_channels(), conv->options.out_channels(), conv->options.kernel_size())
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
	const auto source_bias = conv->bias.defined() ? conv->bias : torch::zeros_like(norm->running_mean);
	fused->weight.copy_(conv->weight * scale.reshape({-1, 1, 1, 1}));
	fused->bias.copy_(norm->bias + (source_bias - norm->running_mean) * scale);
	fused->eval();
	return fused;
}

torch::nn::Conv2d fuse_local_branches(const torch::nn::Conv2d &conv3, const torch::nn::BatchNorm2d &norm3, const torch::nn::Conv2d &conv1, const torch::nn::BatchNorm2d &norm1,
    const torch::nn::BatchNorm2d &identity_norm) {
	auto branch3 = fuse_conv_bn(conv3, norm3);
	auto branch1 = fuse_conv_bn(conv1, norm1);
	auto fused = torch::nn::Conv2d(torch::nn::Conv2dOptions(conv3->options.in_channels(), conv3->options.out_channels(), 3).padding(1).bias(true));
	fused->to(conv3->weight.device(), conv3->weight.scalar_type());

	torch::NoGradGuard guard;
	auto weight = branch3->weight.clone();
	auto center = weight.index({Slice(), Slice(), 1, 1});
	center.add_(branch1->weight.squeeze(3).squeeze(2));
	const auto identity_scale = identity_norm->weight / torch::sqrt(identity_norm->running_var + identity_norm->options.eps());
	center.add_(torch::diag(identity_scale));
	const auto identity_bias = identity_norm->bias - identity_norm->running_mean * identity_scale;
	fused->weight.copy_(weight);
	fused->bias.copy_(branch3->bias + branch1->bias + identity_bias);
	fused->eval();
	return fused;
}

} // namespace

SquareEmbeddingImpl::SquareEmbeddingImpl(int channels) {
	values = register_parameter("values", torch::zeros({1, channels, 8, 8}));
}

torch::Tensor SquareEmbeddingImpl::forward(torch::Tensor x) {
	return x + values;
}

ResidualBlockImpl::ResidualBlockImpl(int channels, int sequence_depth, bool zero_output_scale)
    : channels_(channels), groups_(std::gcd(channels, kMaximumRelationGroups)), group_width_(channels / groups_) {
	if (channels <= 0 || sequence_depth <= 0) {
		throw std::invalid_argument("Gadus residual dimensions must be positive");
	}

	down_conv = register_module("down_conv", torch::nn::Conv2d(torch::nn::Conv2dOptions(channels_, channels_, 1).bias(false)));
	down_norm = register_module("down_norm", torch::nn::BatchNorm2d(channels_));
	local_conv3 = register_module("local_conv3", torch::nn::Conv2d(torch::nn::Conv2dOptions(channels_, channels_, 3).padding(1).bias(false)));
	local_norm3 = register_module("local_norm3", torch::nn::BatchNorm2d(channels_));
	local_conv1 = register_module("local_conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(channels_, channels_, 1).bias(false)));
	local_norm1 = register_module("local_norm1", torch::nn::BatchNorm2d(channels_));
	local_identity_norm = register_module("local_identity_norm", torch::nn::BatchNorm2d(channels_));
	relation_norm = register_module("relation_norm", torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(channels_).affine(false)));
	up_conv = register_module("up_conv", torch::nn::Conv2d(torch::nn::Conv2dOptions(channels_, channels_, 1).bias(false)));
	up_norm = register_module("up_norm", torch::nn::BatchNorm2d(channels_));

	const auto &geometry = relation_geometry();
	displacement_index = register_buffer("displacement_index", geometry.displacement_index.clone());
	displacement_scale = register_buffer("displacement_scale", geometry.displacement_scale.clone());
	displacement_support = register_buffer("displacement_support", geometry.displacement_support.clone());
	relation_coefficients = register_parameter("relation_coefficients", torch::zeros({groups_, kDisplacementCount}));
	relation_residual = register_parameter("relation_residual", torch::zeros({groups_, 64, 64}));
	path_balance = register_parameter("path_balance", torch::ones({channels_}));

	torch::NoGradGuard guard;
	for (int group = 0; group < std::min(groups_, 8); ++group) {
		const auto offsets = initial_relation_offsets(group);
		double squared_norm = 0.0;
		for (const auto &[rank_delta, file_delta] : offsets) {
			squared_norm += displacement_support.index({displacement_slot(rank_delta, file_delta)}).item<float>();
		}
		const double norm = std::sqrt(squared_norm);
		for (const auto &[rank_delta, file_delta] : offsets) {
			const int slot = displacement_slot(rank_delta, file_delta);
			const float coefficient = static_cast<float>(std::sqrt(displacement_support.index({slot}).item<float>()) / norm);
			relation_coefficients.index_put_({group, slot}, coefficient);
		}
	}
	const double local_scale = 1.0 / std::sqrt(3.0);
	for (const auto &norm : {local_norm3, local_norm1, local_identity_norm}) {
		norm->weight.fill_(local_scale);
		norm->bias.zero_();
	}
	up_norm->weight.fill_(zero_output_scale ? 0.0 : 1.0 / std::sqrt(static_cast<double>(sequence_depth)));
	up_norm->bias.zero_();
}

torch::Tensor ResidualBlockImpl::relation_matrices() const {
	const auto flat_index = displacement_index.reshape({-1});
	auto relative = relation_coefficients.index_select(1, flat_index).reshape({groups_, 64, 64});
	return relative * displacement_scale.unsqueeze(0) + relation_residual;
}

void ResidualBlockImpl::save(torch::serialize::OutputArchive &archive) const {
	save_without_buffers(*this, archive, {"displacement_index", "displacement_scale", "displacement_support"});
}

void ResidualBlockImpl::load(torch::serialize::InputArchive &archive) {
	load_without_buffers(*this, archive, {"displacement_index", "displacement_scale", "displacement_support"});
}

void ResidualBlockImpl::project_relation_residual() {
	torch::NoGradGuard guard;
	auto flat = relation_residual.view({groups_, kBoardSquares * kBoardSquares});
	const auto index = displacement_index.reshape({1, -1}).expand({groups_, -1});
	auto sums = torch::zeros({groups_, kDisplacementCount}, relation_residual.options());
	sums.scatter_add_(1, index, flat);
	const auto means = sums / displacement_support.unsqueeze(0);
	flat.sub_(means.gather(1, index));
}

torch::Tensor ResidualBlockImpl::forward(torch::Tensor x) {
	auto reduced = down_conv->forward(x);
	if (!inference_fused_) {
		reduced = down_norm->forward(reduced);
	}
	reduced = torch::relu(reduced);

	torch::Tensor local;
	if (inference_fused_) {
		local = fused_local->forward(reduced);
	} else {
		local = local_norm3->forward(local_conv3->forward(reduced)) + local_norm1->forward(local_conv1->forward(reduced)) + local_identity_norm->forward(reduced);
	}

	const auto batch = reduced.size(0);
	auto grouped = reduced.view({batch, groups_, group_width_, 64}).permute({0, 1, 3, 2});
	const auto matrices = inference_fused_ ? fused_relation : relation_matrices();
	auto related = torch::matmul(matrices.unsqueeze(0), grouped);
	related = related.permute({0, 1, 3, 2}).contiguous().view({batch, channels_, 8, 8});
	torch::Tensor combined;
	if (inference_fused_) {
		related = related * fused_relation_scale.view({1, channels_, 1, 1});
		combined = torch::relu(local + related);
	} else {
		related = relation_norm->forward(related);
		auto balance = path_balance.view({1, channels_, 1, 1});
		auto local_scale = torch::rsqrt(1.0 + balance.square());
		combined = torch::relu(local_scale * local + balance * local_scale * related);
	}
	auto update = up_conv->forward(combined);
	if (!inference_fused_) {
		update = up_norm->forward(update);
	}
	return torch::relu(x + update);
}

void ResidualBlockImpl::fuse_for_inference() {
	if (inference_fused_) {
		return;
	}
	if (is_training()) {
		throw std::logic_error("Gadus residual block must be in evaluation mode before fusion");
	}
	down_conv = replace_module("down_conv", fuse_conv_bn(down_conv, down_norm));
	up_conv = replace_module("up_conv", fuse_conv_bn(up_conv, up_norm));
	fused_relation = register_buffer("fused_relation", relation_matrices().detach().clone());
	fused_local = register_module("fused_local", fuse_local_branches(local_conv3, local_norm3, local_conv1, local_norm1, local_identity_norm));
	const auto balance = path_balance;
	const auto local_scale = torch::rsqrt(1.0 + balance.square());
	const auto relation_path_scale = balance * local_scale;
	const auto relation_norm_scale = torch::rsqrt(relation_norm->running_var + relation_norm->options.eps());
	const auto relation_norm_bias = -relation_norm->running_mean * relation_norm_scale;
	{
		torch::NoGradGuard guard;
		fused_local->weight.mul_(local_scale.reshape({-1, 1, 1, 1}));
		fused_local->bias.mul_(local_scale).add_(relation_path_scale * relation_norm_bias);
	}
	fused_relation_scale = register_buffer("fused_relation_scale", (relation_path_scale * relation_norm_scale).detach().clone());
	inference_fused_ = true;
}

ModelImpl::ModelImpl(int channels, int blocks) : channels_(channels), blocks_(blocks) {
	if (channels_ <= 0 || blocks_ <= 0) {
		throw std::invalid_argument("Gadus channels and blocks must be positive");
	}
	backbone = register_module("backbone", torch::nn::Sequential());
	backbone_conv = torch::nn::Conv2d(torch::nn::Conv2dOptions(kInputPlanes, channels_, 3).padding(1).bias(false));
	backbone_norm = torch::nn::BatchNorm2d(channels_);
	square_embedding = SquareEmbedding(channels_);
	backbone->push_back(backbone_conv);
	backbone->push_back(backbone_norm);
	backbone->push_back(torch::nn::ReLU(torch::nn::ReLUOptions(true)));
	backbone->push_back(square_embedding);
	for (int index = 0; index < blocks_; ++index) {
		backbone->push_back(ResidualBlock(channels_, blocks_));
	}

	policy_conv = register_module("policy_conv", torch::nn::Conv2d(torch::nn::Conv2dOptions(channels_, kPolicyChannels, 1).bias(false)));
	policy_norm = register_module("policy_norm", torch::nn::BatchNorm2d(kPolicyChannels));
	policy_blocks = register_module("policy_blocks", torch::nn::Sequential());
	for (int index = 0; index < kPolicyBlocks; ++index) {
		policy_blocks->push_back(ResidualBlock(kPolicyChannels, kPolicyBlocks));
	}
	policy_source = register_module("policy_source", torch::nn::Linear(torch::nn::LinearOptions(kPolicyChannels, kPolicyPlanes).bias(false)));
	policy_action_bias = register_parameter("policy_action_bias", torch::zeros({kActionSize}));
	auto action_sources = torch::empty({kActionSize}, torch::kInt64);
	auto action_patterns = torch::empty({kActionSize}, torch::kInt64);
	auto source_data = action_sources.accessor<std::int64_t, 1>();
	auto pattern_data = action_patterns.accessor<std::int64_t, 1>();
	for (int action = 0; action < kActionSize; ++action) {
		const int expanded = expanded_action_index(action);
		source_data[action] = expanded / kPolicyPlanes;
		pattern_data[action] = expanded % kPolicyPlanes;
	}
	compact_action_sources = register_buffer("compact_action_sources", action_sources);
	compact_action_patterns = register_buffer("compact_action_patterns", action_patterns);

	value_block = ResidualBlock(channels_, 1);
	value_block_2 = ResidualBlock(channels_, 1, true);
	value_conv = torch::nn::Conv2d(torch::nn::Conv2dOptions(channels_, kValueChannels, 1).bias(false));
	value_norm = torch::nn::BatchNorm2d(kValueChannels);
	value_hidden = torch::nn::Linear(kValueChannels * 8 * 8, kValueHidden);
	value_output = torch::nn::Linear(kValueHidden, 1);
	value_head = register_module("value_head",
	    torch::nn::Sequential(value_block, value_block_2, value_conv, value_norm, torch::nn::ReLU(torch::nn::ReLUOptions(true)), torch::nn::Flatten(), value_hidden,
	        torch::nn::ReLU(torch::nn::ReLUOptions(true)), value_output, torch::nn::Tanh()));
}

void ModelImpl::save(torch::serialize::OutputArchive &archive) const {
	save_without_buffers(*this, archive, {"compact_action_sources", "compact_action_patterns"});
}

void ModelImpl::load(torch::serialize::InputArchive &archive) {
	load_without_buffers(*this, archive, {"compact_action_sources", "compact_action_patterns"});
}

std::pair<torch::Tensor, torch::Tensor> ModelImpl::forward(torch::Tensor x) {
	if (x.dim() != 4 || x.size(1) != kInputPlanes || x.size(2) != 8 || x.size(3) != 8) {
		throw std::invalid_argument("Gadus input must have shape [N,17,8,8]");
	}
	auto features = trunk_features(x);
	return {policy_logits(features), value(features)};
}

std::pair<torch::Tensor, torch::Tensor> ModelImpl::forward_legal(torch::Tensor x, torch::Tensor legal_indices) {
	if (legal_indices.dim() != 2 || legal_indices.size(0) != x.size(0)) {
		throw std::invalid_argument("legal_indices must have shape [batch, legal_width]");
	}
	if (legal_indices.scalar_type() != torch::kInt64 || legal_indices.device() != x.device()) {
		throw std::invalid_argument("legal_indices must be int64 and reside on the input device");
	}

	auto features = trunk_features(x);
	auto compact_features = policy_features(features);
	return {policy_logits_for_indices(compact_features, legal_indices), value(features)};
}

void ModelImpl::initialize_output_priors(const torch::Tensor &action_counts, double mean_value, double smoothing_count, double output_scale) {
	if (action_counts.numel() != kActionSize || smoothing_count <= 0.0 || output_scale <= 0.0 || output_scale >= 1.0 || !std::isfinite(mean_value)) {
		throw std::invalid_argument("invalid Gadus output-prior statistics");
	}
	torch::NoGradGuard guard;
	auto counts = action_counts.to(torch::kCPU, torch::kFloat64).reshape({kActionSize});
	auto prior = (counts + smoothing_count) / (counts.sum().item<double>() + smoothing_count * kActionSize);
	policy_action_bias.copy_(prior.log().to(policy_action_bias.options()));
	policy_source->weight.mul_(output_scale);
	const double clipped = std::clamp(mean_value, -1.0 + kValueClipEpsilon, 1.0 - kValueClipEpsilon);
	value_output->weight.mul_(output_scale);
	value_output->bias.fill_(std::atanh(clipped));
}

torch::Tensor ModelImpl::centered_policy_bias() const {
	return policy_action_bias - policy_action_bias.mean();
}

torch::Tensor ModelImpl::trunk_features(torch::Tensor x) {
	auto result = backbone_conv->forward(x);
	if (!inference_fused_) {
		result = backbone_norm->forward(result);
	}
	result = square_embedding->forward(torch::relu(result));
	const std::size_t offset = inference_fused_ ? 3 : 4;
	for (int index = 0; index < blocks_; ++index) {
		result = backbone->ptr<ResidualBlockImpl>(offset + static_cast<std::size_t>(index))->forward(result);
	}
	return result;
}

torch::Tensor ModelImpl::value(torch::Tensor features) {
	AutocastGuard fp32(ComputePrecision::Fp32, features.device());
	auto result = value_block->forward(features.to(torch::kFloat32));
	result = value_block_2->forward(result);
	result = value_conv->forward(result);
	if (!inference_fused_) {
		result = value_norm->forward(result);
	}
	result = torch::relu(result).flatten(1);
	result = torch::relu(value_hidden->forward(result));
	return torch::tanh(value_output->forward(result));
}

torch::Tensor ModelImpl::policy_features(torch::Tensor features) {
	auto result = policy_conv->forward(features);
	if (!inference_fused_) {
		result = policy_norm->forward(result);
	}
	result = torch::relu(result);
	for (int index = 0; index < kPolicyBlocks; ++index) {
		result = policy_blocks->ptr<ResidualBlockImpl>(static_cast<std::size_t>(index))->forward(result);
	}
	return result;
}

torch::Tensor ModelImpl::policy_logits(torch::Tensor features) {
	auto compact_features = policy_features(features);
	auto indices = torch::arange(kActionSize, torch::TensorOptions().dtype(torch::kInt64).device(compact_features.device())).unsqueeze(0);
	return policy_logits_for_indices(compact_features, indices.expand({compact_features.size(0), -1}));
}

torch::Tensor ModelImpl::policy_logits_for_indices(const torch::Tensor &features, const torch::Tensor &compact_indices) {
	const auto batch = compact_indices.size(0);
	const auto width = compact_indices.size(1);
	auto squares = features.flatten(2).transpose(1, 2);
	auto flat_indices = compact_indices.reshape({-1});
	auto sources = compact_action_sources.index_select(0, flat_indices).view({batch, width});
	auto patterns = compact_action_patterns.index_select(0, flat_indices).view({batch, width});
	auto source_features = squares.gather(1, sources.unsqueeze(2).expand({-1, -1, kPolicyChannels}));
	auto source_weights = policy_source->weight.index_select(0, patterns.reshape({-1})).view({batch, width, kPolicyChannels});
	auto logits = (source_features * source_weights).sum(2);
	return logits + centered_policy_bias().index_select(0, flat_indices).view({batch, width});
}

void ModelImpl::project_relation_residuals() {
	if (inference_fused_) {
		throw std::logic_error("fused Gadus models cannot update relation residuals");
	}
	for (int index = 0; index < blocks_; ++index) {
		backbone->ptr<ResidualBlockImpl>(static_cast<std::size_t>(index + 4))->project_relation_residual();
	}
	for (int index = 0; index < kPolicyBlocks; ++index) {
		policy_blocks->ptr<ResidualBlockImpl>(static_cast<std::size_t>(index))->project_relation_residual();
	}
	value_block->project_relation_residual();
	value_block_2->project_relation_residual();
}

void ModelImpl::fuse_for_inference() {
	if (inference_fused_) {
		return;
	}
	if (is_training()) {
		throw std::logic_error("Gadus model must be in evaluation mode before fusion");
	}

	backbone_conv = fuse_conv_bn(backbone_conv, backbone_norm);
	auto fused_backbone = torch::nn::Sequential(backbone_conv, torch::nn::ReLU(torch::nn::ReLUOptions(true)), square_embedding);
	for (int index = 0; index < blocks_; ++index) {
		auto residual = backbone->ptr<ResidualBlockImpl>(static_cast<std::size_t>(index + 4));
		residual->fuse_for_inference();
		fused_backbone->push_back(ResidualBlock(residual));
	}
	backbone = replace_module("backbone", fused_backbone);
	backbone_norm = nullptr;

	policy_conv = replace_module("policy_conv", fuse_conv_bn(policy_conv, policy_norm));
	policy_norm = nullptr;
	for (int index = 0; index < kPolicyBlocks; ++index) {
		policy_blocks->ptr<ResidualBlockImpl>(static_cast<std::size_t>(index))->fuse_for_inference();
	}

	value_block->fuse_for_inference();
	value_block_2->fuse_for_inference();
	value_conv = fuse_conv_bn(value_conv, value_norm);
	value_head = replace_module("value_head",
	    torch::nn::Sequential(value_block, value_block_2, value_conv, torch::nn::ReLU(torch::nn::ReLUOptions(true)), torch::nn::Flatten(), value_hidden,
	        torch::nn::ReLU(torch::nn::ReLUOptions(true)), value_output, torch::nn::Tanh()));
	value_norm = nullptr;
	inference_fused_ = true;
}

int ModelImpl::channels() const noexcept {
	return channels_;
}
int ModelImpl::blocks() const noexcept {
	return blocks_;
}

std::int64_t parameter_count(const Model &model) {
	std::int64_t count = 0;
	for (const auto &parameter : model->parameters()) {
		count += parameter.numel();
	}
	return count;
}

} // namespace gadus
