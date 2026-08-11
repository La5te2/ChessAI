// Implements independent trainable sparse Policy/Value networks and CPU snapshots.

#include "eleginus/model.hpp"

#include <cmath>
#include <cstring>
#include <stdexcept>

namespace eleginus {

namespace {

std::vector<float> tensor_vector(const torch::Tensor &source) {
	auto tensor = source.detach().to(torch::kCPU).to(torch::kFloat32).contiguous().view({-1});
	std::vector<float> result(static_cast<std::size_t>(tensor.numel()));
	std::memcpy(result.data(), tensor.data_ptr<float>(), result.size() * sizeof(float));
	return result;
}

void copy_tensor(const torch::Tensor &target, const std::vector<float> &source,
				 const char *name) {
	if (static_cast<std::size_t>(target.numel()) != source.size()) {
		throw std::invalid_argument(std::string("invalid Eleginus ") + name + " size");
	}
	auto tensor = torch::from_blob(const_cast<float *>(source.data()),
		{static_cast<std::int64_t>(source.size())}, torch::kFloat32).clone();
	target.copy_(tensor.view(target.sizes()).to(target.device()));
}

torch::Tensor value_bucket_indices(const torch::Tensor &features) {
	auto piece_count = features.index({torch::indexing::Slice(), 0})
		.ne(kPaddingFeature)
		.sum(1) - 2;
	return torch::floor((piece_count.to(torch::kFloat32) - 1.0F) / 4.0F)
		.clamp(0, kValueBucketCount - 1)
		.to(torch::kInt64);
}

torch::Tensor select_bucket(torch::Tensor values, const torch::Tensor &buckets, int width) {
	const auto rows = torch::arange(values.size(0), buckets.options()) * kValueBucketCount +
		buckets;
	return values.view({values.size(0) * kValueBucketCount, width}).index_select(0, rows);
}

torch::Tensor select_bucket_scalar(torch::Tensor values, const torch::Tensor &buckets) {
	const auto rows = torch::arange(values.size(0), buckets.options()) * kValueBucketCount +
		buckets;
	return values.reshape({values.size(0) * kValueBucketCount}).index_select(0, rows);
}

} // namespace

std::pair<torch::Tensor, torch::Tensor>
encode_feature_batch(const std::vector<EncodedFeatures> &positions, const torch::Device &device) {
	auto feature_tensor = torch::empty(
		{static_cast<std::int64_t>(positions.size()), kPerspectiveCount, kFeatureSlots},
		torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
	auto side_tensor = torch::empty({static_cast<std::int64_t>(positions.size())},
								  torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
	auto *feature_data = feature_tensor.data_ptr<std::int64_t>();
	auto *side_data = side_tensor.data_ptr<std::int64_t>();
	std::size_t cursor = 0;
	for (std::size_t position = 0; position < positions.size(); ++position) {
		for (const auto &perspective : positions[position].perspective) {
			const auto canonical = canonicalize_features(perspective);
			for (const int feature : canonical) {
				feature_data[cursor++] = feature;
			}
		}
		side_data[position] = positions[position].white_to_move ? 1 : 0;
	}
	return {feature_tensor.to(device), side_tensor.to(device)};
}

SparseEncoderImpl::SparseEncoderImpl(int width_value) : width(width_value) {
	if (width <= 0) {
		throw std::invalid_argument("Eleginus accumulator width must be positive");
	}
	table = register_module(
		"table", torch::nn::Embedding(torch::nn::EmbeddingOptions(kFeatureVocabulary, width)
									 .padding_idx(kPaddingFeature)));
	bias = register_parameter("bias", torch::full({width}, 0.5F));
	torch::NoGradGuard no_grad;
	table->weight.normal_(0.0, 0.01);
	table->weight.index_put_({kPaddingFeature}, 0.0);
}

torch::Tensor SparseEncoderImpl::accumulate(torch::Tensor features) {
	if (features.dim() != 3 || features.size(1) != kPerspectiveCount ||
		features.size(2) != kFeatureSlots) {
		throw std::runtime_error("expected Eleginus sparse features [batch, 2, 34]");
	}
	return table->forward(features.to(torch::kInt64)).sum(2) + bias.view({1, 1, width});
}

torch::Tensor SparseEncoderImpl::forward(torch::Tensor features, torch::Tensor white_to_move) {
	auto accumulator = accumulate(features).clamp(0.0, 1.0);
	auto white = accumulator.index({torch::indexing::Slice(), 0});
	auto black = accumulator.index({torch::indexing::Slice(), 1});
	auto mask = white_to_move.to(torch::kBool).unsqueeze(1);
	auto first = torch::where(mask, white, black);
	auto second = torch::where(mask, black, white);
	return torch::cat({first, second}, 1);
}

PolicyNetworkImpl::PolicyNetworkImpl() {
	encoder = register_module("encoder", SparseEncoder(kPolicyAccumulatorWidth));
	hidden = register_module("hidden", torch::nn::Linear(kPolicyAccumulatorWidth * 2,
		kPolicyHiddenWidth));
	output = register_module("output", torch::nn::Linear(kPolicyHiddenWidth, kActionSize));
}

torch::Tensor PolicyNetworkImpl::hidden_state(torch::Tensor features,
										   torch::Tensor white_to_move) {
	return torch::relu(hidden->forward(encoder->forward(features, white_to_move)));
}

torch::Tensor PolicyNetworkImpl::forward(torch::Tensor features, torch::Tensor white_to_move) {
	return output->forward(hidden_state(features, white_to_move));
}

ValueNetworkImpl::ValueNetworkImpl() {
	encoder = register_module("encoder", SparseEncoder(kValueFeatureWidth));
	attention = register_module("attention", torch::nn::Embedding(
		torch::nn::EmbeddingOptions(kFeatureVocabulary, kValueAttentionTableWidth)
			.padding_idx(kPaddingFeature)));
	hidden = register_module("hidden", torch::nn::Linear(kValueDenseWidth * 2,
		kValueBucketCount * kValueHiddenWidth));
	bottleneck = register_module(
		"bottleneck", torch::nn::Linear(kValueHiddenWidth,
			kValueBucketCount * kValueBottleneckWidth));
	output = register_module(
		"output", torch::nn::Linear(kValueBottleneckWidth, kValueBucketCount));
	torch::NoGradGuard no_grad;
	encoder->table->weight.index_put_(
		{torch::indexing::Slice(),
		 torch::indexing::Slice(kValueAccumulatorWidth, kValueFeatureWidth)}, 0.0F);
	encoder->bias.index_put_(
		{torch::indexing::Slice(kValueAccumulatorWidth, kValueFeatureWidth)}, 0.0F);
	attention->weight.index({torch::indexing::Slice(),
		torch::indexing::Slice(0, kValueAttentionWidth)}).normal_(0.0, 1.0);
	attention->weight.index({torch::indexing::Slice(),
		torch::indexing::Slice(kValueAttentionWidth, 2 * kValueAttentionWidth)})
		.normal_(0.0, 1.0);
	attention->weight.index({torch::indexing::Slice(),
		torch::indexing::Slice(2 * kValueAttentionWidth, kValueAttentionTableWidth)})
		.normal_(0.0, 0.1);
	attention->weight.index_put_({kPaddingFeature}, 0.0F);
	output->weight.normal_(0.0, 0.01);
	output->bias.zero_();
}

torch::Tensor ValueNetworkImpl::relation_state(torch::Tensor features) {
	if (features.dim() != 3 || features.size(1) != kPerspectiveCount ||
		features.size(2) != kFeatureSlots) {
		throw std::runtime_error("expected Eleginus sparse features [batch, 2, 34]");
	}

	// A centered query-key coefficient weights each directed piece-pair message.
	// Averaging by source-piece count preserves interaction scale across game phases.
	auto piece_mask = features.lt(kPieceFeatureCount);
	auto piece_features = torch::where(piece_mask, features,
		torch::full_like(features, kPaddingFeature));
	auto qkv = attention->forward(piece_features.to(torch::kInt64));
	auto chunks = qkv.split(kValueAttentionWidth, -1);
	auto query = chunks[0];
	auto key = chunks[1];
	auto value = chunks[2];
	auto score = torch::matmul(query, key.transpose(-1, -2)) /
		std::sqrt(static_cast<double>(kValueAttentionWidth));
	auto slots = torch::arange(kFeatureSlots, features.options()).view(
		{1, 1, kFeatureSlots});
	auto pair_mask = piece_mask.unsqueeze(-1).logical_and(piece_mask.unsqueeze(-2))
		.logical_and(slots.unsqueeze(-1).ne(slots.unsqueeze(-2)));
	auto coefficients = (2.0 * torch::sigmoid(score.clamp(-8.0, 8.0)) - 1.0) *
		pair_mask.to(score.scalar_type());
	auto total = torch::matmul(coefficients, value).sum(-2);
	auto count = piece_mask.sum(-1).to(score.scalar_type());
	auto denominator = count.clamp_min(1.0).unsqueeze(-1);
	return total / denominator;
}

torch::Tensor ValueNetworkImpl::forward(torch::Tensor features, torch::Tensor white_to_move) {
	const auto buckets = value_bucket_indices(features);
	auto accumulator = encoder->accumulate(features);
	auto white = accumulator.index({torch::indexing::Slice(), 0});
	auto black = accumulator.index({torch::indexing::Slice(), 1});
	auto mask = white_to_move.to(torch::kBool).unsqueeze(1);
	auto first = torch::where(mask, white, black);
	auto second = torch::where(mask, black, white);
	auto relations = relation_state(features);
	auto white_relation = relations.index({torch::indexing::Slice(), 0});
	auto black_relation = relations.index({torch::indexing::Slice(), 1});
	auto first_relation = torch::where(mask, white_relation, black_relation);
	auto second_relation = torch::where(mask, black_relation, white_relation);
	auto dense = torch::cat({
		first.index({torch::indexing::Slice(),
			torch::indexing::Slice(0, kValueAccumulatorWidth)}).clamp(0.0, 1.0).square(),
		first_relation,
		second.index({torch::indexing::Slice(),
			torch::indexing::Slice(0, kValueAccumulatorWidth)}).clamp(0.0, 1.0).square(),
		second_relation}, 1);
	auto hidden_value = torch::relu(select_bucket(
		hidden->forward(dense), buckets, kValueHiddenWidth));
	auto value = torch::relu(select_bucket(
		bottleneck->forward(hidden_value), buckets, kValueBottleneckWidth));
	auto network = select_bucket_scalar(output->forward(value), buckets);
	auto psqt = 0.5F * select_bucket_scalar(
		first.index({torch::indexing::Slice(),
			torch::indexing::Slice(kValueAccumulatorWidth, kValueFeatureWidth)}) -
		second.index({torch::indexing::Slice(),
			torch::indexing::Slice(kValueAccumulatorWidth, kValueFeatureWidth)}), buckets);
	return network + psqt;
}

ModelImpl::ModelImpl() {
	policy = register_module("policy", PolicyNetwork());
	value = register_module("value", ValueNetwork());
}

CpuPolicy snapshot_policy(const PolicyNetwork &model) {
	if (!model) {
		throw std::invalid_argument("cannot snapshot an empty Eleginus Policy");
	}
	return CpuPolicy(PolicyWeights{
		tensor_vector(model->encoder->table->weight),
		tensor_vector(model->encoder->bias),
		tensor_vector(model->hidden->weight),
		tensor_vector(model->hidden->bias),
		tensor_vector(model->output->weight),
		tensor_vector(model->output->bias),
	});
}

CpuValue snapshot_value(const ValueNetwork &model) {
	if (!model) {
		throw std::invalid_argument("cannot snapshot an empty Eleginus Value");
	}
	return CpuValue(ValueWeights{
		tensor_vector(model->encoder->table->weight),
		tensor_vector(model->encoder->bias),
		tensor_vector(model->attention->weight),
		tensor_vector(model->hidden->weight),
		tensor_vector(model->hidden->bias),
		tensor_vector(model->bottleneck->weight),
		tensor_vector(model->bottleneck->bias),
		tensor_vector(model->output->weight),
		tensor_vector(model->output->bias),
	});
}

void restore_policy(const PolicyNetwork &model, const PolicyWeights &weights) {
	if (!model) {
		throw std::invalid_argument("cannot restore an empty Eleginus Policy");
	}
	torch::NoGradGuard no_grad;
	copy_tensor(model->encoder->table->weight, weights.feature_table, "Policy feature table");
	copy_tensor(model->encoder->bias, weights.accumulator_bias, "Policy accumulator bias");
	copy_tensor(model->hidden->weight, weights.hidden_weight, "Policy hidden weight");
	copy_tensor(model->hidden->bias, weights.hidden_bias, "Policy hidden bias");
	copy_tensor(model->output->weight, weights.output_weight, "Policy output weight");
	copy_tensor(model->output->bias, weights.output_bias, "Policy output bias");
}

void restore_value(const ValueNetwork &model, const ValueWeights &weights) {
	if (!model) {
		throw std::invalid_argument("cannot restore an empty Eleginus Value");
	}
	torch::NoGradGuard no_grad;
	copy_tensor(model->encoder->table->weight, weights.feature_table, "Value feature table");
	copy_tensor(model->encoder->bias, weights.accumulator_bias, "Value accumulator bias");
	copy_tensor(model->attention->weight, weights.attention_table, "Value attention table");
	copy_tensor(model->hidden->weight, weights.hidden_weight, "Value hidden weight");
	copy_tensor(model->hidden->bias, weights.hidden_bias, "Value hidden bias");
	copy_tensor(model->bottleneck->weight, weights.bottleneck_weight,
		"Value bottleneck weight");
	copy_tensor(model->bottleneck->bias, weights.bottleneck_bias,
		"Value bottleneck bias");
	copy_tensor(model->output->weight, weights.output_weight, "Value output weight");
	copy_tensor(model->output->bias, weights.output_bias, "Value output bias");
}

std::int64_t parameter_count(const torch::nn::Module &model) {
	std::int64_t count = 0;
	for (const auto &parameter : model.parameters()) {
		count += parameter.numel();
	}
	return count;
}

} // namespace eleginus
