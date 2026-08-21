// Implements the trainable sparse Value network and CPU snapshots.

#include "eleginus/model.hpp"

#include <cmath>
#include <cstring>
#include <stdexcept>

#include <torch/nn/init.h>

namespace eleginus {

namespace {

constexpr double kControlEdgeInitializationStd = 0.10;
constexpr double kControlContextInitializationStd = 0.05;
constexpr double kControlLocalInitializationBias = 0.25;
constexpr double kAttentionQueryInitializationStd = 0.05;

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

NetworkBatch encode_feature_batch(
	const std::vector<EncodedFeatures> &positions, const torch::Device &device) {
	auto feature_tensor = torch::empty(
		{static_cast<std::int64_t>(positions.size()), kPerspectiveCount, kFeatureSlots},
		torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
	auto side_tensor = torch::empty({static_cast<std::int64_t>(positions.size())},
								  torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
	auto *feature_data = feature_tensor.data_ptr<std::int64_t>();
	auto *side_data = side_tensor.data_ptr<std::int64_t>();
	auto occupancy_tensor = torch::empty(
		{static_cast<std::int64_t>(positions.size()), kPerspectiveCount, 64},
		torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
	auto material_tensor = torch::empty(
		{static_cast<std::int64_t>(positions.size()), kPerspectiveCount, kMaterialFeatureWidth},
		torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
	auto *occupancy_data = occupancy_tensor.data_ptr<std::int64_t>();
	auto *material_data = material_tensor.data_ptr<float>();
	std::vector<std::int64_t> edge_source;
	std::vector<std::int64_t> edge_target;
	std::vector<std::int64_t> edge_geometry;
	std::vector<std::int64_t> edge_destination;
	std::size_t cursor = 0;
	for (std::size_t position = 0; position < positions.size(); ++position) {
		for (const auto &perspective : positions[position].perspective) {
			const auto canonical = canonicalize_features(perspective);
			for (const int feature : canonical) {
				feature_data[cursor++] = feature;
			}
		}
		side_data[position] = positions[position].white_to_move ? 1 : 0;
		const auto controls = control_features(positions[position]);
		for (int perspective = 0; perspective < kPerspectiveCount; ++perspective) {
			for (int square = 0; square < 64; ++square)
				*occupancy_data++ = controls.occupancy[static_cast<std::size_t>(perspective)]
					[static_cast<std::size_t>(square)];
			for (int feature = 0; feature < kMaterialFeatureWidth; ++feature)
				*material_data++ = controls.material[static_cast<std::size_t>(perspective)]
					[static_cast<std::size_t>(feature)];
		}
		for (const auto &edge : controls.edges) {
			edge_source.push_back(edge.source);
			edge_target.push_back(edge.target);
			edge_geometry.push_back(edge.geometry);
			edge_destination.push_back(static_cast<std::int64_t>(
				(((position * kPerspectiveCount + edge.perspective) * 2 + edge.ownership) * 64 +
				 edge.square)));
		}
	}
	const auto edge_options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
	auto vector_tensor = [&](std::vector<std::int64_t> &values) {
		if (values.empty())
			return torch::empty({0}, edge_options).to(device);
		return torch::from_blob(values.data(), {static_cast<std::int64_t>(values.size())},
			edge_options).clone().to(device);
	};
	return NetworkBatch{
		feature_tensor.to(device), side_tensor.to(device), vector_tensor(edge_source),
		vector_tensor(edge_target), vector_tensor(edge_geometry), vector_tensor(edge_destination),
		occupancy_tensor.to(device), material_tensor.to(device),
	};
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

ValueNetworkImpl::ValueNetworkImpl() {
	encoder = register_module("encoder", SparseEncoder(kValueFeatureWidth));
	control_source = register_module("control_source",
		torch::nn::Embedding(torch::nn::EmbeddingOptions(
			kControlSourceVocabulary, kControlWidth)));
	control_target = register_module("control_target",
		torch::nn::Embedding(torch::nn::EmbeddingOptions(
			kControlSourceVocabulary, kControlWidth)));
	control_geometry = register_module("control_geometry",
		torch::nn::Embedding(torch::nn::EmbeddingOptions(
			kControlGeometryVocabulary, kControlWidth)));
	control_occupancy = register_module("control_occupancy",
		torch::nn::Embedding(torch::nn::EmbeddingOptions(
			kControlOccupancyVocabulary, kControlOccupancyWidth)));
	control_count = register_module("control_count",
		torch::nn::Embedding(torch::nn::EmbeddingOptions(
			kControlCountVocabulary, kControlCountWidth)));
	control_square = register_module("control_square",
		torch::nn::Embedding(torch::nn::EmbeddingOptions(64, kControlSquareWidth)));
	control_local = register_module("control_local", torch::nn::Linear(
		kControlWidth * 2 + kControlCountWidth * 2 + kControlOccupancyWidth +
			kControlSquareWidth, kControlLocalWidth));
	attention_key = register_module("attention_key",
		torch::nn::Linear(kControlLocalWidth,
			kControlAttentionHeads * kControlAttentionKeyWidth));
	attention_value = register_module("attention_value",
		torch::nn::Linear(kControlLocalWidth, kControlAttentionWidth));
	attention_query = register_module("attention_query",
		torch::nn::Embedding(torch::nn::EmbeddingOptions(
			kValueBucketCount,
			kControlAttentionHeads * kControlAttentionKeyWidth)));
	material = register_module("material", torch::nn::Linear(
		torch::nn::LinearOptions(kMaterialFeatureWidth, 1).bias(false)));
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
	control_source->weight.normal_(0.0, kControlEdgeInitializationStd);
	control_target->weight.normal_(0.0, kControlEdgeInitializationStd);
	control_geometry->weight.normal_(0.0, kControlEdgeInitializationStd);
	control_occupancy->weight.normal_(0.0, kControlContextInitializationStd);
	control_count->weight.normal_(0.0, kControlContextInitializationStd);
	control_square->weight.normal_(0.0, kControlContextInitializationStd);
	torch::nn::init::xavier_uniform_(control_local->weight);
	control_local->bias.fill_(kControlLocalInitializationBias);
	torch::nn::init::xavier_uniform_(attention_key->weight);
	attention_key->bias.zero_();
	torch::nn::init::xavier_uniform_(attention_value->weight);
	attention_value->bias.zero_();
	attention_query->weight.normal_(0.0, kAttentionQueryInitializationStd);
	material->weight.zero_();
	output->weight.normal_(0.0, 0.01);
	output->bias.zero_();
}

torch::Tensor ValueNetworkImpl::control_state(const NetworkBatch &batch, torch::Tensor buckets) {
	const auto batch_size = batch.features.size(0);
	auto messages = control_source->forward(batch.edge_source) +
		control_target->forward(batch.edge_target) +
		control_geometry->forward(batch.edge_geometry);
	auto field = torch::zeros({batch_size * kPerspectiveCount * 2 * 64, kControlWidth},
		messages.options());
	field.index_add_(0, batch.edge_destination, messages);
	auto counts = torch::zeros({batch_size * kPerspectiveCount * 2 * 64},
		torch::TensorOptions().dtype(torch::kFloat32).device(batch.features.device()));
	counts.index_add_(0, batch.edge_destination,
		torch::ones({batch.edge_destination.size(0)}, counts.options()));
	field = field.view({batch_size, kPerspectiveCount, 2, 64, kControlWidth});
	counts = counts.view({batch_size, kPerspectiveCount, 2, 64})
		.clamp(0, kControlCountVocabulary - 1).to(torch::kInt64);
	auto own = field.index({torch::indexing::Slice(), torch::indexing::Slice(), 0});
	auto opponent = field.index({torch::indexing::Slice(), torch::indexing::Slice(), 1});
	auto own_count = control_count->forward(
		counts.index({torch::indexing::Slice(), torch::indexing::Slice(), 0}));
	auto opponent_count = control_count->forward(
		counts.index({torch::indexing::Slice(), torch::indexing::Slice(), 1}));
	auto occupancy = control_occupancy->forward(batch.occupancy.to(torch::kInt64));
	auto squares = control_square->forward(torch::arange(64, batch.features.options()))
		.view({1, 1, 64, kControlSquareWidth})
		.expand({batch_size, kPerspectiveCount, 64, kControlSquareWidth});
	auto local_input = torch::cat(
		{own, opponent, own_count, opponent_count, occupancy, squares}, -1);
	auto local = control_local->forward(local_input).clamp(0.0, 1.0).square();
	auto mean = local.mean(-2);
	auto key = attention_key->forward(local).view({batch_size, kPerspectiveCount, 64,
		kControlAttentionHeads, kControlAttentionKeyWidth});
	auto value = attention_value->forward(local).view({batch_size, kPerspectiveCount, 64,
		kControlAttentionHeads, kControlAttentionHeadWidth});
	auto query = attention_query->forward(buckets).view(
		{batch_size, 1, 1, kControlAttentionHeads, kControlAttentionKeyWidth});
	auto logits = (key * query).sum(-1) /
		std::sqrt(static_cast<double>(kControlAttentionKeyWidth));
	auto weights = torch::softmax(logits.clamp(-8.0, 8.0), 2);
	auto attention = (weights.unsqueeze(-1) * value).sum(2).reshape(
		{batch_size, kPerspectiveCount, kControlAttentionWidth});
	return torch::cat({mean, attention}, -1);
}

torch::Tensor ValueNetworkImpl::forward(const NetworkBatch &batch) {
	const auto buckets = value_bucket_indices(batch.features);
	auto accumulator = encoder->accumulate(batch.features);
	auto white = accumulator.index({torch::indexing::Slice(), 0});
	auto black = accumulator.index({torch::indexing::Slice(), 1});
	auto mask = batch.white_to_move.to(torch::kBool).unsqueeze(1);
	auto first = torch::where(mask, white, black);
	auto second = torch::where(mask, black, white);
	auto controls = control_state(batch, buckets);
	auto white_control = controls.index({torch::indexing::Slice(), 0});
	auto black_control = controls.index({torch::indexing::Slice(), 1});
	auto first_control = torch::where(mask, white_control, black_control);
	auto second_control = torch::where(mask, black_control, white_control);
	auto dense = torch::cat({
		first.index({torch::indexing::Slice(),
			torch::indexing::Slice(0, kValueAccumulatorWidth)}).clamp(0.0, 1.0).square(),
		first_control,
		second.index({torch::indexing::Slice(),
			torch::indexing::Slice(0, kValueAccumulatorWidth)}).clamp(0.0, 1.0).square(),
		second_control}, 1);
	auto hidden_value = torch::relu(select_bucket(
		hidden->forward(dense), buckets, kValueHiddenWidth));
	auto value = torch::relu(select_bucket(
		bottleneck->forward(hidden_value), buckets, kValueBottleneckWidth));
	auto network = select_bucket_scalar(output->forward(value), buckets);
	auto white_material = batch.material.index({torch::indexing::Slice(), 0});
	auto black_material = batch.material.index({torch::indexing::Slice(), 1});
	auto first_material = torch::where(mask, white_material, black_material);
	auto material_value = material->forward(first_material).squeeze(1);
	auto psqt = 0.5F * select_bucket_scalar(
		first.index({torch::indexing::Slice(),
			torch::indexing::Slice(kValueAccumulatorWidth, kValueFeatureWidth)}) -
		second.index({torch::indexing::Slice(),
			torch::indexing::Slice(kValueAccumulatorWidth, kValueFeatureWidth)}), buckets);
	return network + material_value + psqt;
}

ModelImpl::ModelImpl() {
	value = register_module("value", ValueNetwork());
}

CpuValue snapshot_value(const ValueNetwork &model) {
	if (!model) {
		throw std::invalid_argument("cannot snapshot an empty Eleginus Value");
	}
	return CpuValue(ValueWeights{
		tensor_vector(model->encoder->table->weight),
		tensor_vector(model->encoder->bias),
		tensor_vector(model->control_source->weight),
		tensor_vector(model->control_target->weight),
		tensor_vector(model->control_geometry->weight),
		tensor_vector(model->control_occupancy->weight),
		tensor_vector(model->control_count->weight),
		tensor_vector(model->control_square->weight),
		tensor_vector(model->control_local->weight),
		tensor_vector(model->control_local->bias),
		tensor_vector(model->attention_key->weight),
		tensor_vector(model->attention_key->bias),
		tensor_vector(model->attention_value->weight),
		tensor_vector(model->attention_value->bias),
		tensor_vector(model->attention_query->weight),
		tensor_vector(model->material->weight),
		tensor_vector(model->hidden->weight),
		tensor_vector(model->hidden->bias),
		tensor_vector(model->bottleneck->weight),
		tensor_vector(model->bottleneck->bias),
		tensor_vector(model->output->weight),
		tensor_vector(model->output->bias),
	});
}

void restore_value(const ValueNetwork &model, const ValueWeights &weights) {
	if (!model) {
		throw std::invalid_argument("cannot restore an empty Eleginus Value");
	}
	torch::NoGradGuard no_grad;
	copy_tensor(model->encoder->table->weight, weights.feature_table, "Value feature table");
	copy_tensor(model->encoder->bias, weights.accumulator_bias, "Value accumulator bias");
	copy_tensor(model->control_source->weight, weights.control_source, "control source table");
	copy_tensor(model->control_target->weight, weights.control_target, "control target table");
	copy_tensor(model->control_geometry->weight, weights.control_geometry,
		"control geometry table");
	copy_tensor(model->control_occupancy->weight, weights.control_occupancy,
		"control occupancy table");
	copy_tensor(model->control_count->weight, weights.control_count, "control count table");
	copy_tensor(model->control_square->weight, weights.control_square, "control square table");
	copy_tensor(model->control_local->weight, weights.control_local_weight,
		"control local weight");
	copy_tensor(model->control_local->bias, weights.control_local_bias, "control local bias");
	copy_tensor(model->attention_key->weight, weights.attention_key_weight,
		"attention key weight");
	copy_tensor(model->attention_key->bias, weights.attention_key_bias, "attention key bias");
	copy_tensor(model->attention_value->weight, weights.attention_value_weight,
		"attention value weight");
	copy_tensor(model->attention_value->bias, weights.attention_value_bias,
		"attention value bias");
	copy_tensor(model->attention_query->weight, weights.attention_query, "attention query");
	copy_tensor(model->material->weight, weights.material_weight, "material weight");
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
