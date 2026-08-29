// Implements Melano's geometry-aware token network and Policy/Value heads.

#include "melano/model.hpp"
#include "melano/attention.hpp"
#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace melano {

namespace {

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

// Select a practical multi-head factor without requiring channels to use one fixed width.
int attention_heads_for_channels(int channels) {
	for (const int heads : {8, 4, 2}) {
		if (channels % heads == 0) {
			return heads;
		}
	}
	return 1;
}

// Reject invalid architecture descriptors instead of silently changing model dimensions.
int require_positive(int value, const char *name) {
	if (value <= 0) {
		throw std::invalid_argument(std::string(name) + " must be positive");
	}
	return value;
}

} // namespace

// Embed board contents and rule state into 64 square tokens.
StateEmbeddingImpl::StateEmbeddingImpl(int channels) {
	piece = register_module("piece", torch::nn::Embedding(13, channels));
	square = register_module("square", torch::nn::Embedding(kBoardSquares, channels));
	castling = register_module("castling", torch::nn::Embedding(16, channels));
	ep_file = register_module("ep_file", torch::nn::Embedding(9, channels));
}

// Add piece, absolute-square, and rule-context embeddings for every square.
torch::Tensor StateEmbeddingImpl::forward(torch::Tensor state) {
	if (state.dim() != 2 || state.size(1) != kStateFeatures) {
		throw std::runtime_error("expected Melano state [batch, 66]");
	}
	state = state.to(torch::kInt64);
	auto pieces = state.index({torch::indexing::Slice(), torch::indexing::Slice(0, kBoardSquares)}).clamp(0, 12);
	auto castling_token = state.index({torch::indexing::Slice(), 64}).clamp(0, 15);
	auto ep_token = state.index({torch::indexing::Slice(), 65}).clamp(0, 8);
	auto context = castling->forward(castling_token) + ep_file->forward(ep_token);
	return piece->forward(pieces) + square->weight.unsqueeze(0) + context.unsqueeze(1);
}

// Construct pre-norm attention with a position-conditioned geometric bias generator.
GeometryAttentionBlockImpl::GeometryAttentionBlockImpl(int channel_count)
    : channels(channel_count), heads(attention_heads_for_channels(channel_count)), head_dim(channel_count / heads) {
	position = register_parameter("position", torch::zeros({1, kTokenCount, channels}));
	norm1 = register_module("norm1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({channels})));
	qkv = register_module("qkv", torch::nn::Linear(channels, channels * 3));
	out = register_module("out", torch::nn::Linear(channels, channels));
	gab_square = register_module("gab_square", torch::nn::Linear(channels, kGabSquareChannels));
	gab_state = register_module("gab_state", torch::nn::Linear(kTokenCount * kGabSquareChannels, kGabStateChannels));
	gab_state_norm = register_module("gab_state_norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({kGabStateChannels})));
	gab_coefficients = register_module("gab_coefficients", torch::nn::Linear(kGabStateChannels, heads * kGabTemplateCount));
	gab_coefficient_norm = register_module("gab_coefficient_norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({heads * kGabTemplateCount})));
	norm2 = register_module("norm2", torch::nn::LayerNorm(torch::nn::LayerNormOptions({channels})));
	ffn = register_module("ffn", torch::nn::Sequential(torch::nn::Linear(channels, channels * 2), torch::nn::GELU(), torch::nn::Linear(channels * 2, channels)));
}

// Compress the complete square sequence and mix the model-wide geometric templates.
torch::Tensor GeometryAttentionBlockImpl::geometry_bias(torch::Tensor tokens, torch::Tensor templates) {
	if (tokens.dim() != 3 || tokens.size(1) != kTokenCount || tokens.size(2) != channels) {
		throw std::runtime_error("invalid Melano geometry-attention token shape");
	}
	if (templates.sizes() != torch::IntArrayRef({kGabTemplateCount, kTokenCount, kTokenCount})) {
		throw std::runtime_error("invalid Melano geometric template shape");
	}
	return torch::matmul(geometry_coefficients(tokens), templates.view({kGabTemplateCount, kTokenCount * kTokenCount}))
	    .view({tokens.size(0), heads, kTokenCount, kTokenCount});
}

// Produce the dynamic template coefficients without constructing their square-pair expansion.
torch::Tensor GeometryAttentionBlockImpl::geometry_coefficients(torch::Tensor tokens) {
	const auto batch = tokens.size(0);
	auto compressed = gab_square->forward(tokens).reshape({batch, kTokenCount * kGabSquareChannels});
	compressed = gab_state_norm->forward(torch::gelu(gab_state->forward(compressed)));
	return gab_coefficient_norm->forward(torch::gelu(gab_coefficients->forward(compressed))).view({batch, heads, kGabTemplateCount});
}

// Compute softmax((QK^T)/sqrt(d) + GAB)V with residual updates.
torch::Tensor GeometryAttentionBlockImpl::forward(torch::Tensor tokens, torch::Tensor templates) {
	if (tokens.dim() != 3 || tokens.size(1) != kTokenCount || tokens.size(2) != channels) {
		throw std::runtime_error("invalid Melano geometry-attention token shape");
	}
	const auto batch = tokens.size(0);
	tokens = tokens + position;
	auto packed = qkv->forward(norm1->forward(tokens)).view({batch, kTokenCount, 3, heads, head_dim});
	auto parts = packed.unbind(2);
	auto query = parts[0].transpose(1, 2);
	auto key = parts[1].transpose(1, 2);
	auto value = parts[2].transpose(1, 2);
	auto attention_output = geometry_attention(query, key, value, geometry_coefficients(tokens), templates);
	attention_output = attention_output.transpose(1, 2).contiguous().view({batch, kTokenCount, channels});
	tokens = tokens + out->forward(attention_output);
	return tokens + ffn->forward(norm2->forward(tokens));
}

// Build factorized from/to logits and a dedicated underpromotion suffix.
ActionHeadImpl::ActionHeadImpl(int channels) {
	norm = register_module("norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({channels})));
	from_proj = register_module("from_proj", torch::nn::Linear(channels, channels));
	to_proj = register_module("to_proj", torch::nn::Linear(channels, channels));
	underpromotion = register_module("underpromotion", torch::nn::Linear(channels, kUnderpromotionPlanes));
	auto metadata = torch::empty({kActionSize, 4}, torch::kInt64);
	auto ordinary = torch::empty({kOrdinaryActionSize}, torch::kInt64);
	auto promotions = torch::empty({kUnderpromotionActionSize}, torch::kInt64);
	auto metadata_values = metadata.accessor<std::int64_t, 2>();
	auto ordinary_values = ordinary.accessor<std::int64_t, 1>();
	auto promotion_values = promotions.accessor<std::int64_t, 1>();
	for (int action = 0; action < kActionSize; ++action) {
		const int expanded_action = expanded_action_index(action);
		if (expanded_action < kBoardSquares * kBoardSquares) {
			metadata_values[action][0] = expanded_action / kBoardSquares;
			metadata_values[action][1] = expanded_action % kBoardSquares;
			metadata_values[action][2] = 0;
			metadata_values[action][3] = 1;
			ordinary_values[action] = expanded_action;
		} else {
			const int suffix = expanded_action - kBoardSquares * kBoardSquares;
			const int source = suffix / kUnderpromotionPlanes;
			const int pattern = suffix % kUnderpromotionPlanes;
			metadata_values[action][0] = source;
			metadata_values[action][1] = 7 * 8 + source % 8 + pattern / 3 - 1;
			metadata_values[action][2] = pattern;
			metadata_values[action][3] = 0;
			promotion_values[action - kOrdinaryActionSize] = suffix - 48 * kUnderpromotionPlanes;
		}
	}
	action_metadata = register_buffer("action_metadata", metadata);
	ordinary_actions = register_buffer("ordinary_actions", ordinary);
	promotion_actions = register_buffer("promotion_actions", promotions);
}

// Score ordinary moves by scaled source-destination dot products, then append promotions.
torch::Tensor ActionHeadImpl::forward(torch::Tensor square_tokens) {
	if (square_tokens.dim() != 3 || square_tokens.size(1) != kBoardSquares) {
		throw std::runtime_error("expected Melano square tokens [batch, 64, channels]");
	}
	auto normalized = norm->forward(square_tokens);
	torch::Tensor from_to;
	if (inference_fused) {
		from_to = torch::matmul(torch::matmul(normalized, fused_policy_matrix), normalized.transpose(1, 2));
		from_to.add_(torch::matmul(normalized, fused_source_linear).unsqueeze(2));
		from_to.add_(torch::matmul(normalized, fused_destination_linear).unsqueeze(1));
		from_to.add_(fused_constant);
	} else {
		auto from = from_proj->forward(normalized);
		auto to = to_proj->forward(normalized);
		from_to = torch::matmul(from, to.transpose(1, 2)) / std::sqrt(static_cast<double>(from.size(2)));
	}
	auto ordinary_logits = from_to.contiguous().view({normalized.size(0), kBoardSquares * kBoardSquares}).index_select(1, ordinary_actions);
	auto promotion_logits = underpromotion->forward(normalized.slice(1, 48, 56))
	                            .contiguous()
	                            .view({normalized.size(0), 8 * kUnderpromotionPlanes})
	                            .index_select(1, promotion_actions);
	return torch::cat({ordinary_logits, promotion_logits}, 1);
}

// Score only requested source-destination pairs and underpromotions.
torch::Tensor ActionHeadImpl::forward_legal(torch::Tensor square_tokens, torch::Tensor legal_indices) {
	if (square_tokens.dim() != 3 || square_tokens.size(1) != kBoardSquares) {
		throw std::runtime_error("expected Melano square tokens [batch, 64, channels]");
	}
	if (legal_indices.dim() != 2 || legal_indices.size(0) != square_tokens.size(0)) {
		throw std::runtime_error("legal_indices must have shape [batch, legal_width]");
	}
	const bool compact_or_native = legal_indices.scalar_type() == torch::kInt16 || legal_indices.scalar_type() == torch::kInt64;
	if (!compact_or_native || legal_indices.device() != square_tokens.device()) {
		throw std::runtime_error("legal_indices must be int16 or int64 and reside on the token device");
	}

	legal_indices = legal_indices.to(torch::kInt64);
	auto normalized = norm->forward(square_tokens);
	auto metadata = action_metadata.index_select(0, legal_indices.reshape({-1})).view({legal_indices.size(0), legal_indices.size(1), 4});
	const auto source = metadata.select(2, 0);
	const auto destination = metadata.select(2, 1);
	const auto promotion_indices = metadata.select(2, 2);
	const auto ordinary = metadata.select(2, 3).to(torch::kBool);
	const auto channels = normalized.size(2);
	const auto gather_shape = source.unsqueeze(-1).expand({-1, -1, channels});
	auto selected_source = normalized.gather(1, gather_shape);
	auto selected_destination = normalized.gather(1, destination.unsqueeze(-1).expand({-1, -1, channels}));
	torch::Tensor ordinary_logits;
	if (inference_fused) {
		torch::Tensor transformed_source;
		if (legal_indices.size(1) < kBoardSquares) {
			transformed_source = torch::matmul(selected_source, fused_policy_matrix);
		} else {
			transformed_source = torch::matmul(normalized, fused_policy_matrix).gather(1, gather_shape);
		}
		ordinary_logits = (transformed_source * selected_destination).sum(-1);
		ordinary_logits.add_(torch::matmul(selected_source, fused_source_linear));
		ordinary_logits.add_(torch::matmul(selected_destination, fused_destination_linear));
		ordinary_logits.add_(fused_constant);
	} else {
		torch::Tensor selected_from;
		torch::Tensor selected_to;
		if (legal_indices.size(1) < kBoardSquares) {
			selected_from = from_proj->forward(selected_source);
			selected_to = to_proj->forward(selected_destination);
		} else {
			selected_from = from_proj->forward(normalized).gather(1, gather_shape);
			selected_to = to_proj->forward(normalized).gather(1, destination.unsqueeze(-1).expand({-1, -1, channels}));
		}
		ordinary_logits = (selected_from * selected_to).sum(-1) / std::sqrt(static_cast<double>(channels));
	}

	auto promotion_logits = underpromotion->forward(selected_source).gather(2, promotion_indices.unsqueeze(-1)).squeeze(-1);
	return torch::where(ordinary, ordinary_logits, promotion_logits);
}

void ActionHeadImpl::fuse_for_inference() {
	if (inference_fused) {
		return;
	}
	if (is_training()) {
		throw std::logic_error("Melano Policy head must be in evaluation mode before fusion");
	}
	torch::NoGradGuard guard;
	const auto scale = std::sqrt(static_cast<double>(from_proj->weight.size(0)));
	const auto from_bias = from_proj->bias.defined() ? from_proj->bias : torch::zeros({from_proj->weight.size(0)}, from_proj->weight.options());
	const auto to_bias = to_proj->bias.defined() ? to_proj->bias : torch::zeros({to_proj->weight.size(0)}, to_proj->weight.options());
	fused_policy_matrix = register_buffer("fused_policy_matrix", torch::matmul(from_proj->weight.transpose(0, 1), to_proj->weight).div(scale).detach().clone());
	fused_source_linear = register_buffer("fused_source_linear", torch::matmul(from_proj->weight.transpose(0, 1), to_bias).div(scale).detach().clone());
	fused_destination_linear = register_buffer("fused_destination_linear", torch::matmul(to_proj->weight.transpose(0, 1), from_bias).div(scale).detach().clone());
	fused_constant = register_buffer("fused_constant", torch::dot(from_bias, to_bias).div(scale).detach().clone());
	inference_fused = true;
}

void ActionHeadImpl::save(torch::serialize::OutputArchive &archive) const {
	save_without_buffers(*this, archive,
	    {"action_metadata", "ordinary_actions", "promotion_actions", "fused_policy_matrix", "fused_source_linear", "fused_destination_linear", "fused_constant"});
}

void ActionHeadImpl::load(torch::serialize::InputArchive &archive) {
	load_without_buffers(*this, archive,
	    {"action_metadata", "ordinary_actions", "promotion_actions", "fused_policy_matrix", "fused_source_linear", "fused_destination_linear", "fused_constant"});
}

// Pool the square representation with a learned read-only query before Value prediction.
ValueHeadImpl::ValueHeadImpl(int channels) {
	norm = register_module("norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({channels})));
	query = register_parameter("query", torch::zeros({channels}));
	value = register_module("value", torch::nn::Sequential(torch::nn::Linear(channels, 256), torch::nn::ReLU(), torch::nn::Linear(256, 1), torch::nn::Tanh()));
}

// Pool the contextual square tokens and return the unbounded Value coordinate.
torch::Tensor ValueHeadImpl::logit(torch::Tensor square_tokens) {
	if (square_tokens.dim() != 3 || square_tokens.size(1) != kBoardSquares || square_tokens.size(2) != query.size(0)) {
		throw std::runtime_error("expected Melano square tokens [batch, 64, channels]");
	}
	auto normalized = norm->forward(square_tokens);
	auto scores = torch::matmul(normalized, query) / std::sqrt(static_cast<double>(query.size(0)));
	auto weights = torch::softmax(scores, 1);
	auto pooled = torch::bmm(weights.unsqueeze(1), normalized).squeeze(1);
	auto hidden = value->ptr<torch::nn::LinearImpl>(0)->forward(pooled);
	hidden = value->ptr<torch::nn::ReLUImpl>(1)->forward(hidden);
	return value->ptr<torch::nn::LinearImpl>(2)->forward(hidden);
}

// Bound the training coordinate to the side-to-move Value interval.
torch::Tensor ValueHeadImpl::forward(torch::Tensor square_tokens) {
	return value->ptr<torch::nn::TanhImpl>(3)->forward(logit(std::move(square_tokens)));
}

// Stack geometry-attention blocks and attach the policy and value heads.
ModelImpl::ModelImpl(int channels, int blocks) {
	channels = require_positive(channels, "channels");
	blocks = require_positive(blocks, "blocks");
	state_embedding = register_module("state_embedding", StateEmbedding(channels));
	geometry_templates = register_parameter("geometry_templates", torch::empty({kGabTemplateCount, kTokenCount, kTokenCount}));
	torch::nn::init::xavier_normal_(geometry_templates.view({kGabTemplateCount, kTokenCount * kTokenCount}));
	center_geometry_templates();
	trunk = register_module("trunk", torch::nn::ModuleList());
	for (int index = 0; index < blocks; ++index) {
		trunk->push_back(GeometryAttentionBlock(channels));
	}
	policy_head = register_module("policy_head", ActionHead(channels));
	value_head = register_module("value_head", ValueHead(channels));
}

// Produce policy logits and V(s) from the shared exact-state representation.
std::tuple<torch::Tensor, torch::Tensor> ModelImpl::forward(torch::Tensor state) {
	auto squares = state_embedding->forward(state);
	for (std::size_t index = 0; index < trunk->size(); ++index) {
		squares = trunk->ptr<GeometryAttentionBlockImpl>(index)->forward(squares, geometry_templates);
	}
	return {policy_head->forward(squares), value_head->forward(squares)};
}

// Share the complete encoder with inference while exposing the pre-tanh Value coordinate.
std::tuple<torch::Tensor, torch::Tensor> ModelImpl::forward_training(torch::Tensor state) {
	auto squares = state_embedding->forward(state);
	for (std::size_t index = 0; index < trunk->size(); ++index) {
		squares = trunk->ptr<GeometryAttentionBlockImpl>(index)->forward(squares, geometry_templates);
	}
	return {policy_head->forward(squares), value_head->logit(squares)};
}

// Reuse the shared encoder while evaluating only the legal Policy actions requested by search.
std::tuple<torch::Tensor, torch::Tensor> ModelImpl::forward_legal(torch::Tensor state, torch::Tensor legal_indices) {
	auto squares = state_embedding->forward(state);
	for (std::size_t index = 0; index < trunk->size(); ++index) {
		squares = trunk->ptr<GeometryAttentionBlockImpl>(index)->forward(squares, geometry_templates);
	}
	return {policy_head->forward_legal(squares, legal_indices), value_head->forward(squares)};
}

void ModelImpl::fuse_for_inference() {
	if (is_training()) {
		throw std::logic_error("Melano model must be in evaluation mode before fusion");
	}
	policy_head->fuse_for_inference();
}

void ModelImpl::center_geometry_templates() {
	center_gab_rows(geometry_templates);
}

// Sum tensor element counts rather than serialized bytes or optimizer state.
std::int64_t parameter_count(const Model &model) {
	std::int64_t count = 0;
	for (const auto &parameter : model->parameters()) {
		count += parameter.numel();
	}
	return count;
}

} // namespace melano
