// Implements Melano's geometry-aware token network and P/V/A heads.

#include "melano/model.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace melano {

// Select a practical multi-head factor without requiring channels to use one fixed width.
int attention_heads_for_channels(int channels) {
	for (const int heads : {8, 4, 2}) {
		if (channels % heads == 0) {
			return heads;
		}
	}
	return 1;
}

namespace {

// Classify an ordered square pair by chessboard geometry for attention bias lookup.
int square_geometry_relation(int source, int target) {
	const int source_rank = source / 8;
	const int source_file = source % 8;
	const int target_rank = target / 8;
	const int target_file = target % 8;
	const int dr = target_rank - source_rank;
	const int dc = target_file - source_file;
	const int adr = std::abs(dr);
	const int adc = std::abs(dc);
	int value = 0;
	if (source == target) {
		value = 0;
	} else if (dr == 0) {
		value = adc;
	} else if (dc == 0) {
		value = 7 + adr;
	} else if (adr == adc) {
		value = 14 + adr;
	} else if ((adr == 1 && adc == 2) || (adr == 2 && adc == 1)) {
		value = 22;
	} else if (std::max(adr, adc) == 1) {
		value = 23;
	} else {
		value = 24 + std::min(6, adr + adc - 2);
	}
	return value + 1;
}

} // namespace

// Materialize relation ids once; registered buffers move with the model but are not trained.
torch::Tensor build_geometry_relation_ids() {
	auto relation = torch::zeros({kTokenCount, kTokenCount}, torch::kInt64);
	auto accessor = relation.accessor<std::int64_t, 2>();
	for (int source = 0; source < kBoardSquares; ++source) {
		for (int target = 0; target < kBoardSquares; ++target) {
			accessor[source + 1][target + 1] = square_geometry_relation(source, target);
		}
	}
	return relation;
}

// Embed board contents and global rule state into one global plus 64 square tokens.
StateEmbeddingImpl::StateEmbeddingImpl(int channels) {
	piece = register_module("piece", torch::nn::Embedding(13, channels));
	square = register_module("square", torch::nn::Embedding(kBoardSquares, channels));
	side = register_module("side", torch::nn::Embedding(2, channels));
	castling = register_module("castling", torch::nn::Embedding(16, channels));
	ep_file = register_module("ep_file", torch::nn::Embedding(9, channels));
	global_token = register_parameter("global_token", torch::zeros({1, 1, channels}));
	square_indices = register_buffer("square_indices", torch::arange(kBoardSquares, torch::kInt64));
}

// Add piece, absolute-square, and rule-context embeddings before token concatenation.
torch::Tensor StateEmbeddingImpl::forward(torch::Tensor state) {
	if (state.dim() != 2 || state.size(1) != kStateFeatures) {
		throw std::runtime_error("expected Melano state [batch, 67]");
	}
	state = state.to(torch::kInt64);
	auto pieces = state.index({torch::indexing::Slice(), torch::indexing::Slice(0, kBoardSquares)})
				  .clamp(0, 12);
	auto side_token = state.index({torch::indexing::Slice(), 64}).clamp(0, 1);
	auto castling_token = state.index({torch::indexing::Slice(), 65}).clamp(0, 15);
	auto ep_token = state.index({torch::indexing::Slice(), 66}).clamp(0, 8);
	auto context = side->forward(side_token) + castling->forward(castling_token) +
				   ep_file->forward(ep_token);
	auto squares = piece->forward(pieces) + square->forward(square_indices).unsqueeze(0) +
				   context.unsqueeze(1);
	auto global = global_token.expand({state.size(0), -1, -1}) + context.unsqueeze(1);
	return torch::cat({global, squares}, 1);
}

// Construct pre-norm attention with learned static geometry and state-conditioned bias.
GeometryAttentionBlockImpl::GeometryAttentionBlockImpl(int channel_count)
	: channels(channel_count), heads(attention_heads_for_channels(channel_count)),
	  head_dim(channel_count / heads) {
	position = register_parameter("position", torch::zeros({1, kTokenCount, channels}));
	relation_ids = register_buffer("relation_ids", build_geometry_relation_ids());
	norm1 = register_module("norm1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({channels})));
	qkv = register_module("qkv", torch::nn::Linear(channels, channels * 3));
	out = register_module("out", torch::nn::Linear(channels, channels));
	relation_bias = register_module("relation_bias", torch::nn::Embedding(kGeometryRelations, heads));
	dynamic_relation = register_module(
		"dynamic_relation",
		torch::nn::Sequential(
			torch::nn::LayerNorm(torch::nn::LayerNormOptions({channels})),
			torch::nn::Linear(channels, channels), torch::nn::GELU(),
			torch::nn::Linear(channels, heads * kGeometryRelations)));
	norm2 = register_module("norm2", torch::nn::LayerNorm(torch::nn::LayerNormOptions({channels})));
	ffn = register_module(
		"ffn", torch::nn::Sequential(torch::nn::Linear(channels, channels * 4), torch::nn::GELU(),
								 torch::nn::Linear(channels * 4, channels)));
}

// Compute softmax((QK^T)/sqrt(d) + static_bias + dynamic_bias)V with residual updates.
torch::Tensor GeometryAttentionBlockImpl::forward(torch::Tensor tokens) {
	if (tokens.dim() != 3 || tokens.size(1) != kTokenCount || tokens.size(2) != channels) {
		throw std::runtime_error("invalid Melano geometry-attention token shape");
	}
	const auto batch = tokens.size(0);
	tokens = tokens + position;
	auto packed = qkv->forward(norm1->forward(tokens))
				  .view({batch, kTokenCount, 3, heads, head_dim});
	auto parts = packed.unbind(2);
	auto query = parts[0].transpose(1, 2);
	auto key = parts[1].transpose(1, 2);
	auto value = parts[2].transpose(1, 2);
	auto scores = torch::matmul(query, key.transpose(-2, -1)) / std::sqrt(static_cast<double>(head_dim));
	auto static_bias = relation_bias->forward(relation_ids).permute({2, 0, 1}).unsqueeze(0);
	auto dynamic = dynamic_relation
				   ->forward(tokens.index({torch::indexing::Slice(), 0}))
				   .view({batch, heads, kGeometryRelations});
	auto dynamic_bias = dynamic.index_select(2, relation_ids.reshape({-1}))
						.view({batch, heads, kTokenCount, kTokenCount});
	auto attention = torch::softmax(scores + static_bias + dynamic_bias, -1);
	auto attention_output = torch::matmul(attention, value)
							.transpose(1, 2)
							.contiguous()
							.view({batch, kTokenCount, channels});
	tokens = tokens + out->forward(attention_output);
	return tokens + ffn->forward(norm2->forward(tokens));
}

// Build factorized from/to logits and a dedicated underpromotion suffix.
ActionHeadImpl::ActionHeadImpl(int channels) {
	norm = register_module("norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({channels})));
	from_proj = register_module("from_proj", torch::nn::Linear(channels, channels));
	to_proj = register_module("to_proj", torch::nn::Linear(channels, channels));
	underpromotion = register_module("underpromotion", torch::nn::Linear(channels, kUnderpromotionPlanes));
}

// Score ordinary moves by scaled source-destination dot products, then append promotions.
torch::Tensor ActionHeadImpl::forward(torch::Tensor square_tokens) {
	if (square_tokens.dim() != 3 || square_tokens.size(1) != kBoardSquares) {
		throw std::runtime_error("expected Melano square tokens [batch, 64, channels]");
	}
	auto normalized = norm->forward(square_tokens);
	auto from = from_proj->forward(normalized);
	auto to = to_proj->forward(normalized);
	auto from_to = torch::matmul(from, to.transpose(1, 2)) /
				   std::sqrt(static_cast<double>(from.size(2)));
	auto promotions = underpromotion->forward(normalized);
	return torch::cat(
		{from_to.contiguous().view({normalized.size(0), kBoardSquares * kBoardSquares}),
		 promotions.contiguous().view({normalized.size(0), kBoardSquares * kUnderpromotionPlanes})},
		1);
}

// Map the global token to a bounded side-to-move value.
ValueHeadImpl::ValueHeadImpl(int channels) {
	norm = register_module("norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({channels})));
	value = register_module(
		"value", torch::nn::Sequential(torch::nn::Linear(channels, 256), torch::nn::ReLU(),
									 torch::nn::Linear(256, 1), torch::nn::Tanh()));
}

// Read only the global token because attention has already pooled square information into it.
torch::Tensor ValueHeadImpl::forward(torch::Tensor tokens) {
	return value->forward(norm->forward(tokens.index({torch::indexing::Slice(), 0})));
}

// Reuse the action-shaped projection while initializing the final mappings near zero.
AdvantageHeadImpl::AdvantageHeadImpl(int channels) {
	action_head = register_module("action_head", ActionHead(channels));
	torch::nn::init::normal_(action_head->to_proj->weight, 0.0, 0.01);
	torch::nn::init::zeros_(action_head->to_proj->bias);
	torch::nn::init::normal_(action_head->underpromotion->weight, 0.0, 0.01);
	torch::nn::init::zeros_(action_head->underpromotion->bias);
}

// Enforce the project invariant A(s,a) <= 0 with -2*tanh(raw)^2.
torch::Tensor AdvantageHeadImpl::forward(torch::Tensor square_tokens) {
	auto raw = torch::tanh(action_head->forward(square_tokens));
	return -2.0 * raw.square();
}

// Condition one geometry-attention transition on the selected action id.
LatentDynamicsImpl::LatentDynamicsImpl(int channel_count) : channels(channel_count) {
	action_embedding =
		register_module("action_embedding", torch::nn::Embedding(kActionSize, channels));
	action_projection =
		register_module("action_projection", torch::nn::Linear(channels, channels));
	update_gate = register_module("update_gate", torch::nn::Linear(channels, channels));
	transition = register_module("transition", GeometryAttentionBlock(channels));
	output_norm =
		register_module("output_norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({channels})));
	// Begin with a conservative residual update while still allowing immediate gradients.
	torch::nn::init::zeros_(update_gate->weight);
	torch::nn::init::constant_(update_gate->bias, -2.0);
}

// Apply z'=LN(z+sigmoid(g(a))*(T(z+c(a))-z)) as an action-conditioned latent world step.
torch::Tensor LatentDynamicsImpl::forward(torch::Tensor tokens, torch::Tensor actions) {
	if (tokens.dim() != 3 || tokens.size(1) != kTokenCount || tokens.size(2) != channels) {
		throw std::runtime_error("invalid Melano latent-dynamics token shape");
	}
	actions = actions.to(torch::kInt64).reshape({-1});
	if (actions.size(0) != tokens.size(0)) {
		throw std::runtime_error("latent-dynamics action batch does not match token batch");
	}
	auto action = action_embedding->forward(actions);
	auto conditioned = tokens + action_projection->forward(action).unsqueeze(1);
	auto proposed = transition->forward(conditioned);
	auto gate = torch::sigmoid(update_gate->forward(action)).unsqueeze(1);
	return output_norm->forward(tokens + gate * (proposed - tokens));
}

// Stack geometry-attention blocks and attach independent policy, value, and advantage heads.
ModelImpl::ModelImpl(int channels, int blocks)
	: channels_(channels), blocks_(std::max(1, blocks)) {
	state_embedding = register_module("state_embedding", StateEmbedding(channels_));
	trunk = register_module("trunk", torch::nn::Sequential());
	for (int index = 0; index < blocks_; ++index) {
		trunk->push_back(GeometryAttentionBlock(channels_));
	}
	policy_head = register_module("policy_head", ActionHead(channels_));
	value_head = register_module("value_head", ValueHead(channels_));
	advantage_head = register_module("advantage_head", AdvantageHead(channels_));
	dynamics = register_module("dynamics", LatentDynamics(channels_));
}

// Encode one exact state with the embedding and shared geometry transformer.
torch::Tensor ModelImpl::encode(torch::Tensor state) {
	return trunk->forward(state_embedding->forward(state));
}

// Produce all three heads from one contextual token representation.
ModelOutput ModelImpl::predict(torch::Tensor tokens) {
	auto squares = tokens.index({torch::indexing::Slice(), torch::indexing::Slice(1, torch::indexing::None)});
	return {policy_head->forward(squares), value_head->forward(tokens),
			advantage_head->forward(squares)};
}

// Delegate one imagined action step to the registered latent transition model.
torch::Tensor ModelImpl::transition(torch::Tensor tokens, torch::Tensor actions) {
	return dynamics->forward(tokens, actions);
}

// Preserve the public exact-state inference contract while exposing staged internals for training.
ModelOutput ModelImpl::forward(torch::Tensor state) { return predict(encode(state)); }

// Expose the checkpoint-defining embedding width.
int ModelImpl::channels() const noexcept { return channels_; }
// Expose the checkpoint-defining attention depth.
int ModelImpl::blocks() const noexcept { return blocks_; }

// Sum tensor element counts rather than serialized bytes or optimizer state.
std::int64_t parameter_count(const Model &model) {
	std::int64_t count = 0;
	for (const auto &parameter : model->parameters()) {
		count += parameter.numel();
	}
	return count;
}

} // namespace melano
