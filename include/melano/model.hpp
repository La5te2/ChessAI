#pragma once

// Melano geometry-aware transformer with policy, value, and non-positive advantage heads.

#include <cstdint>
#include <torch/torch.h>
#include "melano/game.hpp"

namespace melano {

inline constexpr int kTokenCount = kBoardSquares + 1;
inline constexpr int kGeometryRelations = 32;

/// Chooses the largest supported attention-head count that evenly divides channels.
int attention_heads_for_channels(int channels);
/// Builds static relation ids for every global/square token pair.
torch::Tensor build_geometry_relation_ids();

struct StateEmbeddingImpl : torch::nn::Module {
	/// Builds embeddings for pieces, squares, side, castling, en-passant, and a global token.
	explicit StateEmbeddingImpl(int channels);
	/// Converts [N, 67] encoded states into [N, 65, C] transformer tokens.
	torch::Tensor forward(torch::Tensor state);

	torch::nn::Embedding piece{nullptr};
	torch::nn::Embedding square{nullptr};
	torch::nn::Embedding side{nullptr};
	torch::nn::Embedding castling{nullptr};
	torch::nn::Embedding ep_file{nullptr};
	torch::Tensor global_token;
	torch::Tensor square_indices;
};
TORCH_MODULE(StateEmbedding);

struct GeometryAttentionBlockImpl : torch::nn::Module {
	/// Builds pre-normalized multi-head attention with static and position-dependent geometry bias.
	explicit GeometryAttentionBlockImpl(int channels);
	/// Applies geometry-biased self-attention and a residual feed-forward transform.
	torch::Tensor forward(torch::Tensor tokens);

	int channels;
	int heads;
	int head_dim;
	torch::Tensor position;
	torch::Tensor relation_ids;
	torch::nn::LayerNorm norm1{nullptr};
	torch::nn::Linear qkv{nullptr};
	torch::nn::Linear out{nullptr};
	torch::nn::Embedding relation_bias{nullptr};
	torch::nn::Sequential dynamic_relation{nullptr};
	torch::nn::LayerNorm norm2{nullptr};
	torch::nn::Sequential ffn{nullptr};
};
TORCH_MODULE(GeometryAttentionBlock);

struct ActionHeadImpl : torch::nn::Module {
	/// Builds source/destination projections plus explicit underpromotion logits.
	explicit ActionHeadImpl(int channels);
	/// Maps 64 square tokens to Melano's 4672 action logits.
	torch::Tensor forward(torch::Tensor square_tokens);

	torch::nn::LayerNorm norm{nullptr};
	torch::nn::Linear from_proj{nullptr};
	torch::nn::Linear to_proj{nullptr};
	torch::nn::Linear underpromotion{nullptr};
};
TORCH_MODULE(ActionHead);

struct ValueHeadImpl : torch::nn::Module {
	/// Builds a bounded side-to-move value predictor over the global token.
	explicit ValueHeadImpl(int channels);
	/// Produces V(s) in [-1, 1] from the transformed global token.
	torch::Tensor forward(torch::Tensor tokens);

	torch::nn::LayerNorm norm{nullptr};
	torch::nn::Sequential value{nullptr};
};
TORCH_MODULE(ValueHead);

struct AdvantageHeadImpl : torch::nn::Module {
	/// Builds the action-shaped head used to predict non-positive A(s,a).
	explicit AdvantageHeadImpl(int channels);
	/// Produces A(s,a) in [-2, 0] as -2*tanh(raw)^2.
	torch::Tensor forward(torch::Tensor square_tokens);

	ActionHead action_head{nullptr};
};
TORCH_MODULE(AdvantageHead);

struct ModelOutput {
	torch::Tensor policy;
	torch::Tensor value;
	torch::Tensor advantages;
};

struct ModelImpl : torch::nn::Module {
	/// Builds the token embedding, geometry-attention trunk, and P/V/A heads.
	ModelImpl(int channels = 128, int blocks = 10);
	/// Returns policy logits, side-to-move V(s), and action advantages A(s,a).
	ModelOutput forward(torch::Tensor state);
	/// Returns the transformer embedding width stored in the checkpoint descriptor.
	int channels() const noexcept;
	/// Returns the number of geometry-attention blocks stored in the descriptor.
	int blocks() const noexcept;

	StateEmbedding state_embedding{nullptr};
	torch::nn::Sequential trunk{nullptr};
	ActionHead policy_head{nullptr};
	ValueHead value_head{nullptr};
	AdvantageHead advantage_head{nullptr};

	private:
	int channels_;
	int blocks_;
};
TORCH_MODULE(Model);

/// Counts all trainable and non-trainable model parameter elements.
std::int64_t parameter_count(const Model &model);

} // namespace melano
