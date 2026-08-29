#pragma once

// Melano geometry-aware transformer with a shared exact-state encoder and policy/value heads.

#include "melano/game.hpp"
#include <cstdint>
#include <torch/nn.h>
#include <tuple>

namespace melano {

inline constexpr int kTokenCount = kBoardSquares;
inline constexpr int kGabTemplateCount = 64;
inline constexpr int kGabSquareChannels = 8;
inline constexpr int kGabStateChannels = 32;

struct StateEmbeddingImpl : torch::nn::Module {
	/// Builds embeddings for pieces, squares, castling, and en-passant state.
	explicit StateEmbeddingImpl(int channels);
	/// Converts [N, 66] encoded states into [N, 64, C] square tokens.
	torch::Tensor forward(torch::Tensor state);

	torch::nn::Embedding piece{nullptr};
	torch::nn::Embedding square{nullptr};
	torch::nn::Embedding castling{nullptr};
	torch::nn::Embedding ep_file{nullptr};
};
TORCH_MODULE(StateEmbedding);

struct GeometryAttentionBlockImpl : torch::nn::Module {
	/// Builds pre-normalized multi-head attention with dynamic geometric attention bias.
	explicit GeometryAttentionBlockImpl(int channels);
	/// Generates one state-dependent square-pair bias matrix per attention head.
	torch::Tensor geometry_bias(torch::Tensor tokens, torch::Tensor templates);
	/// Generates the state-dependent coefficients used by the geometric templates.
	torch::Tensor geometry_coefficients(torch::Tensor tokens);
	/// Applies geometry-biased self-attention and a residual feed-forward transform.
	torch::Tensor forward(torch::Tensor tokens, torch::Tensor templates);

	int channels;
	int heads;
	int head_dim;
	torch::Tensor position;
	torch::nn::LayerNorm norm1{nullptr};
	torch::nn::Linear qkv{nullptr};
	torch::nn::Linear out{nullptr};
	torch::nn::Linear gab_square{nullptr};
	torch::nn::Linear gab_state{nullptr};
	torch::nn::LayerNorm gab_state_norm{nullptr};
	torch::nn::Linear gab_coefficients{nullptr};
	torch::nn::LayerNorm gab_coefficient_norm{nullptr};
	torch::nn::LayerNorm norm2{nullptr};
	torch::nn::Sequential ffn{nullptr};
};
TORCH_MODULE(GeometryAttentionBlock);

struct ActionHeadImpl : torch::nn::Module {
	/// Builds source/destination projections plus explicit underpromotion logits.
	explicit ActionHeadImpl(int channels);
	/// Maps 64 square tokens to Melano's 1858 geometrically valid action logits.
	torch::Tensor forward(torch::Tensor square_tokens);
	/// Computes logits only for the requested Melano action indices [batch, legal_width].
	torch::Tensor forward_legal(torch::Tensor square_tokens, torch::Tensor legal_indices);
	/// Fuses the two ordinary-move projections into one bilinear inference form.
	void fuse_for_inference();
	void save(torch::serialize::OutputArchive &archive) const override;
	void load(torch::serialize::InputArchive &archive) override;

	torch::nn::LayerNorm norm{nullptr};
	torch::nn::Linear from_proj{nullptr};
	torch::nn::Linear to_proj{nullptr};
	torch::nn::Linear underpromotion{nullptr};
	torch::Tensor action_metadata;
	torch::Tensor ordinary_actions;
	torch::Tensor promotion_actions;
	torch::Tensor fused_policy_matrix;
	torch::Tensor fused_source_linear;
	torch::Tensor fused_destination_linear;
	torch::Tensor fused_constant;
	bool inference_fused = false;
};
TORCH_MODULE(ActionHead);

struct ValueHeadImpl : torch::nn::Module {
	/// Builds attention pooling and a bounded side-to-move value predictor.
	explicit ValueHeadImpl(int channels);
	/// Produces the unbounded scalar whose hyperbolic tangent is V(s).
	torch::Tensor logit(torch::Tensor square_tokens);
	/// Produces V(s) in [-1, 1] from the transformed square tokens.
	torch::Tensor forward(torch::Tensor square_tokens);

	torch::nn::LayerNorm norm{nullptr};
	torch::Tensor query;
	torch::nn::Sequential value{nullptr};
};
TORCH_MODULE(ValueHead);

struct ModelImpl : torch::nn::Module {
	/// Builds the token embedding, geometry-attention trunk, and policy/value heads.
	ModelImpl(int channels = 128, int blocks = 10);
	/// Returns policy logits and side-to-move V(s) for an exact board state.
	std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor state);
	/// Returns Policy logits and the unbounded Value logit used by supervised training.
	std::tuple<torch::Tensor, torch::Tensor> forward_training(torch::Tensor state);
	/// Returns legal-action logits [batch, legal_width] and the same side-to-move V(s).
	std::tuple<torch::Tensor, torch::Tensor> forward_legal(torch::Tensor state, torch::Tensor legal_indices);
	/// Precomputes evaluation-only constants without changing the represented function.
	void fuse_for_inference();
	/// Projects every GAB template row onto its zero-mean representative.
	void center_geometry_templates();

	StateEmbedding state_embedding{nullptr};
	torch::Tensor geometry_templates;
	torch::nn::ModuleList trunk{nullptr};
	ActionHead policy_head{nullptr};
	ValueHead value_head{nullptr};
};
TORCH_MODULE(Model);

/// Counts all trainable and non-trainable model parameter elements.
std::int64_t parameter_count(const Model &model);

} // namespace melano
