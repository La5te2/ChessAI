#pragma once

#include <cstdint>

#include <torch/torch.h>

#include "melano/game.hpp"

namespace melano {

inline constexpr int kTokenCount = kBoardSquares + 1;
inline constexpr int kGeometryRelations = 32;

int attention_heads_for_channels(int channels);
torch::Tensor build_geometry_relation_ids();

struct StateEmbeddingImpl : torch::nn::Module {
	explicit StateEmbeddingImpl(int channels);
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
	explicit GeometryAttentionBlockImpl(int channels);
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
	explicit ActionHeadImpl(int channels);
	torch::Tensor forward(torch::Tensor square_tokens);

	torch::nn::LayerNorm norm{nullptr};
	torch::nn::Linear from_proj{nullptr};
	torch::nn::Linear to_proj{nullptr};
	torch::nn::Linear underpromotion{nullptr};
};
TORCH_MODULE(ActionHead);

struct ValueHeadImpl : torch::nn::Module {
	explicit ValueHeadImpl(int channels);
	torch::Tensor forward(torch::Tensor tokens);

	torch::nn::LayerNorm norm{nullptr};
	torch::nn::Sequential value{nullptr};
};
TORCH_MODULE(ValueHead);

struct AdvantageHeadImpl : torch::nn::Module {
	explicit AdvantageHeadImpl(int channels);
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
	ModelImpl(int channels = 128, int blocks = 10);
	ModelOutput forward(torch::Tensor state);
	int channels() const noexcept;
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

std::int64_t parameter_count(const Model &model);

} // namespace melano
