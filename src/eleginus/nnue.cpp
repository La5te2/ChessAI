// Implements sparse feature deltas and independent incremental Policy/Value inference.

#include "eleginus/nnue.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace eleginus {

namespace {

int oriented_square(int square, chess::Color perspective) {
	return perspective == chess::Color::WHITE ? square : square ^ 56;
}

int relative_piece_index(const chess::Piece &piece, chess::Color perspective) {
	const int type = static_cast<int>(piece.type().internal());
	if (type < 0 || type > 5) {
		throw std::runtime_error("invalid piece in Eleginus feature encoder");
	}
	return type + (piece.color() == perspective ? 0 : 6);
}

int castling_feature(const chess::Board &board, chess::Color perspective) {
	const auto rights = board.castlingRights();
	const auto opponent = ~perspective;
	int mask = 0;
	if (rights.has(perspective, chess::Board::CastlingRights::Side::KING_SIDE))
		mask |= 1;
	if (rights.has(perspective, chess::Board::CastlingRights::Side::QUEEN_SIDE))
		mask |= 2;
	if (rights.has(opponent, chess::Board::CastlingRights::Side::KING_SIDE))
		mask |= 4;
	if (rights.has(opponent, chess::Board::CastlingRights::Side::QUEEN_SIDE))
		mask |= 8;
	return kCastlingFeatureBase + mask;
}

int en_passant_feature(const chess::Board &board) {
	const int file = board.enpassantSq().is_valid() ? board.enpassantSq().file() + 1 : 0;
	return kEnPassantFeatureBase + file;
}

void require_size(const std::vector<float> &values, std::size_t expected, const char *name) {
	if (values.size() != expected) {
		throw std::invalid_argument(std::string("invalid Eleginus ") + name + " size");
	}
}

void add_feature(std::vector<float> &accumulator, const std::vector<float> &table, int width,
				 int feature, float sign) {
	const auto offset = static_cast<std::size_t>(feature) * static_cast<std::size_t>(width);
	for (int channel = 0; channel < width; ++channel) {
		accumulator[static_cast<std::size_t>(channel)] +=
			sign * table[offset + static_cast<std::size_t>(channel)];
	}
}

FloatAccumulator refresh_accumulator(const chess::Board &board, const std::vector<float> &table,
									 const std::vector<float> &bias, int width) {
	const auto encoded = encode_features(board);
	FloatAccumulator result;
	result.white_to_move = encoded.white_to_move;
	for (int perspective = 0; perspective < kPerspectiveCount; ++perspective) {
		auto &values = result.perspective[static_cast<std::size_t>(perspective)];
		values = bias;
		for (const int feature : encoded.perspective[static_cast<std::size_t>(perspective)]) {
			if (feature != kPaddingFeature) {
				add_feature(values, table, width, feature, 1.0F);
			}
		}
	}
	return result;
}

FloatAccumulator update_accumulator(const FloatAccumulator &current, const chess::Board &before,
									const chess::Board &after,
									const std::vector<float> &table, int width) {
	const auto old_features = encode_features(before);
	const auto new_features = encode_features(after);
	FloatAccumulator result = current;
	result.white_to_move = new_features.white_to_move;
	for (int perspective = 0; perspective < kPerspectiveCount; ++perspective) {
		const auto &old_list = old_features.perspective[static_cast<std::size_t>(perspective)];
		const auto &new_list = new_features.perspective[static_cast<std::size_t>(perspective)];
		auto &values = result.perspective[static_cast<std::size_t>(perspective)];
		for (const int feature : old_list) {
			if (feature != kPaddingFeature &&
				std::find(new_list.begin(), new_list.end(), feature) == new_list.end()) {
				add_feature(values, table, width, feature, -1.0F);
			}
		}
		for (const int feature : new_list) {
			if (feature != kPaddingFeature &&
				std::find(old_list.begin(), old_list.end(), feature) == old_list.end()) {
				add_feature(values, table, width, feature, 1.0F);
			}
		}
	}
	return result;
}

std::vector<float> oriented_input(const FloatAccumulator &accumulator, int width) {
	std::vector<float> input(static_cast<std::size_t>(width * 2));
	const int first = accumulator.white_to_move ? 0 : 1;
	const int second = 1 - first;
	for (int channel = 0; channel < width; ++channel) {
		input[static_cast<std::size_t>(channel)] =
			std::clamp(accumulator.perspective[static_cast<std::size_t>(first)]
						 [static_cast<std::size_t>(channel)],
					   0.0F, 1.0F);
		input[static_cast<std::size_t>(width + channel)] =
			std::clamp(accumulator.perspective[static_cast<std::size_t>(second)]
						 [static_cast<std::size_t>(channel)],
					   0.0F, 1.0F);
	}
	return input;
}

std::vector<float> linear_relu(const std::vector<float> &input,
							   const std::vector<float> &weight,
							   const std::vector<float> &bias, int output_width) {
	std::vector<float> output(static_cast<std::size_t>(output_width));
	for (int row = 0; row < output_width; ++row) {
		float sum = bias[static_cast<std::size_t>(row)];
		const auto offset = static_cast<std::size_t>(row) * input.size();
		for (std::size_t column = 0; column < input.size(); ++column) {
			sum += weight[offset + column] * input[column];
		}
		output[static_cast<std::size_t>(row)] = std::max(0.0F, sum);
	}
	return output;
}

std::vector<float> legal_softmax(const std::vector<float> &hidden,
								 const PolicyWeights &weights,
								 const std::vector<chess::Move> &moves,
								 chess::Color side_to_move) {
	if (moves.empty()) {
		return {};
	}
	std::vector<float> logits(moves.size());
	float maximum = -std::numeric_limits<float>::infinity();
	for (std::size_t move_index = 0; move_index < moves.size(); ++move_index) {
		const int action = move_to_index(moves[move_index], side_to_move);
		float logit = weights.output_bias[static_cast<std::size_t>(action)];
		const auto offset = static_cast<std::size_t>(action) * kPolicyHiddenWidth;
		for (int channel = 0; channel < kPolicyHiddenWidth; ++channel) {
			logit += weights.output_weight[offset + static_cast<std::size_t>(channel)] *
				hidden[static_cast<std::size_t>(channel)];
		}
		logits[move_index] = logit;
		maximum = std::max(maximum, logit);
	}
	float denominator = 0.0F;
	for (float &logit : logits) {
		logit = std::exp(logit - maximum);
		denominator += logit;
	}
	for (float &probability : logits) {
		probability /= denominator;
	}
	return logits;
}

} // namespace

EncodedFeatures encode_features(const chess::Board &board) {
	EncodedFeatures result;
	for (auto &perspective : result.perspective) {
		perspective.fill(kPaddingFeature);
	}
	result.white_to_move = board.sideToMove() == chess::Color::WHITE;

	for (int perspective_index = 0; perspective_index < kPerspectiveCount; ++perspective_index) {
		const auto perspective =
			perspective_index == 0 ? chess::Color::WHITE : chess::Color::BLACK;
		const int king = oriented_square(board.kingSq(perspective).index(), perspective);
		auto &features = result.perspective[static_cast<std::size_t>(perspective_index)];
		int cursor = 0;
		for (int square = 0; square < kBoardSquares; ++square) {
			const auto piece = board.at(chess::Square(square));
			if (piece == chess::Piece::NONE) {
				continue;
			}
			const int piece_index = relative_piece_index(piece, perspective);
			const int oriented = oriented_square(square, perspective);
			features[static_cast<std::size_t>(cursor++)] =
				king * 12 * 64 + piece_index * 64 + oriented;
		}
		features[static_cast<std::size_t>(cursor++)] = castling_feature(board, perspective);
		features[static_cast<std::size_t>(cursor)] = en_passant_feature(board);
	}
	return result;
}

CpuPolicy::CpuPolicy(PolicyWeights weights) : weights_(std::move(weights)) {
	require_size(weights_.feature_table,
		static_cast<std::size_t>(kFeatureVocabulary) * kPolicyAccumulatorWidth,
		"Policy feature table");
	require_size(weights_.accumulator_bias, kPolicyAccumulatorWidth,
		"Policy accumulator bias");
	require_size(weights_.hidden_weight,
		static_cast<std::size_t>(kPolicyHiddenWidth) * kPolicyAccumulatorWidth * 2,
		"Policy hidden weight");
	require_size(weights_.hidden_bias, kPolicyHiddenWidth, "Policy hidden bias");
	require_size(weights_.output_weight,
		static_cast<std::size_t>(kActionSize) * kPolicyHiddenWidth,
		"Policy output weight");
	require_size(weights_.output_bias, kActionSize, "Policy output bias");
}

FloatAccumulator CpuPolicy::refresh(const chess::Board &board) const {
	return refresh_accumulator(board, weights_.feature_table, weights_.accumulator_bias,
		kPolicyAccumulatorWidth);
}

FloatAccumulator CpuPolicy::update(const FloatAccumulator &current, const chess::Board &before,
								   const chess::Board &after) const {
	return update_accumulator(current, before, after, weights_.feature_table,
		kPolicyAccumulatorWidth);
}

std::vector<float> CpuPolicy::evaluate(const FloatAccumulator &accumulator,
									   const std::vector<chess::Move> &moves) const {
	const auto input = oriented_input(accumulator, kPolicyAccumulatorWidth);
	const auto hidden =
		linear_relu(input, weights_.hidden_weight, weights_.hidden_bias, kPolicyHiddenWidth);
	return legal_softmax(hidden, weights_, moves,
		accumulator.white_to_move ? chess::Color::WHITE : chess::Color::BLACK);
}

CpuValue::CpuValue(ValueWeights weights) : weights_(std::move(weights)) {
	require_size(weights_.feature_table,
		static_cast<std::size_t>(kFeatureVocabulary) * kValueAccumulatorWidth,
		"Value feature table");
	require_size(weights_.accumulator_bias, kValueAccumulatorWidth, "Value accumulator bias");
	require_size(weights_.hidden_weight,
		static_cast<std::size_t>(kValueHiddenWidth) * kValueAccumulatorWidth * 2,
		"Value hidden weight");
	require_size(weights_.hidden_bias, kValueHiddenWidth, "Value hidden bias");
	require_size(weights_.bottleneck_weight,
		static_cast<std::size_t>(kValueBottleneckWidth) * kValueHiddenWidth,
		"Value bottleneck weight");
	require_size(weights_.bottleneck_bias, kValueBottleneckWidth,
		"Value bottleneck bias");
	require_size(weights_.output_weight, kValueBottleneckWidth, "Value output weight");
}

FloatAccumulator CpuValue::refresh(const chess::Board &board) const {
	return refresh_accumulator(board, weights_.feature_table, weights_.accumulator_bias,
		kValueAccumulatorWidth);
}

FloatAccumulator CpuValue::update(const FloatAccumulator &current, const chess::Board &before,
								  const chess::Board &after) const {
	return update_accumulator(current, before, after, weights_.feature_table,
		kValueAccumulatorWidth);
}

float CpuValue::evaluate(const FloatAccumulator &accumulator) const {
	const auto input = oriented_input(accumulator, kValueAccumulatorWidth);
	const auto hidden =
		linear_relu(input, weights_.hidden_weight, weights_.hidden_bias, kValueHiddenWidth);
	const auto bottleneck =
		linear_relu(hidden, weights_.bottleneck_weight, weights_.bottleneck_bias,
			kValueBottleneckWidth);
	float output = weights_.output_bias;
	for (int channel = 0; channel < kValueBottleneckWidth; ++channel) {
		output += weights_.output_weight[static_cast<std::size_t>(channel)] *
				  bottleneck[static_cast<std::size_t>(channel)];
	}
	return 1.0F / (1.0F + std::exp(-std::clamp(output, -30.0F, 30.0F)));
}

} // namespace eleginus
