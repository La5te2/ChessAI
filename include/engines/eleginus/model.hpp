#pragma once

#include "eleginus/formula.hpp"
#include <array>
#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>

namespace eleginus {

	inline constexpr std::uint32_t kArchitectureType = 3;
	inline constexpr float kCentipawnsPerLogit = 150.0F;
	inline constexpr std::size_t kRelationCount = kFormulaCount * kFormulaCount;
	inline constexpr std::size_t kParameterCount = kFormulaCount + kRelationCount;
	int centipawns(float h);

	class Model {
	public:
		Model();
		static std::span<const float> initial() noexcept;
		static Model load(const std::filesystem::path &path);
		void save(const std::filesystem::path &path) const;

		float score(const chess::Board &board) const;
		float score(std::span<const Feature> x) const;
		int centipawns(const chess::Board &board) const;
		void extract(const chess::Board &board, std::vector<Feature> &out) const;

		std::size_t formulas() const noexcept { return kFormulaCount; }
		std::size_t relationIndex(std::size_t row, std::size_t condition) const;
		std::span<const float> base() const noexcept { return {p.data(), kFormulaCount}; }
		std::span<const float> relations() const noexcept { return {p.data() + kFormulaCount, kRelationCount}; }
		const std::vector<float> &params() const noexcept { return p; }
		void update(std::span<const float> values);
		bool columnActive(std::size_t condition) const noexcept { return activeColumns[condition] != 0; }

	private:
		void indexRelations() noexcept;
		std::vector<float> p;
		std::array<std::uint8_t, kFormulaCount> activeColumns{};
	};

	class Accumulator {
	public:
		explicit Accumulator(const Model &model);
		void reset(const chess::Board &board);
		void push();
		void pop();
		float score(const chess::Board &board);

	private:
		struct Change {
			std::uint16_t index;
			float score;
			float condition;
		};

		void extract(const chess::Board &board);
		void addColumn(std::size_t condition, float scale);
		void refresh(const chess::Board &board);

		const Model &net;
		std::array<float, kFormulaCount> scores{};
		std::array<float, kFormulaCount> conditions{};
		std::array<float, kFormulaCount> dynamic{};
		std::array<float, kFormulaCount> nextScores{};
		std::array<float, kFormulaCount> nextConditions{};
		std::vector<Feature> features;
		std::vector<Change> changes;
		std::vector<std::size_t> frames;
		std::vector<std::uint8_t> materialized;
	};

} // namespace eleginus
