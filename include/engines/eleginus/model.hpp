#pragma once

#include "eleginus/formula.hpp"
#include <array>
#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>

namespace eleginus {

	inline constexpr std::uint32_t kArchitectureType = 3;
	inline constexpr std::size_t kRelationLimit = 2048;

	struct Relation {
		std::uint16_t row = 0;
		std::uint16_t condition = 0;
	};

	class Model {
	public:
		struct Cache {
			std::array<float, kFormulaCount> score{};
			std::array<float, kFormulaCount> condition{};
			std::array<std::uint16_t, kFormulaCount> active{};
			std::uint16_t count = 0;
		};

		Model();
		static std::span<const float> initial() noexcept;
		static Model load(const std::filesystem::path &path);
		void save(const std::filesystem::path &path) const;

		float score(const chess::Board &board) const;
		float score(std::span<const Feature> x) const;
		float forward(std::span<const Feature> x, Cache &cache) const;
		void backward(const Cache &cache, float delta, std::span<float> grad) const;
		void weights(std::span<const Feature> x, std::vector<float> &out) const;
		int centipawns(const chess::Board &board) const;
		void extract(const chess::Board &board, std::vector<Feature> &out) const;

		bool activate(std::uint16_t row, std::uint16_t condition);
		bool active(std::uint16_t row, std::uint16_t condition) const;
		void prune(float threshold);
		std::size_t formulas() const noexcept { return kFormulaCount; }
		std::span<const Relation> relations() const noexcept { return links; }
		const std::vector<float> &params() const noexcept { return p; }
		std::vector<float> &params() noexcept { return p; }

	private:
		std::vector<float> p;
		std::vector<Relation> links;
		std::vector<std::uint16_t> rows, conditions;
	};

} // namespace eleginus
