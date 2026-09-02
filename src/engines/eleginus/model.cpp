#include "eleginus/model.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <system_error>
#ifdef _WIN32
	#define NOMINMAX
	#define WIN32_LEAN_AND_MEAN
	#include <windows.h>
#endif

namespace eleginus {
	namespace {
		constexpr std::array<char, 8> magic{'E', 'L', 'E', 'G', 'I', 'N', 'U', 'S'};

		template <typename T> void write(std::ostream &out, const T &x) {
			out.write(reinterpret_cast<const char *>(&x), sizeof(x));
		}

		template <typename T> T read(std::istream &in) {
			T x{};
			in.read(reinterpret_cast<char *>(&x), sizeof(x));
			return x;
		}

		bool finite(std::span<const float> values) {
			return std::all_of(values.begin(), values.end(), [](float x) { return std::isfinite(x); });
		}

		int toCp(float h) {
			if (!std::isfinite(h)) throw std::runtime_error("nonfinite Eleginus evaluation");
			return static_cast<int>(std::lround(std::clamp(400.0F * h, -25000.0F, 25000.0F)));
		}

	} // namespace

	Model::Model() {
		const auto values = detail::initial();
		p.assign(values.begin(), values.end());
	}

	std::span<const float> Model::initial() noexcept {
		return detail::initial();
	}

	float Model::forward(std::span<const Feature> x, Cache &cache) const {
		const auto n = formulas();
		for (std::uint16_t i = 0; i < cache.count; ++i) {
			cache.score[cache.active[i]] = 0.0F;
			cache.condition[cache.active[i]] = 0.0F;
		}
		cache.count = 0;
		for (const auto &f : x) {
			if (f.index >= n) throw std::out_of_range("formula index exceeds model layout");
			cache.score[f.index] = static_cast<float>(f.score);
			cache.condition[f.index] = static_cast<float>(f.condition);
			cache.active[cache.count++] = f.index;
		}
		double result = 0.0;
		for (const auto &f : x) {
			result += static_cast<double>(f.score) * p[f.index];
		}
		for (std::size_t i = 0; i < links.size(); ++i) {
			result += static_cast<double>(p[n + i]) * cache.score[links[i].row] * cache.condition[links[i].condition];
		}
		return static_cast<float>(result);
	}

	float Model::score(std::span<const Feature> x) const {
		thread_local std::array<float, kFormulaCount> score{}, condition{};
		double result = 0.0;
		for (const auto &f : x) {
			if (f.index >= formulas()) throw std::out_of_range("formula index exceeds model layout");
			score[f.index] = static_cast<float>(f.score);
			condition[f.index] = static_cast<float>(f.condition);
			result += static_cast<double>(f.score) * p[f.index];
		}
		for (std::size_t i = 0; i < links.size(); ++i) {
			result += static_cast<double>(p[formulas() + i]) * score[links[i].row] * condition[links[i].condition];
		}
		for (const auto &f : x) {
			score[f.index] = 0.0F;
			condition[f.index] = 0.0F;
		}
		return static_cast<float>(result);
	}

	float Model::score(const chess::Board &board) const {
		float white;
		if (links.empty()) {
			white = FormulaSet::evaluate(board, std::span<const float>(p.data(), formulas()));
		} else {
			const auto relationWeights = std::span<const float>(p.data() + static_cast<std::ptrdiff_t>(formulas()), links.size());
			white = FormulaSet::evaluate(board, std::span<const float>(p.data(), formulas()), rows, conditions, relationWeights);
		}
		return board.sideToMove() == chess::Color::WHITE ? white : -white;
	}

	float Model::score(const chess::Board &board, const FormulaMask &active) const {
		float white;
		if (links.empty()) {
			white = FormulaSet::evaluate(board, std::span<const float>(p.data(), formulas()), active);
		} else {
			const auto relationWeights = std::span<const float>(p.data() + static_cast<std::ptrdiff_t>(formulas()), links.size());
			white = FormulaSet::evaluate(board, std::span<const float>(p.data(), formulas()), rows, conditions, relationWeights, active);
		}
		return board.sideToMove() == chess::Color::WHITE ? white : -white;
	}

	FormulaMask Model::activeFormulas() const noexcept {
		FormulaMask active{};
		const auto mark = [&](std::size_t index) { active[index / 64] |= 1ULL << (index % 64); };
		for (std::size_t i = 0; i < formulas(); ++i) {
			if (p[i] != 0.0F) mark(i);
		}
		for (std::size_t i = 0; i < links.size(); ++i) {
			if (p[formulas() + i] == 0.0F) continue;
			mark(links[i].row);
			mark(links[i].condition);
		}
		return active;
	}

	void Model::backward(const Cache &cache, float delta, std::span<float> grad) const {
		const auto n = formulas();
		if (grad.size() != p.size()) throw std::invalid_argument("gradient shape does not match model");
		for (std::uint16_t i = 0; i < cache.count; ++i) {
			const auto index = cache.active[i];
			grad[index] += delta * cache.score[index];
		}
		for (std::size_t i = 0; i < links.size(); ++i) {
			grad[n + i] += delta * cache.score[links[i].row] * cache.condition[links[i].condition];
		}
	}

	void Model::weights(std::span<const Feature> x, std::vector<float> &out) const {
		std::array<float, kFormulaCount> condition{};
		out.assign(p.begin(), p.begin() + static_cast<std::ptrdiff_t>(formulas()));
		for (const auto &f : x) {
			condition[f.index] = static_cast<float>(f.condition);
		}
		for (std::size_t i = 0; i < links.size(); ++i) {
			out[links[i].row] += p[formulas() + i] * condition[links[i].condition];
		}
	}

	int Model::centipawns(const chess::Board &board) const {
		return toCp(score(board));
	}

	void Model::extract(const chess::Board &board, std::vector<Feature> &out) const {
		FormulaSet::evaluate(board, out);
	}

	bool Model::active(std::uint16_t row, std::uint16_t condition) const {
		return std::find_if(links.begin(), links.end(), [=](const Relation &r) { return r.row == row && r.condition == condition; }) != links.end();
	}

	bool Model::activate(std::uint16_t row, std::uint16_t condition) {
		if (row >= formulas() || condition >= formulas()) throw std::out_of_range("relation coordinate exceeds formula layout");
		if (active(row, condition) || links.size() >= kRelationLimit) return false;
		links.push_back({row, condition});
		rows.push_back(row);
		conditions.push_back(condition);
		p.push_back(0.0F);
		return true;
	}

	void Model::prune(float threshold) {
		if (threshold < 0 || !std::isfinite(threshold)) throw std::invalid_argument("invalid relation pruning threshold");
		const auto n = formulas();
		std::size_t write = 0;
		for (std::size_t read = 0; read < links.size(); ++read) {
			if (std::abs(p[n + read]) <= threshold) continue;
			links[write] = links[read];
			rows[write] = rows[read];
			conditions[write] = conditions[read];
			p[n + write] = p[n + read];
			++write;
		}
		links.resize(write);
		rows.resize(write);
		conditions.resize(write);
		p.resize(n + write);
	}

	void Model::save(const std::filesystem::path &path) const {
		const auto n = formulas();
		if (p.size() != n + links.size() || links.size() > kRelationLimit || !finite(p)) throw std::runtime_error("invalid Eleginus parameters");
		if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
		auto temporary = path;
		temporary += ".tmp";
		try {
			std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
			if (!out) throw std::runtime_error("cannot create Eleginus model: " + temporary.string());
			out.write(magic.data(), magic.size());
			write(out, kArchitectureType);
			write(out, static_cast<std::uint32_t>(n));
			write(out, static_cast<std::uint32_t>(links.size()));
			out.write(reinterpret_cast<const char *>(p.data()), static_cast<std::streamsize>(n * sizeof(float)));
			for (std::size_t i = 0; i < links.size(); ++i) {
				write(out, links[i].row);
				write(out, links[i].condition);
				write(out, p[n + i]);
			}
			out.close();
			if (!out) throw std::runtime_error("cannot write Eleginus model: " + temporary.string());
			#ifdef _WIN32
			if (!MoveFileExW(temporary.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
				throw std::system_error(static_cast<int>(GetLastError()), std::system_category(), "cannot replace Eleginus model");
			}
			#else
			std::filesystem::rename(temporary, path);
			#endif
		} catch (...) {
			std::error_code ignored;
			std::filesystem::remove(temporary, ignored);
			throw;
		}
	}

	Model Model::load(const std::filesystem::path &path) {
		std::ifstream in(path, std::ios::binary);
		if (!in) throw std::runtime_error("cannot open Eleginus model: " + path.string());
		std::array<char, 8> tag{};
		in.read(tag.data(), tag.size());
		const auto architecture = read<std::uint32_t>(in);
		const auto count = read<std::uint32_t>(in);
		const auto relations = read<std::uint32_t>(in);
		Model model;
		if (!in || tag != magic || architecture != kArchitectureType || count != model.formulas() || relations > kRelationLimit) {
			throw std::runtime_error("checkpoint does not match the fixed Eleginus formulas: " + path.string());
		}
		model.p.resize(count);
		in.read(reinterpret_cast<char *>(model.p.data()), static_cast<std::streamsize>(count * sizeof(float)));
		for (std::uint32_t i = 0; i < relations; ++i) {
			const Relation relation{read<std::uint16_t>(in), read<std::uint16_t>(in)};
			const float value = read<float>(in);
			if (relation.row >= count || relation.condition >= count || model.active(relation.row, relation.condition)) {
				throw std::runtime_error("invalid Eleginus relation coordinate: " + path.string());
			}
			model.links.push_back(relation);
			model.rows.push_back(relation.row);
			model.conditions.push_back(relation.condition);
			model.p.push_back(value);
		}
		if (!in || in.peek() != std::char_traits<char>::eof() || !finite(model.p)) {
			throw std::runtime_error("invalid Eleginus model parameters: " + path.string());
		}
		return model;
	}

} // namespace eleginus
