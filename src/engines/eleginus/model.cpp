#include "eleginus/model.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <immintrin.h>
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

		void addScaled(float *target, const float *source, float scale, std::size_t count) noexcept {
			const auto factor = _mm256_set1_ps(scale);
			std::size_t i = 0;
			for (; i + 8 <= count; i += 8) {
				const auto values = _mm256_loadu_ps(source + i);
				const auto current = _mm256_loadu_ps(target + i);
				_mm256_storeu_ps(target + i, _mm256_fmadd_ps(values, factor, current));
			}
			for (; i < count; ++i) target[i] += source[i] * scale;
		}

		float dot(const float *left, const float *right, std::size_t count) noexcept {
			auto sum = _mm256_setzero_ps();
			std::size_t i = 0;
			for (; i + 8 <= count; i += 8) {
				sum = _mm256_fmadd_ps(_mm256_loadu_ps(left + i), _mm256_loadu_ps(right + i), sum);
			}
			const auto high = _mm256_extractf128_ps(sum, 1);
			auto low = _mm_add_ps(_mm256_castps256_ps128(sum), high);
			low = _mm_hadd_ps(low, low);
			low = _mm_hadd_ps(low, low);
			float result = _mm_cvtss_f32(low);
			for (; i < count; ++i) result += left[i] * right[i];
			return result;
		}

	} // namespace

	int centipawns(float h) {
		if (!std::isfinite(h)) throw std::runtime_error("nonfinite Eleginus evaluation");
		return static_cast<int>(std::lround(std::clamp(kCentipawnsPerLogit * h, -25000.0F, 25000.0F)));
	}

	Model::Model() {
		const auto values = detail::initial();
		p.assign(kParameterCount, 0.0F);
		std::copy(values.begin(), values.end(), p.begin());
		indexRelations();
	}

	std::span<const float> Model::initial() noexcept {
		return detail::initial();
	}

	float Model::score(std::span<const Feature> x) const {
		const auto n = formulas();
		for (const auto &f : x) {
			if (f.index >= n) throw std::out_of_range("formula index exceeds model layout");
		}
		double result = 0.0;
		for (const auto &f : x) {
			result += static_cast<double>(f.score) * p[f.index];
		}
		for (const auto &condition : x) {
			if (condition.condition == 0) continue;
			const auto column = n + static_cast<std::size_t>(condition.index) * n;
			for (const auto &row : x) {
				result += static_cast<double>(p[column + row.index]) * row.score * condition.condition;
			}
		}
		return static_cast<float>(result);
	}

	float Model::score(const chess::Board &board) const {
		thread_local std::vector<Feature> x;
		extract(board, x);
		const float white = score(x);
		return board.sideToMove() == chess::Color::WHITE ? white : -white;
	}

	int Model::centipawns(const chess::Board &board) const {
		return eleginus::centipawns(score(board));
	}

	void Model::extract(const chess::Board &board, std::vector<Feature> &out) const {
		FormulaSet::evaluate(board, out);
	}

	std::size_t Model::relationIndex(std::size_t row, std::size_t condition) const {
		if (row >= formulas() || condition >= formulas()) throw std::out_of_range("relation coordinate exceeds formula layout");
		return formulas() + condition * formulas() + row;
	}

	void Model::indexRelations() noexcept {
		const auto matrix = relations();
		for (std::size_t condition = 0; condition < kFormulaCount; ++condition) {
			const auto first = matrix.begin() + static_cast<std::ptrdiff_t>(condition * kFormulaCount);
			activeColumns[condition] = std::any_of(first, first + static_cast<std::ptrdiff_t>(kFormulaCount), [](float value) { return value != 0.0F; });
		}
	}

	void Model::update(std::span<const float> values) {
		if (values.size() != kParameterCount || !finite(values)) throw std::invalid_argument("invalid Eleginus parameter update");
		std::copy(values.begin(), values.end(), p.begin());
		indexRelations();
	}

	Accumulator::Accumulator(const Model &model) : net(model) {
		features.reserve(kFormulaCount);
		changes.reserve(8 * kFormulaCount);
		frames.reserve(128);
		materialized.reserve(128);
	}

	void Accumulator::addColumn(std::size_t condition, float scale) {
		if (!net.columnActive(condition)) return;
		const auto matrix = net.relations();
		const auto offset = condition * kFormulaCount;
		addScaled(dynamic.data(), matrix.data() + offset, scale, kFormulaCount);
	}

	void Accumulator::extract(const chess::Board &board) {
		nextScores.fill(0.0F);
		nextConditions.fill(0.0F);
		net.extract(board, features);
		for (const auto &feature : features) {
			nextScores[feature.index] = static_cast<float>(feature.score);
			nextConditions[feature.index] = static_cast<float>(feature.condition);
		}
	}

	void Accumulator::reset(const chess::Board &board) {
		scores.fill(0.0F);
		conditions.fill(0.0F);
		std::copy(net.base().begin(), net.base().end(), dynamic.begin());
		changes.clear();
		frames.clear();
		materialized.clear();
		extract(board);
		for (std::size_t i = 0; i < kFormulaCount; ++i) {
			scores[i] = nextScores[i];
			conditions[i] = nextConditions[i];
			if (conditions[i] != 0.0F) addColumn(i, conditions[i]);
		}
	}

	void Accumulator::push() {
		frames.push_back(changes.size());
		materialized.push_back(false);
	}

	void Accumulator::refresh(const chess::Board &board) {
		if (frames.empty() || materialized.back()) return;
		extract(board);
		for (std::size_t i = 0; i < kFormulaCount; ++i) {
			if (scores[i] == nextScores[i] && conditions[i] == nextConditions[i]) continue;
			changes.push_back({static_cast<std::uint16_t>(i), scores[i], conditions[i]});
			const float delta = nextConditions[i] - conditions[i];
			if (delta != 0.0F) addColumn(i, delta);
			scores[i] = nextScores[i];
			conditions[i] = nextConditions[i];
		}
		materialized.back() = true;
	}

	void Accumulator::pop() {
		if (frames.empty()) throw std::logic_error("cannot restore the root Eleginus accumulator");
		const auto first = frames.back();
		frames.pop_back();
		if (materialized.back()) {
			for (std::size_t i = changes.size(); i > first; --i) {
				const auto &change = changes[i - 1];
				const float delta = change.condition - conditions[change.index];
				if (delta != 0.0F) addColumn(change.index, delta);
				scores[change.index] = change.score;
				conditions[change.index] = change.condition;
			}
		}
		changes.resize(first);
		materialized.pop_back();
	}

	float Accumulator::score(const chess::Board &board) {
		refresh(board);
		const float value = dot(scores.data(), dynamic.data(), kFormulaCount);
		return board.sideToMove() == chess::Color::WHITE ? value : -value;
	}

	void Model::save(const std::filesystem::path &path) const {
		if (p.size() != kParameterCount || !finite(p)) throw std::runtime_error("invalid Eleginus parameters");
		if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
		auto temporary = path;
		temporary += ".tmp";
		try {
			std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
			if (!out) throw std::runtime_error("cannot create Eleginus model: " + temporary.string());
			out.write(magic.data(), magic.size());
			write(out, kArchitectureType);
			write(out, static_cast<std::uint32_t>(formulas()));
			out.write(reinterpret_cast<const char *>(p.data()), static_cast<std::streamsize>(p.size() * sizeof(float)));
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
		Model model;
		if (!in || tag != magic || architecture != kArchitectureType || count != model.formulas()) {
			throw std::runtime_error("checkpoint does not match the fixed Eleginus formulas: " + path.string());
		}
		in.read(reinterpret_cast<char *>(model.p.data()), static_cast<std::streamsize>(model.p.size() * sizeof(float)));
		if (!in || in.peek() != std::char_traits<char>::eof() || !finite(model.p)) {
			throw std::runtime_error("invalid Eleginus model parameters: " + path.string());
		}
		model.indexRelations();
		return model;
	}

} // namespace eleginus
