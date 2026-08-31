#pragma once

#include "eleginus/formula.hpp"
#include <array>
#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>
#if defined(__SSE2__) || defined(_M_X64)
#include <xmmintrin.h>
#endif

namespace eleginus {

	inline constexpr std::uint32_t kArchitectureType = 3;

	// The fixed DAG supplies sparse signals; only v, E, U, G and b are learned.
	class Model {
	public:
		static constexpr std::size_t width = 16;
		static constexpr std::size_t layers = 2;
		using Vec = std::array<float, width>;
		struct Cache {
			std::array<Vec, layers + 1> h{};
			Vec t{}; // U^T Phi, accumulated together with E z and v^T Phi.
		};
		struct Layout {
			std::size_t n, z, e, u, g, b, total; // Formula/input counts and flat parameter offsets.
		};

		explicit Model(std::uint64_t seed = 2026);
		static Model load(const std::filesystem::path &path);
		void save(const std::filesystem::path &path) const;
		float score(const chess::Board &board) const;
		float score(std::span<const Feature> x) const;
		float forward(std::span<const Feature> x, Cache &cache) const;
		void backward(std::span<const Feature> x, const Cache &cache, float delta, std::span<float> grad) const;
		void weights(std::span<const Feature> x, std::vector<float> &out) const;
		int centipawns(const chess::Board &board) const;
		void extract(const chess::Board &board, std::vector<Feature> &out) const;
		static const Program &program();
		const Layout &layout() const noexcept { return shape; }
		const std::vector<float> &params() const noexcept { return p; }
		std::vector<float> &params() noexcept { return p; }

	private:
		friend class Evaluator;
		// dst += scale * src. SIMD preserves the accumulation order within each channel.
		static void axpy(float *dst, const float *src, float scale) {
#if defined(__SSE2__) || defined(_M_X64)
			const auto a = _mm_set1_ps(scale);
			for (std::size_t k = 0; k < width; k += 4)
				_mm_storeu_ps(dst + k, _mm_add_ps(_mm_loadu_ps(dst + k), _mm_mul_ps(_mm_loadu_ps(src + k), a)));
#else
			for (std::size_t k = 0; k < width; ++k)
				dst[k] += src[k] * scale;
#endif
		}
		float finish(Cache &cache, float base) const;
		Layout shape;
		std::vector<float> p;
	};

	// Search-local projection cache. The model's parameters must stay fixed for this lifetime.
	class Evaluator {
	public:
		explicit Evaluator(const Model &model) : net(model), families(Model::program().families()) {}
		float score(const chess::Board &board);
		int centipawns(const chess::Board &board);

	private:
		friend struct Projection;
		void begin(const std::array<double, 4> &coords);
		struct Row {
			float v = 0;
			Model::Vec e{}, u{};
			bool ready = false;
		};
		struct Cell {
			double phase = -1, coord = -1;
			std::array<float, 4> blend{};
			std::vector<Row> rows;
		};
		Cell &cell(unsigned family, double phase, double coord);
		void prepare(Row &row, const Cell &entry, std::uint32_t index);
		// Inline the accumulation path; coefficient preparation is needed only on cache misses.
		void accept(std::uint32_t i, float value) {
			const auto &s = net.layout();
			if (i >= s.n) {
				const auto &p = net.params();
				Model::axpy(cache.h[0].data(), p.data() + s.e + i * Model::width, value);
				return;
			}
			auto &entry = *active[families[i / 4]];
			auto &row = entry.rows[i / 4];
			if (!row.ready)
				prepare(row, entry, i);
			base += row.v * value;
			Model::axpy(cache.h[0].data(), row.e.data(), value);
			Model::axpy(cache.t.data(), row.u.data(), value);
		}
		const Model &net;
		std::span<const std::uint8_t> families;
		Model::Cache cache{};
		float base = 0;
		std::array<Cell *, 3> active{};
		std::array<std::array<Cell, 4>, 3> cells;
		std::array<unsigned, 3> next{};
	};

} // namespace eleginus
