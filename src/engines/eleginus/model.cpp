#include "eleginus/model.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <random>
#include <stdexcept>
#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

namespace eleginus {
	namespace {

		constexpr std::array<char, 8> magic{'E', 'L', 'E', 'G', 'I', 'N', 'U', 'S'};
		constexpr float inputScale = kInputScale; // E uses scaled scored signals; v/U retain the original Phi.

		template <typename T> void write(std::ostream &out, const T &x) {
			out.write(reinterpret_cast<const char *>(&x), sizeof(x));
		}

		template <typename T> T read(std::istream &in) {
			T x{};
			in.read(reinterpret_cast<char *>(&x), sizeof(x));
			return x;
		}

		bool finite(std::span<const float> x) {
			return std::all_of(x.begin(), x.end(), [](float v) { return std::isfinite(v); });
		}

		int toCp(float h) {
			if (!std::isfinite(h))
				throw std::runtime_error("nonfinite Eleginus evaluation");
			return static_cast<int>(std::lround(std::clamp(400.0F * h, -25000.0F, 25000.0F)));
		}

	} // namespace

	const Program &Model::program() {
		return Program::fixed();
	}

	Model::Model(std::uint64_t seed) {
		const auto &roots = program().weights();
		const auto n = roots.size() - kContext;
		const auto z = roots.size();
		const auto e = n;
		const auto u = e + z * width;
		const auto g = u + n * width;
		const auto b = g + layers * width * width;
		shape = {n, z, e, u, g, b, b + layers * width};
		p.assign(shape.total, 0.0F);
		for (std::size_t i = 0; i < n; ++i)
			p[i] = roots[i];
		std::mt19937_64 rng(seed);
		std::uniform_real_distribution<float> unit(-1.0F, 1.0F);
		for (std::size_t i = e; i < u; ++i)
			p[i] = unit(rng) * 0.05F;
		const float scale = std::sqrt(6.0F / static_cast<float>(width));
		for (std::size_t i = g; i < b; ++i)
			p[i] = unit(rng) * scale;
		// U=0 preserves the initial HCE score; E/G are nonzero to permit subsequent learning.
	}

	float Model::forward(std::span<const Feature> x, Cache &cache) const {
		cache = {};
		float result = 0.0F;
		const auto &s = shape;
		for (const auto &f : x) {
			const auto i = f.index;
			const float z = f.value * (i < s.n ? inputScale : 1.0F);
			axpy(cache.h[0].data(), p.data() + s.e + i * width, z);
			if (i >= s.n)
				continue;
			result += p[i] * f.value;
			axpy(cache.t.data(), p.data() + s.u + i * width, f.value);
		}
		return finish(cache, result);
	}

	float Model::finish(Cache &cache, float result) const {
		const auto &s = shape;
		for (std::size_t l = 0; l < layers; ++l) {
			for (std::size_t j = 0; j < width; ++j) {
				float a = p[s.b + l * width + j];
				const auto row = s.g + (l * width + j) * width;
				for (std::size_t k = 0; k < width; ++k)
					a += p[row + k] * cache.h[l][k];
				cache.h[l + 1][j] = std::max(a, 0.0F);
			}
		}
		for (std::size_t k = 0; k < width; ++k)
			result += cache.t[k] * cache.h[layers][k];
		return result;
	}

	void Model::backward(std::span<const Feature> x, const Cache &cache, float delta, std::span<float> grad) const {
		if (grad.size() != p.size())
			throw std::invalid_argument("gradient size does not match model");
		const auto &s = shape;
		Vec dh{};
		for (std::size_t k = 0; k < width; ++k)
			dh[k] = delta * cache.t[k];
		for (const auto &f : x) {
			if (f.index >= s.n)
				continue;
			const float d = delta * f.value;
			grad[f.index] += d;
			axpy(grad.data() + s.u + f.index * width, cache.h[layers].data(), d);
		}
		for (std::size_t l = layers; l-- > 0;) {
			Vec next{};
			for (std::size_t j = 0; j < width; ++j) {
				const float d = cache.h[l + 1][j] > 0.0F ? dh[j] : 0.0F;
				grad[s.b + l * width + j] += d;
				const auto row = s.g + (l * width + j) * width;
				axpy(grad.data() + row, cache.h[l].data(), d);
				axpy(next.data(), p.data() + row, d);
			}
			dh = next;
		}
		for (const auto &f : x) {
			const float z = f.value * (f.index < s.n ? inputScale : 1.0F);
			axpy(grad.data() + s.e + f.index * width, dh.data(), z);
		}
	}

	void Model::weights(std::span<const Feature> x, std::vector<float> &out) const {
		Cache cache;
		forward(x, cache);
		out.assign(p.begin(), p.begin() + shape.n);
		for (std::size_t i = 0; i < out.size(); ++i) {
			for (std::size_t k = 0; k < width; ++k)
				out[i] += p[shape.u + i * width + k] * cache.h[layers][k];
		}
	}

	float Model::score(std::span<const Feature> x) const {
		Cache cache;
		return forward(x, cache);
	}

	float Model::score(const chess::Board &board) const {
		thread_local std::vector<Feature> x;
		extract(board, x);
		return score(x);
	}

	int Model::centipawns(const chess::Board &board) const {
		return toCp(score(board));
	}

	Evaluator::Cell &Evaluator::cell(unsigned family, double phase, double coord) {
		auto &set = cells[family];
		for (auto &entry : set)
			if (entry.phase == phase && entry.coord == coord)
				return entry;
		auto &entry = set[next[family]++ % set.size()];
		entry.phase = phase;
		entry.coord = coord;
		entry.blend = {
		    static_cast<float>((1 - phase) * (1 - coord)), static_cast<float>((1 - phase) * coord), static_cast<float>(phase * (1 - coord)), static_cast<float>(phase * coord)};
		entry.rows.resize(net.layout().n / 4);
		for (auto &row : entry.rows)
			row.ready = false;
		return entry;
	}

	void Evaluator::begin(const std::array<double, 4> &coords) {
		for (unsigned f = 0; f < active.size(); ++f)
			active[f] = &cell(f, coords[0], coords[f + 1]);
		cache = {};
		base = 0;
	}

	void Evaluator::prepare(Row &row, const Cell &entry, std::uint32_t i) {
		const auto &p = net.params();
		const auto &s = net.layout();
		row = {};
		// Distribute interpolation over v/E/U before the nonlinear graybox layers.
		for (unsigned c = 0; c < 4; ++c) {
			const float a = entry.blend[c];
			if (a == 0)
				continue;
			row.v += p[i + c] * a;
			for (std::size_t k = 0; k < Model::width; ++k) {
				row.e[k] += p[s.e + (i + c) * Model::width + k] * a * inputScale;
				row.u[k] += p[s.u + (i + c) * Model::width + k] * a;
			}
		}
		row.ready = true;
	}

	float Evaluator::score(const chess::Board &board) {
		Program::evaluate(board, *this);
		return net.finish(cache, base);
	}

	int Evaluator::centipawns(const chess::Board &board) {
		return toCp(score(board));
	}

	void Model::extract(const chess::Board &board, std::vector<Feature> &out) const {
		program().evaluate(board, out);
	}

	void Model::save(const std::filesystem::path &path) const {
		if (p.size() != shape.total || !finite(p))
			throw std::runtime_error("invalid Eleginus parameters");
		if (!path.parent_path().empty())
			std::filesystem::create_directories(path.parent_path());
		auto temporary = path;
		temporary += ".tmp";
		try {
			std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
			if (!out)
				throw std::runtime_error("cannot create Eleginus model: " + temporary.string());
			out.write(magic.data(), magic.size());
			for (const auto x : {static_cast<std::size_t>(kArchitectureType), shape.n, shape.z, width, layers, shape.total}) {
				write(out, static_cast<std::uint32_t>(x));
			}
			write(out, program().signature());
			out.write(reinterpret_cast<const char *>(p.data()), static_cast<std::streamsize>(p.size() * sizeof(float)));
			out.close();
			if (!out)
				throw std::runtime_error("cannot write Eleginus model: " + temporary.string());
#ifdef _WIN32
			if (!MoveFileExW(temporary.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
				throw std::system_error(static_cast<int>(GetLastError()), std::system_category(), "cannot replace Eleginus model");
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
		if (!in)
			throw std::runtime_error("cannot open Eleginus model: " + path.string());
		std::array<char, 8> tag{};
		in.read(tag.data(), tag.size());
		std::array<std::uint32_t, 6> dims{};
		for (auto &x : dims)
			x = read<std::uint32_t>(in);
		const auto sig = read<std::uint64_t>(in);
		Model model;
		const auto &s = model.shape;
		const std::array<std::uint32_t, 6> expected{kArchitectureType, static_cast<std::uint32_t>(s.n), static_cast<std::uint32_t>(s.z), static_cast<std::uint32_t>(width),
		    static_cast<std::uint32_t>(layers), static_cast<std::uint32_t>(s.total)};
		if (!in || tag != magic || dims != expected || sig != program().signature() || std::filesystem::file_size(path) != 40 + 4 * s.total) {
			throw std::runtime_error("checkpoint does not match the fixed Eleginus formulas and graybox shape: " + path.string());
		}
		in.read(reinterpret_cast<char *>(model.p.data()), static_cast<std::streamsize>(s.total * sizeof(float)));
		if (!in || !finite(model.p))
			throw std::runtime_error("invalid Eleginus model parameters: " + path.string());
		return model;
	}

} // namespace eleginus
