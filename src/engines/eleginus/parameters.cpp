#include "eleginus/parameters.hpp"
#include <algorithm>
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
		// --------------------------- Parameter storage helpers ---------------------------
		// Source defaults, fixed checkpoint fields and stream operations.

		constexpr std::uint32_t architectureType = 3;
		constexpr std::array<char, 8> checkpointMagic{'E', 'L', 'E', 'G', 'I', 'N', 'U', 'S'};
		namespace initial {
			using eleginus::AdjustmentWeights;
			using eleginus::FormulaWeights;
			#include "weights.inl"
		}

		// Write one fixed-width header value.
		template <class T> void write(std::ostream &stream, const T &value) {
			stream.write(reinterpret_cast<const char *>(&value), sizeof(value));
		}

		// Read one fixed-width header value.
		template <class T> T read(std::istream &stream) {
			T value{};
			stream.read(reinterpret_cast<char *>(&value), sizeof(value));
			return value;
		}

		// Report whether every value in one parameter group is finite.
		template <class Range> bool allFinite(const Range &values) {
			return std::ranges::all_of(values, [](float value) { return std::isfinite(value); });
		}

	}

	void validateParameters(const FormulaParameters &parameters) {
		for (const auto &formula : parameters.formulas) {
			if (!std::isfinite(formula.base) || !allFinite(formula.material)) {
				throw std::invalid_argument("Eleginus parameters contain a nonfinite value");
			}
		}
		const auto &adjustments = parameters.adjustments;
		if (!std::isfinite(adjustments.pressureCenter) || !std::isfinite(adjustments.pressureWidth) || !allFinite(adjustments.winnability) ||
			!allFinite(adjustments.scaling)) throw std::invalid_argument("Eleginus parameters contain a nonfinite value");
		if (!(adjustments.pressureWidth > 0.0F)) {
			throw std::invalid_argument("Eleginus pressure width must be positive");
		}
	}

	FormulaParameters initialParameters() {
		return {initial::formulaWeights, initial::formulaGlobals};
	}

	ParameterValues flattenParameters(const FormulaParameters &parameters) {
		validateParameters(parameters);
		ParameterValues values{};
		std::size_t index = 0;
		for (const auto &formula : parameters.formulas) {
			values[index++] = formula.base;
			for (float response : formula.material) values[index++] = response;
		}
		values[index++] = parameters.adjustments.pressureCenter;
		values[index++] = parameters.adjustments.pressureWidth;
		for (float value : parameters.adjustments.winnability) values[index++] = value;
		for (float value : parameters.adjustments.scaling) values[index++] = value;
		return values;
	}

	FormulaParameters expandParameters(const ParameterValues &values) {
		FormulaParameters parameters;
		std::size_t index = 0;
		for (auto &formula : parameters.formulas) {
			formula.base = values[index++];
			for (float &response : formula.material) response = values[index++];
		}
		parameters.adjustments.pressureCenter = values[index++];
		parameters.adjustments.pressureWidth = values[index++];
		for (float &value : parameters.adjustments.winnability) value = values[index++];
		for (float &value : parameters.adjustments.scaling) value = values[index++];
		validateParameters(parameters);
		return parameters;
	}

	// Read and validate the magic, architecture id, formula count and complete float payload.
	FormulaParameters loadParameters(const std::filesystem::path &path) {
		std::ifstream stream(path, std::ios::binary);
		if (!stream) throw std::runtime_error("cannot open Eleginus checkpoint: " + path.string());
		std::array<char, 8> magic{};
		stream.read(magic.data(), magic.size());
		const auto architecture = read<std::uint32_t>(stream);
		const auto formulas = read<std::uint32_t>(stream);
		ParameterValues values{};
		stream.read(reinterpret_cast<char *>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(float)));
		if (!stream || stream.peek() != std::char_traits<char>::eof() || magic != checkpointMagic || architecture != architectureType ||
			formulas != formulaCount) {
			throw std::runtime_error("checkpoint does not match the fixed Eleginus formulas: " + path.string());
		}
		return expandParameters(values);
	}

	// Write the complete parameter file to a temporary path, then atomically replace the destination.
	void saveParameters(const std::filesystem::path &path, const FormulaParameters &parameters) {
		const auto values = flattenParameters(parameters);
		if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
		auto temporary = path;
		temporary += ".tmp";
		try {
			std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
			if (!stream) throw std::runtime_error("cannot create Eleginus checkpoint: " + temporary.string());
			stream.write(checkpointMagic.data(), checkpointMagic.size());
			write(stream, architectureType);
			write(stream, static_cast<std::uint32_t>(formulaCount));
			stream.write(reinterpret_cast<const char *>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(float)));
			stream.close();
			if (!stream) throw std::runtime_error("cannot write Eleginus checkpoint: " + temporary.string());
			#ifdef _WIN32
			if (!MoveFileExW(temporary.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
				throw std::system_error(static_cast<int>(GetLastError()), std::system_category(), "cannot replace Eleginus checkpoint");
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

} // namespace eleginus
