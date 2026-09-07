#include "eleginus/evaluate.hpp"
#include <array>
#include <cmath>
#include <fstream>
#include <ranges>
#include <stdexcept>
#include <system_error>
#ifdef _WIN32
	#define NOMINMAX
	#define WIN32_LEAN_AND_MEAN
	#include <windows.h>
#endif

namespace eleginus {
	namespace {
		// ---------------------------- Parameter file layout ------------------------------
		// Fixed header fields and formula-major payload conversion.

		constexpr std::uint32_t architectureType = 3;
		constexpr std::array<char, 8> checkpointMagic{'E', 'L', 'E', 'G', 'I', 'N', 'U', 'S'};

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

		// Flatten structured parameters in formula-major order.
		ParameterValues flatten(const FormulaParameters &parameters) {
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

		// Expand formula-major values into named parameter groups.
		FormulaParameters expand(const ParameterValues &values) {
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
			return parameters;
		}

	}

	void validateParameters(const FormulaParameters &parameters) {
		const auto values = flatten(parameters);
		if (!std::ranges::all_of(values, [](float value) { return std::isfinite(value); })) {
			throw std::invalid_argument("Eleginus parameters contain a nonfinite value");
		}
		if (!(parameters.adjustments.pressureWidth > 0.0F)) {
			throw std::invalid_argument("Eleginus pressure width must be positive");
		}
	}

	ParameterValues flattenParameters(const FormulaParameters &parameters) {
		validateParameters(parameters);
		return flatten(parameters);
	}

	FormulaParameters expandParameters(const ParameterValues &values) {
		auto parameters = expand(values);
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
