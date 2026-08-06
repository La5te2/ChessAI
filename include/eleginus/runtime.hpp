#pragma once

// Torch-free Eleginus Value inference and executable-embedded representation.

#include <cstdint>
#include <filesystem>

#include "eleginus/nnue.hpp"

namespace eleginus {

inline constexpr std::int64_t kEleginusCheckpointType = 3;

/// Copies an executable and appends the Torch-free runtime representation atomically.
void embed_runtime_model_atomic(const std::filesystem::path &input,
								const std::filesystem::path &output,
								const ValueWeights &value);
/// Loads the embedded runtime representation from an executable. An empty path selects this process.
ValueWeights load_embedded_runtime_model(
	const std::filesystem::path &executable = {});

} // namespace eleginus
