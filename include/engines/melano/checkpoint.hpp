#pragma once

// Stable checkpoint serialization and atomic file replacement for Melano.

#include <cstdint>
#include <filesystem>
#include "melano/model.hpp"

namespace melano {

struct ArchitectureInfo {
	int channels = 128;
	int blocks = 10;
};

/// Saves model parameters plus the minimal architecture descriptor using atomic replacement.
void save_checkpoint_atomic(const std::filesystem::path &path, const Model &model,
							const ArchitectureInfo &arch);

/// Loads a Melano checkpoint, validates its architecture tag, and moves it to device.
Model load_checkpoint(const std::filesystem::path &path, const torch::Device &device,
					  ArchitectureInfo *arch = nullptr);

} // namespace melano
