#pragma once

// Stable checkpoint serialization and atomic replacement for Gadus.

#include "gadus/model.hpp"
#include <cstdint>
#include <filesystem>
#include <string>

namespace gadus {

struct ArchitectureInfo {
	int channels = 128;
	int blocks = 12;
};

/// Saves model parameters plus the minimal architecture descriptor using atomic replacement.
void save_checkpoint_atomic(const std::filesystem::path &path, const Model &model, const ArchitectureInfo &arch);

/// Loads a Gadus checkpoint, validates its architecture tag, and moves it to device.
Model load_checkpoint(const std::filesystem::path &path, const torch::Device &device, ArchitectureInfo *arch = nullptr);

} // namespace gadus
