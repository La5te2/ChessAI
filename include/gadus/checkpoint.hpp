#pragma once

// Stable checkpoint serialization, hashing, and atomic file replacement for Gadus.

#include <cstdint>
#include <filesystem>
#include <string>
#include "gadus/model.hpp"

namespace gadus {

struct ArchitectureInfo {
	int channels = 128;
	int blocks = 10;
};

/// Saves model parameters plus the minimal architecture descriptor using atomic replacement.
void save_checkpoint_atomic(const std::filesystem::path &path, const Model &model,
							const ArchitectureInfo &arch);

/// Loads a Gadus checkpoint, validates its architecture tag, and moves it to device.
Model load_checkpoint(const std::filesystem::path &path, const torch::Device &device,
					  ArchitectureInfo *arch = nullptr);

/// Computes the lowercase hexadecimal SHA-256 digest of a file.
std::string file_sha256(const std::filesystem::path &path);
/// Copies source to target through a sibling temporary file and atomic replacement.
void atomic_copy(const std::filesystem::path &source, const std::filesystem::path &target);

} // namespace gadus
