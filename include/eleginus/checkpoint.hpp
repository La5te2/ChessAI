#pragma once

// Eleginus independent Policy/Value checkpoint and executable embedding bridge.

#include <cstdint>
#include <filesystem>

#include "eleginus/model.hpp"
#include "eleginus/runtime.hpp"

namespace eleginus {

Model make_model(const torch::Device &device, std::uint64_t seed);
/// Saves the sole persistent Eleginus checkpoint as a LibTorch model/arch archive.
void save_checkpoint_atomic(const std::filesystem::path &path, const Model &model);
Model load_checkpoint(const std::filesystem::path &path, const torch::Device &device);
/// Converts one training checkpoint into a self-contained Torch-free executable.
void embed_checkpoint_atomic(const std::filesystem::path &model,
							 const std::filesystem::path &input,
							 const std::filesystem::path &output);

} // namespace eleginus
