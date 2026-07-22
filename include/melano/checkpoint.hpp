#pragma once

#include <cstdint>
#include <filesystem>
#include <string>

#include <torch/torch.h>

#include "melano/model.hpp"

namespace melano {

struct ArchitectureInfo {
	int channels = 128;
	int blocks = 10;
};

void save_checkpoint_atomic(const std::filesystem::path &path, const Model &model,
							const ArchitectureInfo &arch);

Model load_checkpoint(const std::filesystem::path &path, const torch::Device &device,
					  ArchitectureInfo *arch = nullptr);

std::string file_sha256(const std::filesystem::path &path);
void atomic_copy(const std::filesystem::path &source, const std::filesystem::path &target);

} // namespace melano
