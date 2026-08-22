// Implements Gadus checkpoint metadata and atomic replacement.

#include "gadus/checkpoint.hpp"
#include <stdexcept>
#include <torch/serialize.h>
#ifdef _WIN32
#include <windows.h>
#endif

namespace gadus {

namespace {

// Store integer metadata as one-element tensors supported by LibTorch archives.
torch::Tensor scalar(std::int64_t value) {
	return torch::tensor(value, torch::TensorOptions().dtype(torch::kInt64));
}

// Read a required integer checkpoint field and normalize it to int64.
std::int64_t read_scalar(torch::serialize::InputArchive &archive, const std::string &key) {
	torch::Tensor value;
	archive.read(key, value, true);
	return value.item<std::int64_t>();
}

// Replace a checkpoint atomically on each platform so readers never observe a partial file.
void replace_file(const std::filesystem::path &temporary, const std::filesystem::path &target) {
#ifdef _WIN32
	if (!MoveFileExW(temporary.wstring().c_str(), target.wstring().c_str(),
					 MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
		throw std::runtime_error("atomic file replacement failed: " + target.string());
	}
#else
	if (::rename(temporary.c_str(), target.c_str()) != 0) {
		throw std::runtime_error("atomic file replacement failed: " + target.string());
	}
#endif
}

} // namespace

// Serialize parameters and the exact Gadus architecture descriptor to a sibling temp file.
void save_checkpoint_atomic(const std::filesystem::path &path, const Model &model,
							const ArchitectureInfo &arch) {
	if (!path.parent_path().empty()) {
		std::filesystem::create_directories(path.parent_path());
	}
	const auto temporary = path.string() + ".tmp";
	torch::serialize::OutputArchive archive;
	torch::serialize::OutputArchive model_archive;
	torch::serialize::OutputArchive arch_archive;
	model->save(model_archive);
	arch_archive.write("type_id", scalar(1), true);
	arch_archive.write("channels", scalar(arch.channels), true);
	arch_archive.write("blocks", scalar(arch.blocks), true);
	arch_archive.write("action_size", scalar(kActionSize), true);
	archive.write("model", model_archive);
	archive.write("arch", arch_archive);
	archive.save_to(temporary);
	replace_file(temporary, path);
}

// Validate the Gadus type/action dimensions before constructing and loading the model.
Model load_checkpoint(const std::filesystem::path &path, const torch::Device &device,
					  ArchitectureInfo *arch) {
	if (!std::filesystem::exists(path)) {
		throw std::runtime_error("model not found: " + path.string());
	}
	torch::serialize::InputArchive archive;
	try {
		archive.load_from(path.string(), device);
	} catch (const c10::Error &error) {
		throw std::runtime_error("cannot read Gadus checkpoint " + path.string() + ": " +
								 error.what_without_backtrace());
	}
	torch::serialize::InputArchive model_archive;
	torch::serialize::InputArchive arch_archive;
	archive.read("model", model_archive);
	archive.read("arch", arch_archive);
	if (read_scalar(arch_archive, "type_id") != 1) {
		throw std::runtime_error("checkpoint is not a Gadus model: " + path.string());
	}
	ArchitectureInfo loaded;
	loaded.channels = static_cast<int>(read_scalar(arch_archive, "channels"));
	loaded.blocks = static_cast<int>(read_scalar(arch_archive, "blocks"));
	if (read_scalar(arch_archive, "action_size") != kActionSize) {
		throw std::runtime_error("checkpoint action size does not match Gadus: " + path.string());
	}
	auto model = Model(loaded.channels, loaded.blocks);
	model->to(device);
	model->load(model_archive);
	if (arch != nullptr) {
		*arch = loaded;
	}
	return model;
}

} // namespace gadus
