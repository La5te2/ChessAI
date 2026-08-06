// Implements the sole Eleginus Value checkpoint and executable embedding bridge.

#include "eleginus/checkpoint.hpp"

#include <cstdio>
#include <system_error>

#ifdef _WIN32
#include <windows.h>
#endif

namespace eleginus {

namespace {

torch::Tensor scalar(std::int64_t value) {
	return torch::tensor(value, torch::TensorOptions().dtype(torch::kInt64));
}

std::int64_t read_scalar(torch::serialize::InputArchive &archive, const std::string &key) {
	torch::Tensor value;
	archive.read(key, value, true);
	if (value.numel() != 1)
		throw std::runtime_error("Eleginus checkpoint field is not scalar: " + key);
	return value.item<std::int64_t>();
}

void require_scalar(torch::serialize::InputArchive &archive, const std::string &key,
					std::int64_t expected) {
	if (read_scalar(archive, key) != expected)
		throw std::runtime_error("Eleginus checkpoint field does not match this build: " + key);
}

void write_architecture(torch::serialize::OutputArchive &archive) {
	archive.write("type_id", scalar(kEleginusCheckpointType), true);
	archive.write("feature_count", scalar(kFeatureVocabulary), true);
	archive.write("feature_slots", scalar(kFeatureSlots), true);
	archive.write("accumulator", scalar(kValueAccumulatorWidth), true);
	archive.write("hidden", scalar(kValueHiddenWidth), true);
	archive.write("bottleneck", scalar(kValueBottleneckWidth), true);
	archive.write("action_size", scalar(kActionSize), true);
}

void validate_architecture(torch::serialize::InputArchive &archive) {
	require_scalar(archive, "type_id", kEleginusCheckpointType);
	require_scalar(archive, "feature_count", kFeatureVocabulary);
	require_scalar(archive, "feature_slots", kFeatureSlots);
	require_scalar(archive, "accumulator", kValueAccumulatorWidth);
	require_scalar(archive, "hidden", kValueHiddenWidth);
	require_scalar(archive, "bottleneck", kValueBottleneckWidth);
	require_scalar(archive, "action_size", kActionSize);
}

void replace_file(const std::filesystem::path &temporary, const std::filesystem::path &target) {
#ifdef _WIN32
	if (!MoveFileExW(temporary.wstring().c_str(), target.wstring().c_str(),
					 MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
		throw std::runtime_error("cannot replace Eleginus checkpoint: " + target.string());
#else
	if (::rename(temporary.c_str(), target.c_str()) != 0)
		throw std::runtime_error("cannot replace Eleginus checkpoint: " + target.string());
#endif
}

} // namespace

ValueNetwork make_model(const torch::Device &device, std::uint64_t seed) {
	torch::manual_seed(static_cast<std::int64_t>(seed));
	auto model = ValueNetwork();
	model->to(device);
	return model;
}

void save_checkpoint_atomic(const std::filesystem::path &path, const ValueNetwork &model) {
	if (!model)
		throw std::invalid_argument("cannot save an empty Eleginus model");
	if (path.extension() != ".pth")
		throw std::invalid_argument("Eleginus training checkpoint must use the .pth extension");
	if (!path.parent_path().empty())
		std::filesystem::create_directories(path.parent_path());
	const auto temporary = std::filesystem::path(path.string() + ".tmp");
	try {
		torch::serialize::OutputArchive archive;
		torch::serialize::OutputArchive model_archive;
		torch::serialize::OutputArchive arch_archive;
		model->save(model_archive);
		write_architecture(arch_archive);
		archive.write("model", model_archive);
		archive.write("arch", arch_archive);
		archive.save_to(temporary.string());
		replace_file(temporary, path);
	} catch (...) {
		std::error_code ignored;
		std::filesystem::remove(temporary, ignored);
		throw;
	}
}

ValueNetwork load_checkpoint(const std::filesystem::path &path, const torch::Device &device) {
	if (path.extension() != ".pth")
		throw std::invalid_argument("Eleginus training checkpoint must use the .pth extension");
	if (!std::filesystem::is_regular_file(path))
		throw std::runtime_error("Eleginus checkpoint not found: " + path.string());
	torch::serialize::InputArchive archive;
	try {
		archive.load_from(path.string(), device);
	} catch (const c10::Error &error) {
		throw std::runtime_error("cannot read Eleginus checkpoint " + path.string() + ": " +
			error.what_without_backtrace());
	}
	torch::serialize::InputArchive model_archive;
	torch::serialize::InputArchive arch_archive;
	archive.read("model", model_archive);
	archive.read("arch", arch_archive);
	validate_architecture(arch_archive);
	auto model = make_model(device, 0);
	model->load(model_archive);
	return model;
}

void embed_checkpoint_atomic(const std::filesystem::path &model_path,
							 const std::filesystem::path &input,
							 const std::filesystem::path &output) {
	auto model = load_checkpoint(model_path, torch::Device(torch::kCPU));
	embed_runtime_model_atomic(input, output, snapshot_value(model).weights());
}

} // namespace eleginus
