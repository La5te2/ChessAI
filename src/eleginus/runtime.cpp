// Implements the Torch-free Policy/Value representation embedded in an executable.

#include "eleginus/runtime.hpp"

#include <array>
#include <bit>
#include <cstdio>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>
#include <type_traits>
#include <utility>

#ifdef _WIN32
#include <windows.h>
#endif

namespace eleginus {

namespace {

constexpr std::array<char, 8> kMagic{'E', 'L', 'E', 'G', 'I', 'N', 'U', 'S'};
constexpr std::array<char, 8> kFooterMagic{'E', 'L', 'E', 'G', 'E', 'M', 'B', 'D'};
constexpr std::uint32_t kEndianMarker = 0x01020304U;

static_assert(std::endian::native == std::endian::little,
	"Eleginus native weights currently require a little-endian target");
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
	"Eleginus native weights require IEEE-754 float32");

template <typename T>
void write_scalar(std::ofstream &output, T value) {
	static_assert(std::is_trivially_copyable_v<T>);
	output.write(reinterpret_cast<const char *>(&value), sizeof(value));
	if (!output) {
		throw std::runtime_error("cannot write embedded Eleginus model");
	}
}

template <typename T>
T read_scalar(std::ifstream &input, const std::filesystem::path &path) {
	static_assert(std::is_trivially_copyable_v<T>);
	T value{};
	input.read(reinterpret_cast<char *>(&value), sizeof(value));
	if (!input) {
		throw std::runtime_error("truncated embedded Eleginus model: " + path.string());
	}
	return value;
}

void write_vector(std::ofstream &output, const std::vector<float> &values,
				  std::size_t expected, const char *name) {
	if (values.size() != expected) {
		throw std::invalid_argument(std::string("invalid Eleginus ") + name + " size");
	}
	write_scalar(output, static_cast<std::uint64_t>(values.size()));
	output.write(reinterpret_cast<const char *>(values.data()),
		static_cast<std::streamsize>(values.size() * sizeof(float)));
	if (!output) {
		throw std::runtime_error("cannot write embedded Eleginus model");
	}
}

std::vector<float> read_vector(std::ifstream &input, const std::filesystem::path &path,
						   std::size_t expected, const char *name) {
	const auto count = read_scalar<std::uint64_t>(input, path);
	if (count != expected) {
		throw std::runtime_error(std::string("invalid Eleginus ") + name +
			" count in " + path.string());
	}
	std::vector<float> values(expected);
	input.read(reinterpret_cast<char *>(values.data()),
		static_cast<std::streamsize>(values.size() * sizeof(float)));
	if (!input) {
		throw std::runtime_error("truncated embedded Eleginus model: " + path.string());
	}
	return values;
}

void write_policy(std::ofstream &output, const PolicyWeights &weights) {
	write_vector(output, weights.feature_table,
		static_cast<std::size_t>(kFeatureVocabulary) * kPolicyAccumulatorWidth,
		"Policy feature table");
	write_vector(output, weights.accumulator_bias, kPolicyAccumulatorWidth,
		"Policy accumulator bias");
	write_vector(output, weights.hidden_weight,
		static_cast<std::size_t>(kPolicyHiddenWidth) * kPolicyAccumulatorWidth * 2,
		"Policy hidden weight");
	write_vector(output, weights.hidden_bias, kPolicyHiddenWidth, "Policy hidden bias");
	write_vector(output, weights.output_weight,
		static_cast<std::size_t>(kActionSize) * kPolicyHiddenWidth,
		"Policy output weight");
	write_vector(output, weights.output_bias, kActionSize, "Policy output bias");
}

PolicyWeights read_policy(std::ifstream &input, const std::filesystem::path &path) {
	PolicyWeights weights;
	weights.feature_table = read_vector(input, path,
		static_cast<std::size_t>(kFeatureVocabulary) * kPolicyAccumulatorWidth,
		"Policy feature table");
	weights.accumulator_bias = read_vector(input, path, kPolicyAccumulatorWidth,
		"Policy accumulator bias");
	weights.hidden_weight = read_vector(input, path,
		static_cast<std::size_t>(kPolicyHiddenWidth) * kPolicyAccumulatorWidth * 2,
		"Policy hidden weight");
	weights.hidden_bias = read_vector(input, path, kPolicyHiddenWidth, "Policy hidden bias");
	weights.output_weight = read_vector(input, path,
		static_cast<std::size_t>(kActionSize) * kPolicyHiddenWidth,
		"Policy output weight");
	weights.output_bias = read_vector(input, path, kActionSize, "Policy output bias");
	return weights;
}

void write_value(std::ofstream &output, const ValueWeights &weights) {
	write_vector(output, weights.feature_table,
		static_cast<std::size_t>(kFeatureVocabulary) * kValueFeatureWidth,
		"Value feature table");
	write_vector(output, weights.accumulator_bias, kValueFeatureWidth,
		"Value accumulator bias");
	write_vector(output, weights.hidden_weight,
		static_cast<std::size_t>(kValueBucketCount) * kValueHiddenWidth *
			kValueAccumulatorWidth * 2,
		"Value hidden weight");
	write_vector(output, weights.hidden_bias, kValueBucketCount * kValueHiddenWidth,
		"Value hidden bias");
	write_vector(output, weights.bottleneck_weight,
		static_cast<std::size_t>(kValueBucketCount) * kValueBottleneckWidth *
			kValueHiddenWidth,
		"Value bottleneck weight");
	write_vector(output, weights.bottleneck_bias, kValueBucketCount * kValueBottleneckWidth,
		"Value bottleneck bias");
	write_vector(output, weights.output_weight,
		kValueBucketCount * kValueBottleneckWidth,
		"Value output weight");
	write_vector(output, weights.output_bias, kValueBucketCount, "Value output bias");
}

ValueWeights read_value(std::ifstream &input, const std::filesystem::path &path) {
	ValueWeights weights;
	weights.feature_table = read_vector(input, path,
		static_cast<std::size_t>(kFeatureVocabulary) * kValueFeatureWidth,
		"Value feature table");
	weights.accumulator_bias = read_vector(input, path, kValueFeatureWidth,
		"Value accumulator bias");
	weights.hidden_weight = read_vector(input, path,
		static_cast<std::size_t>(kValueBucketCount) * kValueHiddenWidth *
			kValueAccumulatorWidth * 2,
		"Value hidden weight");
	weights.hidden_bias = read_vector(input, path, kValueBucketCount * kValueHiddenWidth,
		"Value hidden bias");
	weights.bottleneck_weight = read_vector(input, path,
		static_cast<std::size_t>(kValueBucketCount) * kValueBottleneckWidth *
			kValueHiddenWidth,
		"Value bottleneck weight");
	weights.bottleneck_bias = read_vector(input, path,
		kValueBucketCount * kValueBottleneckWidth,
		"Value bottleneck bias");
	weights.output_weight = read_vector(input, path,
		kValueBucketCount * kValueBottleneckWidth,
		"Value output weight");
	weights.output_bias = read_vector(input, path, kValueBucketCount, "Value output bias");
	return weights;
}

void replace_file(const std::filesystem::path &temporary, const std::filesystem::path &target) {
#ifdef _WIN32
	if (!MoveFileExW(temporary.wstring().c_str(), target.wstring().c_str(),
					 MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
		throw std::runtime_error("cannot replace embedded Eleginus executable: " + target.string());
	}
#else
	if (::rename(temporary.c_str(), target.c_str()) != 0) {
		throw std::runtime_error("cannot replace embedded Eleginus executable: " + target.string());
	}
#endif
}

std::filesystem::path current_executable_path() {
#ifdef _WIN32
	std::wstring buffer(32768, L'\0');
	const DWORD length = GetModuleFileNameW(nullptr, buffer.data(),
		static_cast<DWORD>(buffer.size()));
	if (length == 0 || length >= buffer.size())
		throw std::runtime_error("cannot locate the Eleginus executable");
	buffer.resize(length);
	return std::filesystem::path(buffer);
#elif defined(__linux__)
	std::error_code error;
	auto path = std::filesystem::read_symlink("/proc/self/exe", error);
	if (error)
		throw std::runtime_error("cannot locate the Eleginus executable: " + error.message());
	return path;
#else
	throw std::runtime_error("automatic Eleginus executable discovery is unsupported");
#endif
}

} // namespace

void embed_runtime_model_atomic(const std::filesystem::path &input,
								const std::filesystem::path &output_path,
								const RuntimeWeights &weights) {
	if (!std::filesystem::is_regular_file(input))
		throw std::runtime_error("Eleginus executable template not found: " + input.string());
	const auto source = std::filesystem::weakly_canonical(input);
	const auto target = std::filesystem::absolute(output_path).lexically_normal();
	if (source == target)
		throw std::invalid_argument("embedded Eleginus output must differ from its template");
	if (!target.parent_path().empty())
		std::filesystem::create_directories(target.parent_path());
	const std::filesystem::path temporary = target.string() + ".tmp";
	try {
		std::ifstream executable(source, std::ios::binary);
		if (!executable)
			throw std::runtime_error("cannot read Eleginus executable template: " + source.string());
		std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
		if (!output)
			throw std::runtime_error("cannot create embedded Eleginus executable: " + target.string());
		output << executable.rdbuf();
		if (!output)
			throw std::runtime_error("cannot copy Eleginus executable template");
		const auto payload_begin = output.tellp();
		output.write(kMagic.data(), static_cast<std::streamsize>(kMagic.size()));
		write_scalar(output, kEndianMarker);
		write_scalar(output, static_cast<std::uint32_t>(kEleginusCheckpointType));
		write_scalar(output, static_cast<std::uint32_t>(kActionSize));
		write_scalar(output, static_cast<std::uint32_t>(kFeatureVocabulary));
		write_scalar(output, static_cast<std::uint32_t>(kPolicyAccumulatorWidth));
		write_scalar(output, static_cast<std::uint32_t>(kPolicyHiddenWidth));
		write_scalar(output, static_cast<std::uint32_t>(kValueAccumulatorWidth));
		write_scalar(output, static_cast<std::uint32_t>(kValueHiddenWidth));
		write_scalar(output, static_cast<std::uint32_t>(kValueBottleneckWidth));
		write_scalar(output, static_cast<std::uint32_t>(kValueBucketCount));
		write_policy(output, weights.policy);
		write_value(output, weights.value);
		const auto payload_end = output.tellp();
		if (payload_begin < 0 || payload_end < payload_begin)
			throw std::runtime_error("cannot measure embedded Eleginus weights");
		const auto payload_size = static_cast<std::uint64_t>(payload_end - payload_begin);
		output.write(kFooterMagic.data(), static_cast<std::streamsize>(kFooterMagic.size()));
		write_scalar(output, payload_size);
		output.flush();
		if (!output)
			throw std::runtime_error("cannot flush embedded Eleginus executable: " + target.string());
		output.close();
		std::error_code permission_error;
		std::filesystem::permissions(temporary, std::filesystem::status(source).permissions(),
			std::filesystem::perm_options::replace, permission_error);
		if (permission_error)
			throw std::runtime_error("cannot preserve Eleginus executable permissions: " +
				permission_error.message());
		replace_file(temporary, target);
	} catch (...) {
		std::error_code ignored;
		std::filesystem::remove(temporary, ignored);
		throw;
	}
}

RuntimeWeights load_embedded_runtime_model(const std::filesystem::path &executable) {
	const auto path = executable.empty() ? current_executable_path() : executable;
	std::ifstream input(path, std::ios::binary);
	if (!input)
		throw std::runtime_error("embedded Eleginus executable not found: " + path.string());
	input.seekg(0, std::ios::end);
	const auto file_end = input.tellg();
	constexpr auto footer_size = static_cast<std::streamoff>(kFooterMagic.size() +
		sizeof(std::uint64_t));
	if (file_end < footer_size)
		throw std::runtime_error("executable contains no embedded Eleginus model: " + path.string());
	input.seekg(file_end - footer_size);
	std::array<char, kFooterMagic.size()> footer_magic{};
	input.read(footer_magic.data(), static_cast<std::streamsize>(footer_magic.size()));
	const auto payload_size = read_scalar<std::uint64_t>(input, path);
	if (!input || footer_magic != kFooterMagic ||
		payload_size > static_cast<std::uint64_t>(file_end - footer_size))
		throw std::runtime_error("executable contains no embedded Eleginus model: " + path.string());
	const auto payload_begin = file_end - footer_size - static_cast<std::streamoff>(payload_size);
	input.seekg(payload_begin);
	std::array<char, kMagic.size()> magic{};
	input.read(magic.data(), static_cast<std::streamsize>(magic.size()));
	if (!input || magic != kMagic) {
		throw std::runtime_error("embedded data is not an Eleginus model: " + path.string());
	}
	const auto endian = read_scalar<std::uint32_t>(input, path);
	const auto architecture = read_scalar<std::uint32_t>(input, path);
	const auto action_size = read_scalar<std::uint32_t>(input, path);
	const auto feature_count = read_scalar<std::uint32_t>(input, path);
	const auto policy_accumulator = read_scalar<std::uint32_t>(input, path);
	const auto policy_hidden = read_scalar<std::uint32_t>(input, path);
	const auto value_accumulator = read_scalar<std::uint32_t>(input, path);
	const auto value_hidden = read_scalar<std::uint32_t>(input, path);
	const auto value_bottleneck = read_scalar<std::uint32_t>(input, path);
	const auto value_buckets = read_scalar<std::uint32_t>(input, path);
	if (endian != kEndianMarker || architecture != kEleginusCheckpointType ||
		action_size != kActionSize || feature_count != kFeatureVocabulary ||
		policy_accumulator != kPolicyAccumulatorWidth || policy_hidden != kPolicyHiddenWidth ||
		value_accumulator != kValueAccumulatorWidth || value_hidden != kValueHiddenWidth ||
		value_bottleneck != kValueBottleneckWidth || value_buckets != kValueBucketCount) {
		throw std::runtime_error("embedded Eleginus model does not match this build: " +
			path.string());
	}
	auto policy = read_policy(input, path);
	auto value = read_value(input, path);
	if (input.tellg() != file_end - footer_size)
		throw std::runtime_error("embedded Eleginus model length does not match its executable");
	return RuntimeWeights{std::move(policy), std::move(value)};
}

} // namespace eleginus
