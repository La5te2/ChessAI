#include "gadus/checkpoint.hpp"

#include <array>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

namespace gadus {

namespace {

class Sha256 {
	public:
	void update(const std::uint8_t *data, std::size_t size) {
		for (std::size_t index = 0; index < size; ++index) {
			buffer_[buffer_size_++] = data[index];
			if (buffer_size_ == 64) {
				transform();
				bit_count_ += 512;
				buffer_size_ = 0;
			}
		}
	}

	std::array<std::uint8_t, 32> finish() {
		bit_count_ += static_cast<std::uint64_t>(buffer_size_) * 8;
		buffer_[buffer_size_++] = 0x80;
		if (buffer_size_ > 56) {
			while (buffer_size_ < 64) {
				buffer_[buffer_size_++] = 0;
			}
			transform();
			buffer_size_ = 0;
		}
		while (buffer_size_ < 56) {
			buffer_[buffer_size_++] = 0;
		}
		for (int shift = 56; shift >= 0; shift -= 8) {
			buffer_[buffer_size_++] = static_cast<std::uint8_t>(bit_count_ >> shift);
		}
		transform();

		std::array<std::uint8_t, 32> digest{};
		for (int index = 0; index < 8; ++index) {
			for (int shift = 24; shift >= 0; shift -= 8) {
				digest[index * 4 + (24 - shift) / 8] =
					static_cast<std::uint8_t>(state_[index] >> shift);
			}
		}
		return digest;
	}

	private:
	static constexpr std::array<std::uint32_t, 64> constants_{{
		0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
		0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
		0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
		0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
		0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
		0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
		0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
		0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
		0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
		0xc67178f2,
	}};

	static std::uint32_t rotate(std::uint32_t value, int shift) {
		return (value >> shift) | (value << (32 - shift));
	}

	void transform() {
		std::array<std::uint32_t, 64> words{};
		for (int index = 0; index < 16; ++index) {
			words[index] = (static_cast<std::uint32_t>(buffer_[index * 4]) << 24) |
						   (static_cast<std::uint32_t>(buffer_[index * 4 + 1]) << 16) |
						   (static_cast<std::uint32_t>(buffer_[index * 4 + 2]) << 8) |
						   static_cast<std::uint32_t>(buffer_[index * 4 + 3]);
		}
		for (int index = 16; index < 64; ++index) {
			const auto s0 = rotate(words[index - 15], 7) ^ rotate(words[index - 15], 18) ^
							(words[index - 15] >> 3);
			const auto s1 = rotate(words[index - 2], 17) ^ rotate(words[index - 2], 19) ^
							(words[index - 2] >> 10);
			words[index] = words[index - 16] + s0 + words[index - 7] + s1;
		}

		auto a = state_[0];
		auto b = state_[1];
		auto c = state_[2];
		auto d = state_[3];
		auto e = state_[4];
		auto f = state_[5];
		auto g = state_[6];
		auto h = state_[7];
		for (int index = 0; index < 64; ++index) {
			const auto s1 = rotate(e, 6) ^ rotate(e, 11) ^ rotate(e, 25);
			const auto choice = (e & f) ^ (~e & g);
			const auto temp1 = h + s1 + choice + constants_[index] + words[index];
			const auto s0 = rotate(a, 2) ^ rotate(a, 13) ^ rotate(a, 22);
			const auto majority = (a & b) ^ (a & c) ^ (b & c);
			const auto temp2 = s0 + majority;
			h = g;
			g = f;
			f = e;
			e = d + temp1;
			d = c;
			c = b;
			b = a;
			a = temp1 + temp2;
		}
		state_[0] += a;
		state_[1] += b;
		state_[2] += c;
		state_[3] += d;
		state_[4] += e;
		state_[5] += f;
		state_[6] += g;
		state_[7] += h;
	}

	std::array<std::uint8_t, 64> buffer_{};
	std::size_t buffer_size_ = 0;
	std::uint64_t bit_count_ = 0;
	std::array<std::uint32_t, 8> state_{{
		0x6a09e667,
		0xbb67ae85,
		0x3c6ef372,
		0xa54ff53a,
		0x510e527f,
		0x9b05688c,
		0x1f83d9ab,
		0x5be0cd19,
	}};
};

torch::Tensor scalar(std::int64_t value) {
	return torch::tensor(value, torch::TensorOptions().dtype(torch::kInt64));
}

std::int64_t read_scalar(torch::serialize::InputArchive &archive, const std::string &key) {
	torch::Tensor value;
	archive.read(key, value);
	return value.item<std::int64_t>();
}

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

void save_checkpoint_atomic(const std::filesystem::path &path, const Model &model,
							const CheckpointInfo &info) {
	if (!path.parent_path().empty()) {
		std::filesystem::create_directories(path.parent_path());
	}
	const auto temporary = path.string() + ".tmp";
	torch::serialize::OutputArchive archive;
	model->save(archive);
	archive.write("meta_format", scalar(1));
	archive.write("meta_arch", scalar(1));
	archive.write("meta_channels", scalar(info.channels));
	archive.write("meta_blocks", scalar(info.blocks));
	archive.write("meta_epoch", scalar(info.epoch));
	archive.write("meta_global_step", scalar(info.global_step));
	archive.save_to(temporary);
	replace_file(temporary, path);
}

Model load_checkpoint(const std::filesystem::path &path, const torch::Device &device,
					  CheckpointInfo *info) {
	if (!std::filesystem::exists(path)) {
		throw std::runtime_error("model not found: " + path.string());
	}
	torch::serialize::InputArchive archive;
	archive.load_from(path.string(), device);
	if (read_scalar(archive, "meta_arch") != 1) {
		throw std::runtime_error("checkpoint is not a Gadus model: " + path.string());
	}
	CheckpointInfo loaded;
	loaded.channels = static_cast<int>(read_scalar(archive, "meta_channels"));
	loaded.blocks = static_cast<int>(read_scalar(archive, "meta_blocks"));
	loaded.epoch = static_cast<int>(read_scalar(archive, "meta_epoch"));
	loaded.global_step = read_scalar(archive, "meta_global_step");
	auto model = Model(loaded.channels, loaded.blocks);
	model->to(device);
	model->load(archive);
	if (info != nullptr) {
		*info = loaded;
	}
	return model;
}

std::string file_sha256(const std::filesystem::path &path) {
	std::ifstream input(path, std::ios::binary);
	if (!input) {
		throw std::runtime_error("cannot hash missing file: " + path.string());
	}
	Sha256 hash;
	std::array<std::uint8_t, 1 << 16> buffer{};
	while (input) {
		input.read(reinterpret_cast<char *>(buffer.data()), buffer.size());
		hash.update(buffer.data(), static_cast<std::size_t>(input.gcount()));
	}
	const auto digest = hash.finish();
	std::ostringstream output;
	output << std::hex << std::setfill('0');
	for (const auto byte : digest) {
		output << std::setw(2) << static_cast<int>(byte);
	}
	return output.str();
}

void atomic_copy(const std::filesystem::path &source, const std::filesystem::path &target) {
	if (!target.parent_path().empty()) {
		std::filesystem::create_directories(target.parent_path());
	}
	const auto temporary = target.string() + ".tmp";
	std::filesystem::copy_file(source, temporary,
							   std::filesystem::copy_options::overwrite_existing);
	replace_file(temporary, target);
}

} // namespace gadus
