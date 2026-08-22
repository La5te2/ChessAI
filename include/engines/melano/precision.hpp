#pragma once

// Melano compute-precision policy and scoped LibTorch CUDA autocast control.

#include <stdexcept>
#include <string>
#include <ATen/autocast_mode.h>
#include <torch/types.h>

namespace melano {

enum class ComputePrecision { Fp32, Bf16 };

/// Parses the exact public precision names used by Melano command-line programs.
inline ComputePrecision parse_compute_precision(const std::string &value) {
	if (value == "fp32") {
		return ComputePrecision::Fp32;
	}
	if (value == "bf16") {
		return ComputePrecision::Bf16;
	}
	throw std::invalid_argument("precision must be fp32 or bf16");
}

/// Returns the stable external spelling of one Melano compute precision.
inline const char *compute_precision_name(ComputePrecision value) noexcept {
	return value == ComputePrecision::Bf16 ? "bf16" : "fp32";
}

/// Rejects reduced precision on devices where Melano has no calibrated path.
inline void validate_compute_precision(ComputePrecision precision, const torch::Device &device) {
	if (precision == ComputePrecision::Bf16 && !device.is_cuda()) {
		throw std::invalid_argument("bf16 precision requires a CUDA device");
	}
}

class AutocastGuard {
	public:
	/// Enables CUDA BF16 autocast for one forward scope and preserves nested caller state.
	AutocastGuard(ComputePrecision precision, const torch::Device &device)
		: active_(precision == ComputePrecision::Bf16) {
		if (!active_) {
			return;
		}
		validate_compute_precision(precision, device);
		previous_enabled_ = at::autocast::is_autocast_enabled(at::kCUDA);
		previous_dtype_ = at::autocast::get_autocast_dtype(at::kCUDA);
		at::autocast::increment_nesting();
		at::autocast::set_autocast_dtype(at::kCUDA, at::kBFloat16);
		at::autocast::set_autocast_enabled(at::kCUDA, true);
	}

	/// Restores the prior thread-local autocast state and clears the outermost cast cache.
	~AutocastGuard() {
		if (!active_) {
			return;
		}
		at::autocast::set_autocast_enabled(at::kCUDA, previous_enabled_);
		at::autocast::set_autocast_dtype(at::kCUDA, previous_dtype_);
		if (at::autocast::decrement_nesting() == 0) {
			at::autocast::clear_cache();
		}
	}

	AutocastGuard(const AutocastGuard &) = delete;
	AutocastGuard &operator=(const AutocastGuard &) = delete;

	private:
	bool active_ = false;
	bool previous_enabled_ = false;
	at::ScalarType previous_dtype_ = at::kFloat;
};

} // namespace melano
