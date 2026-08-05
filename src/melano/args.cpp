// Implements the small dependency-free argument grammar shared by Melano tools.

#include "melano/args.hpp"
#include <stdexcept>

namespace melano {

namespace {

// Canonicalize callers that use either "name" or "--name".
std::string normalized_name(std::string name) {
	while (name.starts_with('-')) {
		name.erase(name.begin());
	}
	return name;
}

// Accept explicit Boolean values while rejecting misspellings early.
bool parse_bool_text(const std::string &value) {
	if (value == "1" || value == "true" || value == "yes" || value == "on") {
		return true;
	}
	if (value == "0" || value == "false" || value == "no" || value == "off") {
		return false;
	}
	throw std::invalid_argument("invalid boolean value: " + value);
}

} // namespace

// Consume argv once, storing valued options separately from standalone flags.
Args::Args(int argc, char **argv) {
	for (int i = 1; i < argc; ++i) {
		std::string token = argv[i];
		if (!token.starts_with("--")) {
			throw std::invalid_argument("unexpected positional argument: " + token);
		}
		token = normalized_name(token);
		const auto equal = token.find('=');
		if (equal != std::string::npos) {
			values_[token.substr(0, equal)] = token.substr(equal + 1);
			continue;
		}
		if (i + 1 < argc && !std::string(argv[i + 1]).starts_with("--")) {
			values_[token] = argv[++i];
		} else {
			flags_.insert(token);
		}
	}
}

// Test both valued options and standalone flags after name normalization.
bool Args::has(const std::string &name) const {
	const auto key = normalized_name(name);
	return values_.contains(key) || flags_.contains(key);
}

// Read a textual option without imposing a type conversion.
std::string Args::get(const std::string &name, const std::string &fallback) const {
	const auto key = normalized_name(name);
	if (const auto it = values_.find(key); it != values_.end()) {
		return it->second;
	}
	return flags_.contains(key) ? "true" : fallback;
}

// Preserve absence for callers where an empty string is a meaningful value.
std::optional<std::string> Args::optional(const std::string &name) const {
	const auto key = normalized_name(name);
	if (const auto it = values_.find(key); it != values_.end()) {
		return it->second;
	}
	if (flags_.contains(key)) {
		return std::string("true");
	}
	return std::nullopt;
}

// Convert a present option to int with standard-library range/error checking.
int Args::get_int(const std::string &name, int fallback) const {
	const auto value = optional(name);
	return value ? std::stoi(*value) : fallback;
}

// Convert a present option to int64 for step and dataset counts.
std::int64_t Args::get_int64(const std::string &name, std::int64_t fallback) const {
	const auto value = optional(name);
	return value ? std::stoll(*value) : fallback;
}

// Convert a present option to double for probabilities, times, and loss weights.
double Args::get_double(const std::string &name, double fallback) const {
	const auto value = optional(name);
	return value ? std::stod(*value) : fallback;
}

// Treat a standalone flag as true and a valued option through parse_bool_text.
bool Args::get_bool(const std::string &name, bool fallback) const {
	const auto value = optional(name);
	return value ? parse_bool_text(*value) : fallback;
}

} // namespace melano
