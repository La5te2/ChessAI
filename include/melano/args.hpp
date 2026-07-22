#pragma once

// Command-line parsing and run-identifier helpers used by Melano executables.

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace melano {

class Args {
	public:
	/// Parses GNU-style --name value options and standalone --flag switches.
	Args(int argc, char **argv);

	/// Reports whether an option or flag was present on the command line.
	bool has(const std::string &name) const;
	/// Returns an option value, or fallback when the option was omitted.
	std::string get(const std::string &name, const std::string &fallback = "") const;
	/// Returns an option value while preserving the distinction between absent and empty.
	std::optional<std::string> optional(const std::string &name) const;
	/// Parses an option as a signed int, using fallback when absent.
	int get_int(const std::string &name, int fallback) const;
	/// Parses an option as a signed 64-bit int, using fallback when absent.
	std::int64_t get_int64(const std::string &name, std::int64_t fallback) const;
	/// Parses an option as a floating-point value, using fallback when absent.
	double get_double(const std::string &name, double fallback) const;
	/// Parses common textual Boolean spellings, using fallback when absent.
	bool get_bool(const std::string &name, bool fallback) const;

	private:
	std::unordered_map<std::string, std::string> values_;
	std::unordered_set<std::string> flags_;
};

/// Returns a local wall-clock timestamp suitable for user-facing run names.
std::string timestamp();
/// Builds a collision-resistant run id from a prefix, timestamp, and random suffix.
std::string create_run_id(const std::string &prefix);

} // namespace melano
