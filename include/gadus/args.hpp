#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace gadus {

class Args {
	public:
	Args(int argc, char **argv);

	bool has(const std::string &name) const;
	std::string get(const std::string &name, const std::string &fallback = "") const;
	std::optional<std::string> optional(const std::string &name) const;
	int get_int(const std::string &name, int fallback) const;
	std::int64_t get_int64(const std::string &name, std::int64_t fallback) const;
	double get_double(const std::string &name, double fallback) const;
	bool get_bool(const std::string &name, bool fallback) const;

	private:
	std::unordered_map<std::string, std::string> values_;
	std::unordered_set<std::string> flags_;
};

std::string timestamp();
std::string create_run_id(const std::string &prefix);

} // namespace gadus
