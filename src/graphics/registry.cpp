// Implements the persistent, process-free registry used by the UCI engine importer.
#include "graphics/registry.hpp"
#include <algorithm>
#include <stdexcept>
#include <utility>

namespace gadidae::graphics {

std::size_t Registry::add(EngineConfig config) {
	if (config.path.empty()) {
		throw std::invalid_argument("UCI engine path is empty");
	}
	if (!std::filesystem::is_regular_file(config.path)) {
		throw std::invalid_argument("UCI engine executable does not exist");
	}
	config.path = normalized_path(config.path);
	if (config.name.empty()) {
		config.name = config.path.stem().string();
	}
	const auto existing = std::find_if(engines_.begin(), engines_.end(), [&](const EngineConfig &engine) { return normalized_path(engine.path) == config.path; });
	if (existing != engines_.end()) {
		*existing = std::move(config);
		return static_cast<std::size_t>(std::distance(engines_.begin(), existing));
	}
	engines_.push_back(std::move(config));
	return engines_.size() - 1;
}

void Registry::replace(std::vector<EngineConfig> engines) {
	engines_.clear();
	for (auto &engine : engines) {
		try {
			add(std::move(engine));
		} catch (const std::invalid_argument &) {
			// A moved or removed executable should not invalidate all GUI settings.
		}
	}
}

const std::vector<EngineConfig> &Registry::engines() const {
	return engines_;
}

std::filesystem::path Registry::normalized_path(const std::filesystem::path &path) {
	std::error_code error;
	auto normalized = std::filesystem::weakly_canonical(path, error);
	if (error) {
		normalized = std::filesystem::absolute(path, error);
	}
	return error ? path.lexically_normal() : normalized.lexically_normal();
}

} // namespace gadidae::graphics
