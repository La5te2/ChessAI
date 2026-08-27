// Stores reusable UCI engine configurations independently from running sessions.
#pragma once
#include "graphics/uci.hpp"
#include <cstddef>
#include <filesystem>
#include <vector>

namespace gadidae::graphics {

class Registry {
public:
	/// Adds or replaces an engine identified by its normalized executable path.
	std::size_t add(EngineConfig config);

	/// Replaces all persisted entries after validating and de-duplicating them.
	void replace(std::vector<EngineConfig> engines);

	/// Returns every imported engine in stable presentation order.
	const std::vector<EngineConfig> &engines() const;

private:
	/// Produces a comparable absolute path without requiring the target to exist.
	static std::filesystem::path normalized_path(const std::filesystem::path &path);

	std::vector<EngineConfig> engines_;
};

} // namespace gadidae::graphics
