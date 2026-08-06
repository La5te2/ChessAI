// Encapsulates Simulator chess state, one UCI engine, live analysis, and history view.
#pragma once
#include "graphics/game.hpp"
#include "graphics/uci.hpp"
#include <chrono>
#include <cstddef>
#include <optional>
#include <string>

namespace gadidae::graphics {

class SimulatorWorkspace {
public:
	SimulatorWorkspace() = default;
	SimulatorWorkspace(const SimulatorWorkspace &) = delete;
	SimulatorWorkspace &operator=(const SimulatorWorkspace &) = delete;

	/// Returns the complete live game and the currently observed position.
	const GameState &game() const;
	GameState &game();
	const GameState &visible_game() const;

	/// Returns or replaces the engine configuration used by this workspace.
	const EngineConfig &config() const;
	EngineConfig &config();
	void set_config(const EngineConfig &config);

	/// Returns the latest immutable analysis snapshot.
	const AnalysisSnapshot &display() const;

	/// Advances asynchronous engine startup and analysis for the observed position.
	void update_analysis();
	void invalidate_analysis();

	/// Enables or disables analysis without changing the board.
	bool analysis_open() const;
	void start_analysis();
	void stop_analysis();

	/// Stores and applies the position from which Reset starts a new game.
	const std::string &start_fen() const;
	void set_start_fen(const std::string &fen);
	void reset();

	/// Applies game mutations and returns the history view to the live position.
	void make_move(const chess::Move &move);
	bool undo();
	void import_pgn(const std::string &document);

	/// Selects a read-only historical ply or returns to the current position.
	bool viewing_history() const;
	std::optional<std::size_t> viewed_ply() const;
	void view_ply(std::size_t ply);
	void follow_live();

private:
	using Clock = std::chrono::steady_clock;

	GameState game_;
	GameState preview_;
	UciEngine engine_;
	EngineConfig config_;
	std::string start_fen_ = "startpos";
	std::optional<std::size_t> viewed_ply_;
	std::string analysis_position_;
	AnalysisSnapshot display_;
	Clock::time_point last_display_{};
	bool analysis_open_ = true;
};

} // namespace gadidae::graphics
