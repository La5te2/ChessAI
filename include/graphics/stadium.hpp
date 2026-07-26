// Encapsulates one independently running UCI match, its engines, game, and history view.
#pragma once
#include "graphics/game.hpp"
#include "graphics/uci.hpp"
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace gadidae::graphics {

class StadiumSession {
public:
	explicit StadiumSession(std::size_t id);
	~StadiumSession();
	StadiumSession(const StadiumSession &) = delete;
	StadiumSession &operator=(const StadiumSession &) = delete;

	/// Returns the live match and the position currently selected for display.
	const GameState &game() const;
	const GameState &visible_game() const;

	/// Returns stable identity and the optional user-defined match name.
	std::size_t id() const;
	const std::string &name() const;
	void set_name(std::string name);
	std::string display_name() const;

	/// Returns or changes both UCI engine configurations.
	const EngineConfig &white_config() const;
	const EngineConfig &black_config() const;
	EngineConfig &white_config();
	EngineConfig &black_config();
	void set_configs(const EngineConfig &white, const EngineConfig &black);

	/// Stores match timing and termination limits.
	int display_delay_ms() const;
	int max_plies() const;
	void set_match_limits(int display_delay_ms, int max_plies);

	/// Configures one shared time control; zero initial time disables the clock.
	std::int64_t clock_initial_ms() const;
	std::int64_t clock_increment_ms() const;
	void set_clock(std::int64_t initial_ms, std::int64_t increment_ms);
	std::int64_t white_remaining_ms() const;
	std::int64_t black_remaining_ms() const;
	bool clock_enabled() const;

	/// Stores and applies the position from which Start and Reset begin.
	const std::string &start_fen() const;
	void set_start_fen(const std::string &fen);
	void reset();

	/// Controls the lifetime of this match and both owned UCI processes.
	void start();
	void stop();
	void toggle_pause();
	void update();

	/// Applies one legal board move when the current side is configured as Human.
	void make_human_move(const chess::Move &move);
	bool human_to_move() const;

	/// Returns observable match state for menus and presentation.
	bool running() const;
	bool paused() const;
	const std::string &status() const;
	const AnalysisSnapshot &display() const;
	std::string white_name() const;
	std::string black_name() const;
	std::string result() const;
	std::string termination() const;

	/// Selects a read-only historical ply or returns to the live position.
	std::optional<std::size_t> viewed_ply() const;
	void view_ply(std::size_t ply);
	void follow_live();

	/// Returns and clears an asynchronous match error for UI presentation.
	std::optional<std::string> take_error();

private:
	using Clock = std::chrono::steady_clock;

	/// Returns the engine and configuration for the current side to move.
	UciEngine &active_engine();
	const EngineConfig &active_config() const;
	bool engine_configured(chess::Color color) const;
	bool engines_ready();
	std::int64_t remaining_ms(chess::Color color,
							  Clock::time_point now) const;
	bool finish_turn_clock(Clock::time_point now, bool add_increment);
	void finish_on_time(chess::Color loser);

	std::size_t id_ = 0;
	std::string name_;
	GameState game_;
	GameState preview_;
	UciEngine white_engine_;
	UciEngine black_engine_;
	EngineConfig white_config_;
	EngineConfig black_config_;
	std::optional<std::size_t> viewed_ply_;
	std::optional<std::string> error_;
	std::string start_fen_ = "startpos";
	std::string root_fen_;
	std::string status_ = "Ready";
	std::string result_override_;
	std::string termination_override_;
	AnalysisSnapshot display_;
	Clock::time_point last_display_{};
	Clock::time_point next_turn_{};
	Clock::time_point turn_clock_started_{};
	std::uint64_t generation_ = 0;
	std::int64_t clock_initial_ms_ = 0;
	std::int64_t clock_increment_ms_ = 0;
	std::int64_t white_remaining_ms_ = 0;
	std::int64_t black_remaining_ms_ = 0;
	int display_delay_ms_ = 250;
	int max_plies_ = 240;
	bool running_ = false;
	bool paused_ = false;
	bool turn_started_ = false;
};

/// Owns every Stadium match, advances them in the background, and selects one for display.
class StadiumWorkspace {
public:
	StadiumWorkspace();
	~StadiumWorkspace();
	StadiumWorkspace(const StadiumWorkspace &) = delete;
	StadiumWorkspace &operator=(const StadiumWorkspace &) = delete;

	/// Returns the session currently presented by the Stadium interface.
	StadiumSession &active();
	const StadiumSession &active() const;
	StadiumSession &at(std::size_t index);
	const StadiumSession &at(std::size_t index) const;

	/// Creates, selects, or removes independent match sessions.
	std::size_t create_session();
	void select(std::size_t index);
	void close(std::size_t index);
	std::size_t active_index() const;
	std::size_t size() const;
	std::size_t next_id() const;

	/// Advances every match and terminates every owned UCI process at shutdown.
	void update_all();
	void stop_all();

	/// Collects asynchronous errors from all background sessions.
	std::vector<std::string> take_errors();

private:
	std::vector<std::unique_ptr<StadiumSession>> sessions_;
	std::size_t active_index_ = 0;
	std::size_t next_id_ = 1;
};

} // namespace gadidae::graphics
