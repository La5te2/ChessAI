// Implements the non-visual Simulator workspace and its asynchronous UCI lifecycle.
#include "graphics/simulator.hpp"
#include <algorithm>
#include <stdexcept>

namespace gadidae::graphics {

const GameState &SimulatorWorkspace::game() const {
	return game_;
}

GameState &SimulatorWorkspace::game() {
	return game_;
}

const GameState &SimulatorWorkspace::visible_game() const {
	return viewed_ply_ ? preview_ : game_;
}

const EngineConfig &SimulatorWorkspace::config() const {
	return config_;
}

EngineConfig &SimulatorWorkspace::config() {
	return config_;
}

void SimulatorWorkspace::set_config(const EngineConfig &config) {
	if(config_ == config) {
		return;
	}
	engine_.close();
	config_ = config;
	invalidate_analysis();
}

const AnalysisSnapshot &SimulatorWorkspace::display() const {
	return display_;
}

void SimulatorWorkspace::update_analysis() {
	if(!analysis_open_ || config_.path.empty()) {
		return;
	}
	const std::string position = visible_game().uci_position();
	try {
		if(!engine_.ready()) {
			const auto startup = engine_.snapshot();
			if(!startup.error.empty()) {
				throw std::runtime_error(startup.error);
			}
			if(!engine_.starting()) {
				engine_.start_async(config_);
				display_ = {};
				display_.engine_name = "Loading engine";
			}
			return;
		}
		config_.discovered_options = engine_.option_definitions();
		config_.button_commands.clear();
		if(analysis_position_ != position) {
			engine_.analyse(position, true);
			analysis_position_ = position;
			last_display_ = Clock::time_point{};
		}
		const auto now = Clock::now();
		const auto snapshot = engine_.snapshot();
		if(last_display_ == Clock::time_point{} ||
		   now - last_display_ >=
			   std::chrono::milliseconds(100) ||
		   snapshot.finished) {
			display_ = snapshot;
			last_display_ = now;
		}
	} catch(...) {
		analysis_open_ = false;
		engine_.close();
		throw;
	}
}

void SimulatorWorkspace::invalidate_analysis() {
	engine_.stop_search();
	analysis_position_.clear();
	display_ = {};
	last_display_ = Clock::time_point{};
}

bool SimulatorWorkspace::analysis_open() const {
	return analysis_open_;
}

void SimulatorWorkspace::start_analysis() {
	if(analysis_open_) {
		return;
	}
	analysis_open_ = true;
	invalidate_analysis();
}

void SimulatorWorkspace::stop_analysis() {
	if(!analysis_open_) {
		return;
	}
	analysis_open_ = false;
	engine_.stop_search();
	display_ = {};
}

const std::string &SimulatorWorkspace::start_fen() const {
	return start_fen_;
}

void SimulatorWorkspace::set_start_fen(const std::string &fen) {
	const std::string normalized = fen.empty() ? "startpos" : fen;
	GameState validation;
	validation.reset(normalized);
	start_fen_ = normalized;
}

void SimulatorWorkspace::reset() {
	game_.reset(start_fen_);
	follow_live();
	invalidate_analysis();
}

void SimulatorWorkspace::make_move(const chess::Move &move) {
	game_.make_move(move);
	follow_live();
	invalidate_analysis();
}

bool SimulatorWorkspace::undo() {
	if(!game_.undo()) {
		return false;
	}
	follow_live();
	invalidate_analysis();
	return true;
}

void SimulatorWorkspace::import_pgn(const std::string &document) {
	game_.import_pgn(document);
	start_fen_ = game_.start_fen();
	follow_live();
	invalidate_analysis();
}

bool SimulatorWorkspace::viewing_history() const {
	return viewed_ply_.has_value();
}

std::optional<std::size_t> SimulatorWorkspace::viewed_ply() const {
	return viewed_ply_;
}

void SimulatorWorkspace::view_ply(std::size_t ply) {
	if(ply >= game_.plies()) {
		follow_live();
		return;
	}
	preview_ = game_.position_at(ply);
	viewed_ply_ = ply;
	invalidate_analysis();
}

void SimulatorWorkspace::follow_live() {
	viewed_ply_.reset();
}

} // namespace gadidae::graphics
