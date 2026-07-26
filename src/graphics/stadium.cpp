// Implements one background-capable Stadium match with two owned UCI child processes.
#include "graphics/stadium.hpp"
#include <algorithm>
#include <stdexcept>
#include <utility>

namespace gadidae::graphics {

StadiumSession::~StadiumSession() {
	stop();
}

const GameState &StadiumSession::game() const {
	return game_;
}

const GameState &StadiumSession::visible_game() const {
	return viewed_ply_ ? preview_ : game_;
}

const EngineConfig &StadiumSession::white_config() const {
	return white_config_;
}

const EngineConfig &StadiumSession::black_config() const {
	return black_config_;
}

EngineConfig &StadiumSession::white_config() {
	return white_config_;
}

EngineConfig &StadiumSession::black_config() {
	return black_config_;
}

void StadiumSession::set_configs(const EngineConfig &white,
								 const EngineConfig &black) {
	if(white_config_ == white && black_config_ == black) {
		return;
	}
	stop();
	white_config_ = white;
	black_config_ = black;
}

int StadiumSession::display_delay_ms() const {
	return display_delay_ms_;
}

int StadiumSession::max_plies() const {
	return max_plies_;
}

void StadiumSession::set_match_limits(int display_delay_ms, int max_plies) {
	display_delay_ms_ = std::max(0, display_delay_ms);
	max_plies_ = std::max(1, max_plies);
}

const std::string &StadiumSession::start_fen() const {
	return start_fen_;
}

void StadiumSession::set_start_fen(const std::string &fen) {
	const std::string normalized = fen.empty() ? "startpos" : fen;
	GameState validation;
	validation.reset(normalized);
	start_fen_ = normalized;
}

void StadiumSession::reset() {
	stop();
	game_.reset(start_fen_);
	follow_live();
	display_ = {};
	status_ = "Ready";
}

void StadiumSession::start() {
	if(white_config_.path.empty() || black_config_.path.empty()) {
		throw std::invalid_argument("Stadium requires two UCI engine paths");
	}
	stop();
	game_.reset(start_fen_);
	follow_live();
	error_.reset();
	display_ = {};
	white_engine_.start_async(white_config_);
	black_engine_.start_async(black_config_);
	running_ = true;
	paused_ = false;
	turn_started_ = false;
	next_turn_ = Clock::now();
	status_ = "Loading engines";
}

void StadiumSession::stop() {
	white_engine_.close();
	black_engine_.close();
	running_ = false;
	paused_ = false;
	turn_started_ = false;
	if(status_ == "Running" || status_ == "Paused" ||
	   status_ == "Loading engines") {
		status_ = "Stopped";
	}
}

void StadiumSession::toggle_pause() {
	if(!running_) {
		return;
	}
	paused_ = !paused_;
	if(paused_) {
		active_engine().stop_search();
		turn_started_ = false;
		status_ = "Paused";
		return;
	}
	next_turn_ = Clock::now();
	status_ = "Running";
}

void StadiumSession::update() {
	if(!running_ || paused_) {
		return;
	}
	try {
		if(white_engine_.starting() || black_engine_.starting()) {
			status_ = "Loading engines";
			return;
		}
		if(!white_engine_.ready() || !black_engine_.ready()) {
			const auto white = white_engine_.snapshot();
			const auto black = black_engine_.snapshot();
			if(!white.error.empty()) {
				throw std::runtime_error("White engine: " + white.error);
			}
			if(!black.error.empty()) {
				throw std::runtime_error("Black engine: " + black.error);
			}
			throw std::runtime_error("UCI engine initialization failed");
		}
		if(status_ == "Loading engines") {
			status_ = "Running";
		}
		if(game_.over()) {
			status_ = game_.termination();
			stop();
			return;
		}
		if(game_.plies() >= static_cast<std::size_t>(max_plies_)) {
			status_ = "Maximum plies reached";
			stop();
			return;
		}

		const auto now = Clock::now();
		if(!turn_started_) {
			if(now < next_turn_) {
				return;
			}
			root_fen_ = game_.board().getFen();
			generation_ = active_engine().analyse(root_fen_, false);
			turn_started_ = true;
			display_ = active_engine().snapshot();
			last_display_ = Clock::time_point{};
			return;
		}

		const auto snapshot = active_engine().snapshot();
		const auto interval =
			std::max(50, active_config().progress_interval_ms);
		if(last_display_ == Clock::time_point{} ||
		   now - last_display_ >= std::chrono::milliseconds(interval) ||
		   snapshot.finished) {
			display_ = snapshot;
			last_display_ = now;
		}
		if(snapshot.generation != generation_ || !snapshot.finished) {
			return;
		}
		if(snapshot.bestmove.empty()) {
			throw std::runtime_error("UCI engine returned no legal bestmove");
		}
		game_.make_uci(snapshot.bestmove);
		follow_live();
		turn_started_ = false;
		next_turn_ =
			now + std::chrono::milliseconds(display_delay_ms_);
	} catch(const std::exception &exception) {
		error_ = exception.what();
		status_ = "Error";
		stop();
	}
}

bool StadiumSession::running() const {
	return running_;
}

bool StadiumSession::paused() const {
	return paused_;
}

const std::string &StadiumSession::status() const {
	return status_;
}

const AnalysisSnapshot &StadiumSession::display() const {
	return display_;
}

std::string StadiumSession::white_name() const {
	if(!white_config_.name.empty()) {
		return white_config_.name;
	}
	const auto reported = white_engine_.display_name();
	return reported.empty() ? "White" : reported;
}

std::string StadiumSession::black_name() const {
	if(!black_config_.name.empty()) {
		return black_config_.name;
	}
	const auto reported = black_engine_.display_name();
	return reported.empty() ? "Black" : reported;
}

std::optional<std::size_t> StadiumSession::viewed_ply() const {
	return viewed_ply_;
}

void StadiumSession::view_ply(std::size_t ply) {
	if(ply >= game_.plies()) {
		follow_live();
		return;
	}
	preview_ = game_.position_at(ply);
	viewed_ply_ = ply;
}

void StadiumSession::follow_live() {
	viewed_ply_.reset();
}

std::optional<std::string> StadiumSession::take_error() {
	auto result = std::move(error_);
	error_.reset();
	return result;
}

UciEngine &StadiumSession::active_engine() {
	return game_.board().sideToMove() == chess::Color::WHITE ? white_engine_
															 : black_engine_;
}

const EngineConfig &StadiumSession::active_config() const {
	return game_.board().sideToMove() == chess::Color::WHITE ? white_config_
															 : black_config_;
}

StadiumWorkspace::StadiumWorkspace() {
	create_session();
}

StadiumWorkspace::~StadiumWorkspace() {
	stop_all();
}

StadiumSession &StadiumWorkspace::active() {
	return *sessions_.at(active_index_);
}

const StadiumSession &StadiumWorkspace::active() const {
	return *sessions_.at(active_index_);
}

std::size_t StadiumWorkspace::create_session() {
	sessions_.push_back(std::make_unique<StadiumSession>());
	active_index_ = sessions_.size() - 1;
	return active_index_;
}

void StadiumWorkspace::select(std::size_t index) {
	if(index >= sessions_.size()) {
		throw std::out_of_range("Stadium session index is out of range");
	}
	active_index_ = index;
}

void StadiumWorkspace::close(std::size_t index) {
	if(index >= sessions_.size()) {
		throw std::out_of_range("Stadium session index is out of range");
	}
	sessions_[index]->stop();
	sessions_.erase(sessions_.begin() + static_cast<std::ptrdiff_t>(index));
	if(sessions_.empty()) {
		create_session();
		return;
	}
	if(active_index_ > index) {
		--active_index_;
	} else if(active_index_ >= sessions_.size()) {
		active_index_ = sessions_.size() - 1;
	}
}

std::size_t StadiumWorkspace::active_index() const {
	return active_index_;
}

std::size_t StadiumWorkspace::size() const {
	return sessions_.size();
}

void StadiumWorkspace::update_all() {
	for(auto &session : sessions_) {
		session->update();
	}
}

void StadiumWorkspace::stop_all() {
	for(auto &session : sessions_) {
		session->stop();
	}
}

std::vector<std::string> StadiumWorkspace::take_errors() {
	std::vector<std::string> errors;
	for(std::size_t index = 0; index < sessions_.size(); ++index) {
		if(auto error = sessions_[index]->take_error()) {
			errors.push_back("Stadium session " + std::to_string(index + 1) +
							 ": " + *error);
		}
	}
	return errors;
}

} // namespace gadidae::graphics
