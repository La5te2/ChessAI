// Implements independently running Stadium matches with optional Human seats and clocks.
#include "graphics/stadium.hpp"
#include <algorithm>
#include <filesystem>
#include <stdexcept>
#include <utility>

namespace gadidae::graphics {

StadiumSession::StadiumSession(std::size_t id) : id_(id) {
}

StadiumSession::~StadiumSession() {
	stop();
}

const GameState &StadiumSession::game() const {
	return game_;
}

const GameState &StadiumSession::visible_game() const {
	return viewed_ply_ ? preview_ : game_;
}

std::size_t StadiumSession::id() const {
	return id_;
}

const std::string &StadiumSession::name() const {
	return name_;
}

void StadiumSession::set_name(std::string name) {
	name_ = std::move(name);
}

std::string StadiumSession::display_name() const {
	if (!name_.empty()) {
		return name_;
	}
	return "#" + std::to_string(id_) + " " + white_name() + " vs. " + black_name();
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

void StadiumSession::set_configs(const EngineConfig &white, const EngineConfig &black) {
	if (white_config_ == white && black_config_ == black) {
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

std::int64_t StadiumSession::clock_initial_ms() const {
	return clock_initial_ms_;
}

std::int64_t StadiumSession::clock_increment_ms() const {
	return clock_increment_ms_;
}

void StadiumSession::set_clock(std::int64_t initial_ms, std::int64_t increment_ms) {
	if (running_) {
		throw std::logic_error("stop the match before changing its clock");
	}
	clock_initial_ms_ = std::max<std::int64_t>(0, initial_ms);
	clock_increment_ms_ = std::max<std::int64_t>(0, increment_ms);
	white_remaining_ms_ = clock_initial_ms_;
	black_remaining_ms_ = clock_initial_ms_;
}

std::int64_t StadiumSession::remaining_ms(chess::Color color, Clock::time_point now) const {
	const auto stored = color == chess::Color::WHITE ? white_remaining_ms_ : black_remaining_ms_;
	if (!clock_enabled() || !running_ || paused_ || !turn_started_ || game_.board().sideToMove() != color) {
		return stored;
	}
	const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - turn_clock_started_).count();
	return std::max<std::int64_t>(0, stored - elapsed);
}

std::int64_t StadiumSession::white_remaining_ms() const {
	return remaining_ms(chess::Color::WHITE, Clock::now());
}

std::int64_t StadiumSession::black_remaining_ms() const {
	return remaining_ms(chess::Color::BLACK, Clock::now());
}

bool StadiumSession::clock_enabled() const {
	return clock_initial_ms_ > 0;
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
	result_override_.clear();
	termination_override_.clear();
	white_remaining_ms_ = clock_initial_ms_;
	black_remaining_ms_ = clock_initial_ms_;
	status_ = "Ready";
}

bool StadiumSession::engine_configured(chess::Color color) const {
	const auto &config = color == chess::Color::WHITE ? white_config_ : black_config_;
	return !config.path.empty();
}

bool StadiumSession::engines_ready() {
	const auto check = [](const char *side, const UciEngine &engine, bool configured) {
		if (!configured) {
			return true;
		}
		if (engine.starting()) {
			return false;
		}
		const auto snapshot = engine.snapshot();
		if (!snapshot.error.empty()) {
			throw std::runtime_error(std::string(side) + " engine: " + snapshot.error);
		}
		if (engine.ready()) {
			return true;
		}
		throw std::runtime_error(std::string(side) + " engine initialization failed");
	};
	const bool ready = check("White", white_engine_, engine_configured(chess::Color::WHITE)) && check("Black", black_engine_, engine_configured(chess::Color::BLACK));
	if (ready) {
		if (engine_configured(chess::Color::WHITE)) {
			white_config_.discovered_options = white_engine_.option_definitions();
			white_config_.button_commands.clear();
		}
		if (engine_configured(chess::Color::BLACK)) {
			black_config_.discovered_options = black_engine_.option_definitions();
			black_config_.button_commands.clear();
		}
	}
	return ready;
}

void StadiumSession::start() {
	stop();
	game_.reset(start_fen_);
	follow_live();
	error_.reset();
	display_ = {};
	result_override_.clear();
	termination_override_.clear();
	white_remaining_ms_ = clock_initial_ms_;
	black_remaining_ms_ = clock_initial_ms_;
	if (engine_configured(chess::Color::WHITE)) {
		white_engine_.start_async(white_config_);
	}
	if (engine_configured(chess::Color::BLACK)) {
		black_engine_.start_async(black_config_);
	}
	running_ = true;
	paused_ = false;
	turn_started_ = false;
	next_turn_ = Clock::now();
	status_ = engine_configured(chess::Color::WHITE) || engine_configured(chess::Color::BLACK) ? "Loading engines" : "Running";
}

void StadiumSession::stop() {
	white_engine_.close();
	black_engine_.close();
	running_ = false;
	paused_ = false;
	turn_started_ = false;
	if (status_ == "Running" || status_ == "Paused" || status_ == "Loading engines" || status_ == "Waiting for Human") {
		status_ = "Stopped";
	}
}

bool StadiumSession::finish_turn_clock(Clock::time_point now, bool add_increment) {
	if (!clock_enabled() || !turn_started_) {
		return true;
	}
	const auto side = game_.board().sideToMove();
	auto &stored = side == chess::Color::WHITE ? white_remaining_ms_ : black_remaining_ms_;
	const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - turn_clock_started_).count();
	stored = std::max<std::int64_t>(0, stored - elapsed);
	if (stored == 0) {
		finish_on_time(side);
		return false;
	}
	if (add_increment) {
		stored += clock_increment_ms_;
	}
	return true;
}

void StadiumSession::finish_on_time(chess::Color loser) {
	const bool white_lost = loser == chess::Color::WHITE;
	result_override_ = white_lost ? "0-1" : "1-0";
	termination_override_ = "time forfeit";
	status_ = white_lost ? "White lost on time" : "Black lost on time";
	white_engine_.close();
	black_engine_.close();
	running_ = false;
	paused_ = false;
	turn_started_ = false;
}

void StadiumSession::toggle_pause() {
	if (!running_) {
		return;
	}
	if (!paused_) {
		if (turn_started_ && !finish_turn_clock(Clock::now(), false)) {
			return;
		}
		if (engine_configured(game_.board().sideToMove())) {
			active_engine().stop_search();
		}
		paused_ = true;
		turn_started_ = false;
		status_ = "Paused";
		return;
	}
	paused_ = false;
	turn_started_ = false;
	next_turn_ = Clock::now();
	status_ = "Running";
}

void StadiumSession::update() {
	if (!running_ || paused_) {
		return;
	}
	try {
		if (!engines_ready()) {
			status_ = "Loading engines";
			return;
		}
		if (status_ == "Loading engines") {
			status_ = "Running";
		}
		if (game_.over()) {
			status_ = game_.termination();
			stop();
			return;
		}
		if (game_.plies() >= static_cast<std::size_t>(max_plies_)) {
			status_ = "Maximum plies reached";
			stop();
			return;
		}

		const auto now = Clock::now();
		if (turn_started_ && clock_enabled() && remaining_ms(game_.board().sideToMove(), now) == 0) {
			finish_on_time(game_.board().sideToMove());
			return;
		}
		if (!turn_started_) {
			if (now < next_turn_) {
				return;
			}
			turn_started_ = true;
			turn_clock_started_ = now;
			last_display_ = Clock::time_point{};
			if (!engine_configured(game_.board().sideToMove())) {
				display_ = {};
				status_ = "Waiting for Human";
				return;
			}
			root_position_ = game_.uci_position();
			generation_ = active_engine().analyse(root_position_, false);
			display_ = active_engine().snapshot();
			status_ = "Running";
			return;
		}

		if (!engine_configured(game_.board().sideToMove())) {
			status_ = "Waiting for Human";
			return;
		}
		const auto snapshot = active_engine().snapshot();
		if (!snapshot.error.empty()) {
			throw std::runtime_error(snapshot.error);
		}
		constexpr int interval = 100;
		if (last_display_ == Clock::time_point{} || now - last_display_ >= std::chrono::milliseconds(interval) || snapshot.finished) {
			display_ = snapshot;
			last_display_ = now;
		}
		if (snapshot.generation != generation_ || !snapshot.finished) {
			return;
		}
		if (snapshot.bestmove.empty()) {
			throw std::runtime_error("UCI engine returned no legal bestmove");
		}
		if (!finish_turn_clock(now, true)) {
			return;
		}
		game_.make_uci(snapshot.bestmove);
		follow_live();
		turn_started_ = false;
		next_turn_ = now + std::chrono::milliseconds(display_delay_ms_);
	} catch (const std::exception &exception) {
		error_ = exception.what();
		status_ = "Error";
		stop();
	}
}

void StadiumSession::make_human_move(const chess::Move &move) {
	if (!human_to_move()) {
		throw std::logic_error("the current seat is not ready for a Human move");
	}
	const auto now = Clock::now();
	if (!finish_turn_clock(now, true)) {
		return;
	}
	game_.make_move(move);
	follow_live();
	display_ = {};
	turn_started_ = false;
	next_turn_ = now + std::chrono::milliseconds(display_delay_ms_);
	status_ = "Running";
}

bool StadiumSession::human_to_move() const {
	return running_ && !paused_ && turn_started_ && !engine_configured(game_.board().sideToMove()) && !game_.over();
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
	if (!engine_configured(chess::Color::WHITE)) {
		return white_config_.name.empty() ? "Human" : white_config_.name;
	}
	if (!white_config_.name.empty()) {
		return white_config_.name;
	}
	const auto reported = white_engine_.display_name();
	if (!reported.empty()) {
		return reported;
	}
	const auto stem = white_config_.path.stem().string();
	return stem.empty() ? "White" : stem;
}

std::string StadiumSession::black_name() const {
	if (!engine_configured(chess::Color::BLACK)) {
		return black_config_.name.empty() ? "Human" : black_config_.name;
	}
	if (!black_config_.name.empty()) {
		return black_config_.name;
	}
	const auto reported = black_engine_.display_name();
	if (!reported.empty()) {
		return reported;
	}
	const auto stem = black_config_.path.stem().string();
	return stem.empty() ? "Black" : stem;
}

std::string StadiumSession::result() const {
	return result_override_.empty() ? game_.result() : result_override_;
}

std::string StadiumSession::termination() const {
	return termination_override_.empty() ? game_.termination() : termination_override_;
}

std::optional<std::size_t> StadiumSession::viewed_ply() const {
	return viewed_ply_;
}

void StadiumSession::view_ply(std::size_t ply) {
	if (ply >= game_.plies()) {
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
	return game_.board().sideToMove() == chess::Color::WHITE ? white_engine_ : black_engine_;
}

const EngineConfig &StadiumSession::active_config() const {
	return game_.board().sideToMove() == chess::Color::WHITE ? white_config_ : black_config_;
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

StadiumSession &StadiumWorkspace::at(std::size_t index) {
	return *sessions_.at(index);
}

const StadiumSession &StadiumWorkspace::at(std::size_t index) const {
	return *sessions_.at(index);
}

std::size_t StadiumWorkspace::create_session() {
	sessions_.push_back(std::make_unique<StadiumSession>(next_id_++));
	active_index_ = sessions_.size() - 1;
	return active_index_;
}

void StadiumWorkspace::select(std::size_t index) {
	if (index >= sessions_.size()) {
		throw std::out_of_range("Stadium session index is out of range");
	}
	active_index_ = index;
}

void StadiumWorkspace::close(std::size_t index) {
	if (index >= sessions_.size()) {
		throw std::out_of_range("Stadium session index is out of range");
	}
	sessions_[index]->stop();
	sessions_.erase(sessions_.begin() + static_cast<std::ptrdiff_t>(index));
	if (sessions_.empty()) {
		create_session();
		return;
	}
	if (active_index_ > index) {
		--active_index_;
	} else if (active_index_ >= sessions_.size()) {
		active_index_ = sessions_.size() - 1;
	}
}

std::size_t StadiumWorkspace::active_index() const {
	return active_index_;
}

std::size_t StadiumWorkspace::size() const {
	return sessions_.size();
}

std::size_t StadiumWorkspace::next_id() const {
	return next_id_;
}

void StadiumWorkspace::update_all() {
	for (auto &session : sessions_) {
		session->update();
	}
}

void StadiumWorkspace::stop_all() {
	for (auto &session : sessions_) {
		session->stop();
	}
}

std::vector<std::string> StadiumWorkspace::take_errors() {
	std::vector<std::string> errors;
	for (const auto &session : sessions_) {
		if (auto error = session->take_error()) {
			errors.push_back(session->display_name() + ": " + *error);
		}
	}
	return errors;
}

} // namespace gadidae::graphics
