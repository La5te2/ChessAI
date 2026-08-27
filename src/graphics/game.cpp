// Implements legal game mutation, reversible history, and PGN/FEN serialization.
#include "graphics/game.hpp"
#include <algorithm>
#include <sstream>
#include <stdexcept>

namespace gadidae::graphics {
namespace {

/// Escapes one PGN tag value according to PGN string syntax.
std::string escape_pgn(const std::string &value) {
	std::string escaped;
	for (const char character : value) {
		if (character == '\\' || character == '"') {
			escaped.push_back('\\');
		}
		escaped.push_back(character);
	}
	return escaped;
}

/// Wraps PGN movetext without splitting SAN tokens.
std::string wrap_movetext(const std::string &text, std::size_t width = 80) {
	std::istringstream input(text);
	std::ostringstream output;
	std::size_t column = 0;
	for (std::string token; input >> token;) {
		const std::size_t required = token.size() + (column == 0 ? 0 : 1);
		if (column > 0 && column + required > width) {
			output << '\n';
			column = 0;
		}
		if (column > 0) {
			output << ' ';
			++column;
		}
		output << token;
		column += token.size();
	}
	return output.str();
}

} // namespace

GameState::GameState() {
	reset("startpos");
}

void GameState::reset(const std::string &fen) {
	start_fen_ = fen.empty() || fen == "startpos" ? std::string(chess::constants::STARTPOS) : fen;
	board_ = chess::Board(start_fen_);
	moves_.clear();
	san_.clear();
}

const chess::Board &GameState::board() const {
	return board_;
}

chess::Board &GameState::board() {
	return board_;
}

const std::string &GameState::start_fen() const {
	return start_fen_;
}

std::vector<chess::Move> GameState::legal_moves() const {
	chess::Movelist moves;
	chess::movegen::legalmoves(moves, board_);
	return {moves.begin(), moves.end()};
}

std::vector<chess::Move> GameState::legal_moves_from(int square) const {
	std::vector<chess::Move> result;
	for (const auto &move : legal_moves()) {
		if (move.from().index() == square) {
			result.push_back(move);
		}
	}
	return result;
}

void GameState::make_move(const chess::Move &move) {
	const auto legal = legal_moves();
	if (std::find(legal.begin(), legal.end(), move) == legal.end()) {
		throw std::invalid_argument("illegal move");
	}
	san_.push_back(chess::uci::moveToSan(board_, move));
	moves_.push_back(move);
	board_.makeMove(move);
}

void GameState::make_uci(const std::string &uci) {
	make_move(chess::uci::uciToMove(board_, uci));
}

bool GameState::undo() {
	if (moves_.empty()) {
		return false;
	}
	board_.unmakeMove(moves_.back());
	moves_.pop_back();
	san_.pop_back();
	return true;
}

bool GameState::over() const {
	return board_.isGameOver().first != chess::GameResultReason::NONE;
}

std::string GameState::result() const {
	const auto outcome = board_.isGameOver();
	if (outcome.first == chess::GameResultReason::NONE) {
		return "*";
	}
	if (outcome.second == chess::GameResult::DRAW) {
		return "1/2-1/2";
	}
	const bool side_wins = outcome.second == chess::GameResult::WIN;
	const bool white_wins = side_wins == (board_.sideToMove() == chess::Color::WHITE);
	return white_wins ? "1-0" : "0-1";
}

std::string GameState::termination() const {
	switch (board_.isGameOver().first) {
	case chess::GameResultReason::CHECKMATE:
		return "checkmate";
	case chess::GameResultReason::STALEMATE:
		return "stalemate";
	case chess::GameResultReason::INSUFFICIENT_MATERIAL:
		return "insufficient material";
	case chess::GameResultReason::FIFTY_MOVE_RULE:
		return "fifty-move rule";
	case chess::GameResultReason::THREEFOLD_REPETITION:
		return "threefold repetition";
	default:
		return over() ? "game over" : "";
	}
}

std::size_t GameState::plies() const {
	return moves_.size();
}

std::optional<chess::Move> GameState::last_move() const {
	if (moves_.empty()) {
		return std::nullopt;
	}
	return moves_.back();
}

GameState GameState::position_at(std::size_t ply) const {
	if (ply > moves_.size()) {
		throw std::out_of_range("history ply exceeds the current game");
	}
	GameState position = *this;
	while (position.plies() > ply) {
		position.undo();
	}
	return position;
}

std::string GameState::uci_position() const {
	std::ostringstream position;
	position << "fen " << start_fen_;
	if (!moves_.empty()) {
		position << " moves";
		for (const auto &move : moves_) {
			position << ' ' << chess::uci::moveToUci(move);
		}
	}
	return position.str();
}

std::string GameState::movetext(const std::string &result_override) const {
	std::ostringstream text;
	for (std::size_t index = 0; index < san_.size(); ++index) {
		if (index % 2 == 0) {
			if (index > 0) {
				text << ' ';
			}
			text << (index / 2 + 1) << ". ";
		} else {
			text << ' ';
		}
		text << san_[index];
	}
	if (!san_.empty()) {
		text << ' ';
	}
	text << (result_override.empty() ? result() : result_override);
	return wrap_movetext(text.str());
}

std::string GameState::pgn(const std::string &white, const std::string &black, const std::string &result_override, const std::string &termination_override) const {
	const auto effective_result = result_override.empty() ? result() : result_override;
	const auto effective_termination = termination_override.empty() ? termination() : termination_override;
	std::ostringstream output;
	output << "[Event \"Gadidae\"]\n"
	       << "[Site \"?\"]\n"
	       << "[Date \"????.??.??\"]\n"
	       << "[Round \"?\"]\n"
	       << "[White \"" << escape_pgn(white) << "\"]\n"
	       << "[Black \"" << escape_pgn(black) << "\"]\n"
	       << "[Result \"" << effective_result << "\"]\n";
	if (start_fen_ != chess::constants::STARTPOS) {
		output << "[SetUp \"1\"]\n[FEN \"" << escape_pgn(start_fen_) << "\"]\n";
	}
	if (!effective_termination.empty()) {
		output << "[Termination \"" << escape_pgn(effective_termination) << "\"]\n";
	}
	output << '\n' << movetext(effective_result) << '\n';
	return output.str();
}

void GameState::import_pgn(const std::string &document) {
	std::string fen = "startpos";
	std::istringstream lines(document);
	std::string movetext_source;
	for (std::string line; std::getline(lines, line);) {
		if (line.rfind("[FEN \"", 0) == 0) {
			const auto end = line.rfind("\"]");
			if (end != std::string::npos && end > 6) {
				fen = line.substr(6, end - 6);
			}
		} else if (line.empty() || line.front() != '[') {
			movetext_source += line + '\n';
		}
	}
	reset(fen);

	std::string clean;
	int variation_depth = 0;
	bool brace_comment = false;
	bool line_comment = false;
	for (const char character : movetext_source) {
		if (line_comment) {
			if (character == '\n') {
				line_comment = false;
				clean.push_back(' ');
			}
			continue;
		}
		if (brace_comment) {
			if (character == '}') {
				brace_comment = false;
				clean.push_back(' ');
			}
			continue;
		}
		if (character == ';') {
			line_comment = true;
		} else if (character == '{') {
			brace_comment = true;
		} else if (character == '(') {
			++variation_depth;
		} else if (character == ')' && variation_depth > 0) {
			--variation_depth;
		} else if (variation_depth == 0) {
			clean.push_back(character);
		}
	}

	std::istringstream tokens(clean);
	for (std::string token; tokens >> token;) {
		if (token[0] == '$' || token == "1-0" || token == "0-1" || token == "1/2-1/2" || token == "*") {
			continue;
		}
		const auto dot = token.find_last_of('.');
		if (dot != std::string::npos) {
			token = token.substr(dot + 1);
			if (token.empty()) {
				continue;
			}
		}
		while (!token.empty() && (token.back() == '!' || token.back() == '?')) {
			token.pop_back();
		}
		if (!token.empty()) {
			make_move(chess::uci::parseSan(board_, token));
		}
	}
}

} // namespace gadidae::graphics
