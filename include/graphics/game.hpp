// Owns one legal chess game, its reversible move history, and PGN/FEN views.
#pragma once
#include <chess.hpp>
#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace gadidae::graphics {

class GameState {
public:
	GameState();

	/// Replaces the game with a validated start position and clears history.
	void reset(const std::string &fen);

	/// Returns the current board.
	const chess::Board &board() const;
	chess::Board &board();

	/// Returns the initial FEN from which PGN numbering and replay begin.
	const std::string &start_fen() const;

	/// Generates every legal move or only moves from one selected square.
	std::vector<chess::Move> legal_moves() const;
	std::vector<chess::Move> legal_moves_from(int square) const;

	/// Applies one legal chess move represented as a move object or UCI string.
	void make_move(const chess::Move &move);
	void make_uci(const std::string &uci);

	/// Removes the latest move while preserving the chess library's history state.
	bool undo();

	/// Reports and formats rules-defined terminal state.
	bool over() const;
	std::string result() const;
	std::string termination() const;

	/// Returns move count, last move, and one non-mutating historical snapshot.
	std::size_t plies() const;
	std::optional<chess::Move> last_move() const;
	GameState position_at(std::size_t ply) const;

	/// Formats movetext or a complete PGN document.
	std::string movetext(const std::string &result_override = "") const;
	std::string pgn(const std::string &white = "White",
					const std::string &black = "Black",
					const std::string &result_override = "",
					const std::string &termination_override = "") const;

	/// Imports the main line of a PGN document.
	void import_pgn(const std::string &document);

private:
	chess::Board board_;
	std::string start_fen_;
	std::vector<chess::Move> moves_;
	std::vector<std::string> san_;
};

} // namespace gadidae::graphics
