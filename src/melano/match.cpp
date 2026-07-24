// Implements batched, paired Melano arena games and result-only promotion statistics.

#include "melano/arena.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>
#include "melano/checkpoint.hpp"

namespace melano {

namespace {

struct GameRecord {
	int index = 0;
	bool candidate_white = true;
	std::string start_fen;
	std::string result = "1/2-1/2";
	std::string termination = "max plies";
	int plies = 0;
	std::vector<std::string> san_moves;
};

struct ActiveGame {
	GameRecord record;
	chess::Board board;
	bool finished = false;
};

// Convert a PGN result to candidate points after accounting for assigned color.
double candidate_score(const GameRecord &record) {
	if (record.result == "1/2-1/2") {
		return 0.5;
	}
	const bool white_won = record.result == "1-0";
	return white_won == record.candidate_white ? 1.0 : 0.0;
}

// Freeze one active game into its immutable result record.
void finish_game(ActiveGame &game, const std::string &result, const std::string &termination) {
	game.finished = true;
	game.record.result = result;
	game.record.termination = termination;
	game.record.plies = static_cast<int>(game.record.san_moves.size());
}

// Apply exact terminal rules or the configured max-ply draw boundary.
void update_finished(ActiveGame &game, int max_plies) {
	if (game.finished) {
		return;
	}
	if (game_is_over(game.board)) {
		finish_game(game, game_result(game.board), game_termination(game.board));
	} else if (static_cast<int>(game.record.san_moves.size()) >= max_plies) {
		finish_game(game, "1/2-1/2", "max plies");
	}
}

// Advance all active games by batching positions according to which model must move.
void play_turns(std::vector<ActiveGame> &games, Searcher &candidate, Searcher &baseline,
				int max_plies) {
	while (true) {
		for (auto &game : games) {
			update_finished(game, max_plies);
		}
		std::vector<std::size_t> candidate_indices;
		std::vector<std::size_t> baseline_indices;
		std::vector<chess::Board> candidate_boards;
		std::vector<chess::Board> baseline_boards;
		for (std::size_t index = 0; index < games.size(); ++index) {
			const auto &game = games[index];
			if (game.finished) {
				continue;
			}
			const bool white_turn = game.board.sideToMove() == chess::Color::WHITE;
			const bool candidate_turn = white_turn == game.record.candidate_white;
			if (candidate_turn) {
				candidate_indices.push_back(index);
				candidate_boards.push_back(game.board);
			} else {
				baseline_indices.push_back(index);
				baseline_boards.push_back(game.board);
			}
		}
		if (candidate_indices.empty() && baseline_indices.empty()) {
			break;
		}

		// Commit a model batch while preserving each result's originating game index.
		auto apply_results = [&](const std::vector<std::size_t> &indices,
								 const std::vector<SearchResult> &results) {
			for (std::size_t row = 0; row < indices.size(); ++row) {
				auto &game = games[indices[row]];
				const auto move = results[row].move;
				game.record.san_moves.push_back(move_san(game.board, move));
				game.board.makeMove(move);
				update_finished(game, max_plies);
			}
		};
		if (!candidate_boards.empty()) {
			apply_results(candidate_indices, candidate.search_many(candidate_boards));
		}
		if (!baseline_boards.empty()) {
			apply_results(baseline_indices, baseline.search_many(baseline_boards));
		}
	}
}

// Format SAN movetext with PGN move numbers and conventional 80-column wrapping.
std::string wrap_moves(const std::vector<std::string> &moves) {
	std::ostringstream output;
	int column = 0;
	for (std::size_t ply = 0; ply < moves.size(); ++ply) {
		std::ostringstream token;
		if (ply % 2 == 0) {
			token << ply / 2 + 1 << ". ";
		}
		token << moves[ply] << ' ';
		const auto text = token.str();
		if (column > 0 && column + static_cast<int>(text.size()) > 80) {
			output << '\n';
			column = 0;
		}
		output << text;
		column += static_cast<int>(text.size());
	}
	return output.str();
}

// Write all arena records, colors, starts, terminations, and SAN moves as PGN.
void write_pgn(const std::filesystem::path &path, const std::vector<GameRecord> &records,
			   const std::string &candidate, const std::string &baseline) {
	if (path.empty()) {
		return;
	}
	if (!path.parent_path().empty()) {
		std::filesystem::create_directories(path.parent_path());
	}
	std::ofstream output(path);
	if (!output) {
		throw std::runtime_error("cannot write PGN: " + path.string());
	}
	for (const auto &record : records) {
		const std::string white = record.candidate_white ? candidate : baseline;
		const std::string black = record.candidate_white ? baseline : candidate;
		output << "[Event \"Melano Arena\"]\n"
			   << "[Site \"?\"]\n"
			   << "[Round \"" << record.index + 1 << "\"]\n"
			   << "[White \"" << white << "\"]\n"
			   << "[Black \"" << black << "\"]\n"
			   << "[Result \"" << record.result << "\"]\n"
			   << "[Termination \"" << record.termination << "\"]\n";
		if (record.start_fen != chess::constants::STARTPOS) {
			output << "[SetUp \"1\"]\n[FEN \"" << record.start_fen << "\"]\n";
		}
		output << '\n' << wrap_moves(record.san_moves) << record.result << "\n\n";
	}
}

// Convert score fraction to logistic Elo difference, clamped away from infinities.
double elo_from_score(double score) {
	const double bounded = std::clamp(score, 1e-6, 1.0 - 1e-6);
	return 400.0 * std::log10(bounded / (1.0 - bounded));
}

} // namespace

// Load both models once, play paired batches, and evaluate the net-win acceptance rule.
nlohmann::json evaluate_models(const ArenaOptions &options) {
	if (options.candidate.empty() || options.baseline.empty()) {
		throw std::invalid_argument("arena requires candidate and baseline models");
	}
	if (options.games <= 0 || options.games % 2 != 0 || options.games_in_flight <= 0 ||
		options.max_plies <= 0) {
		throw std::invalid_argument(
			"arena games must be positive and even; games in flight and max plies must be positive");
	}
	const auto device = resolve_device(options.device);
	const auto specs = make_arena_specs(options.games, options.opening_book, options.book_plies,
										options.max_book_positions, options.seed);
	std::cout << "arena: start candidate=" << options.candidate.string()
			  << " candidate_sha256=" << file_sha256(options.candidate)
			  << " baseline=" << options.baseline.string()
			  << " baseline_sha256=" << file_sha256(options.baseline) << " games=" << options.games
			  << " sims=" << options.search.mcts_sims
			  << " search_type=" << search_type_name(options.search.type)
			  << " device=" << device.str() << " games_in_flight=" << options.games_in_flight
			  << std::endl;

	auto candidate_model = load_checkpoint(options.candidate, device);
	auto baseline_model = load_checkpoint(options.baseline, device);
	Searcher candidate_searcher(candidate_model, device, options.search);
	Searcher baseline_searcher(baseline_model, device, options.search);
	std::cout << "arena: models ready candidate_arch=" << kArchType
			  << " baseline_arch=" << kArchType << std::endl;

	std::vector<GameRecord> records;
	records.reserve(specs.size());
	for (std::size_t begin = 0; begin < specs.size(); begin += options.games_in_flight) {
		const auto end = std::min(specs.size(), begin + options.games_in_flight);
		std::vector<ActiveGame> active;
		active.reserve(end - begin);
		for (std::size_t index = begin; index < end; ++index) {
			ActiveGame game;
			game.record.index = static_cast<int>(index);
			game.record.candidate_white = specs[index].candidate_white;
			game.record.start_fen = specs[index].fen;
			game.board = chess::Board(specs[index].fen);
			active.push_back(std::move(game));
		}
		play_turns(active, candidate_searcher, baseline_searcher, options.max_plies);
		for (auto &game : active) {
			const double score = candidate_score(game.record);
			records.push_back(std::move(game.record));
			const auto &record = records.back();
			if (options.log_every > 0 &&
				((record.index + 1) % options.log_every == 0 || record.index == 0)) {
				std::cout << "arena game " << record.index + 1 << '/' << options.games
						  << ": candidate_color=" << (record.candidate_white ? "white" : "black")
						  << " result=" << record.result << " candidate_score=" << score
						  << " plies=" << record.plies << std::endl;
			}
		}
	}

	int wins = 0;
	int draws = 0;
	int losses = 0;
	std::vector<double> scores;
	for (const auto &record : records) {
		const double score = candidate_score(record);
		scores.push_back(score);
		wins += score == 1.0;
		draws += score == 0.5;
		losses += score == 0.0;
	}
	const double score = std::accumulate(scores.begin(), scores.end(), 0.0) /
						 std::max<std::size_t>(1, scores.size());
	double variance = 0.0;
	for (const double item : scores) {
		variance += (item - score) * (item - score);
	}
	variance /= std::max<std::size_t>(1, scores.size());
	const double margin = 1.96 * std::sqrt(variance / std::max<std::size_t>(1, scores.size()));
	const double low = std::clamp(score - margin, 0.0, 1.0);
	const double high = std::clamp(score + margin, 0.0, 1.0);
	const int net_wins = wins - losses;
	const bool accepted = net_wins >= options.min_net_wins;

	nlohmann::json summary = {
		{"candidate", options.candidate.string()},
		{"baseline", options.baseline.string()},
		{"games", options.games},
		{"wins", wins},
		{"draws", draws},
		{"losses", losses},
		{"net_wins", net_wins},
		{"score", score},
		{"score_ci_low", low},
		{"score_ci_high", high},
		{"elo_diff", elo_from_score(score)},
		{"elo_ci_low", elo_from_score(low)},
		{"elo_ci_high", elo_from_score(high)},
		{"search_type", search_type_name(options.search.type)},
		{"sims_soft_cap", options.search.mcts_sims},
		{"mcts_batch_size", options.search.mcts_batch_size},
		{"movetime_ms", options.search.movetime_ms},
		{"paired_openings", true},
		{"unique_start_positions", (options.games + 1) / 2},
		{"result_ok", accepted},
		{"accepted", accepted},
		{"min_net_wins", options.min_net_wins},
	};
	write_pgn(options.pgn_output, records, options.candidate.filename().string(),
			  options.baseline.filename().string());
	std::ostringstream line;
	line << "arena: finished wins=" << wins << " draws=" << draws << " losses=" << losses
		 << " net_wins=" << net_wins << " score=" << std::fixed << std::setprecision(3) << score
		 << " elo_diff=" << std::showpos << std::setprecision(1)
		 << summary["elo_diff"].get<double>() << std::noshowpos;
	std::cout << line.str() << std::endl;
	return summary;
}

} // namespace melano
