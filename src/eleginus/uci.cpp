// Minimal UCI adapter so Eleginus remains usable by the shared Graphics client.

#include "eleginus/runtime.hpp"
#include "eleginus/search.hpp"

#include <cmath>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

void apply_position(chess::Board &board, const std::string &command) {
	std::istringstream input(command);
	std::string token;
	input >> token; // position
	input >> token;
	if (token == "startpos") {
		board = chess::Board();
		input >> token;
	} else if (token == "fen") {
		std::vector<std::string> fields;
		for (int index = 0; index < 6 && input >> token; ++index) {
			fields.push_back(token);
		}
		if (fields.size() != 6) {
			throw std::invalid_argument("position fen requires six fields");
		}
		std::string fen;
		for (std::size_t index = 0; index < fields.size(); ++index) {
			if (index != 0)
				fen += ' ';
			fen += fields[index];
		}
		board = chess::Board(fen);
		input >> token;
	}
	if (token == "moves") {
		while (input >> token) {
			board.makeMove(chess::uci::uciToMove(board, token));
		}
	}
}

} // namespace

int main(int argc, char **argv) {
	try {
		int expansions = 32;
		for (int index = 1; index < argc; ++index) {
			const std::string argument = argv[index];
			if (index + 1 >= argc)
				throw std::invalid_argument("missing value for " + argument);
			const std::string value = argv[++index];
			if (argument == "--expansions")
				expansions = std::stoi(value);
			else
				throw std::invalid_argument("unknown argument: " + argument);
		}
		eleginus::CpuValue value(eleginus::load_embedded_runtime_model());
		chess::Board board;
		std::string line;
		while (std::getline(std::cin, line)) {
			if (line == "uci") {
				std::cout << "id name Eleginus\nid author Gadidae\n"
						  << "option name BFMExpansions type spin default 32 min 1 max 1000000\n"
						  << "uciok\n" << std::flush;
			} else if (line == "isready") {
				std::cout << "readyok\n" << std::flush;
			} else if (line.starts_with("position ")) {
				apply_position(board, line);
			} else if (line.starts_with("setoption name BFMExpansions value ")) {
				constexpr std::string_view prefix = "setoption name BFMExpansions value ";
				expansions = std::stoi(line.substr(prefix.size()));
			} else if (line.starts_with("go")) {
				eleginus::SearchOptions options;
				options.expansions = expansions;
				const auto result = eleginus::Searcher(value, options).search(board);
				const int cp = static_cast<int>(std::lround((result.value - 0.5F) * 2000.0F));
				std::cout << "info depth " << result.expanded_nodes << " nodes "
						  << result.evaluated_nodes << " score cp " << cp << " pv "
						  << eleginus::move_uci(result.move) << '\n';
				std::cout << "bestmove " << eleginus::move_uci(result.move) << '\n' << std::flush;
			} else if (line == "ucinewgame") {
				board = chess::Board();
			} else if (line == "quit") {
				break;
			}
		}
		return 0;
	} catch (const std::exception &error) {
		std::cerr << "Eleginus UCI failed: " << error.what() << '\n';
		return 1;
	}
}
