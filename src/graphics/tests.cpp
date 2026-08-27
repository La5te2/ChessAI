// CTest entrypoint for the graphics game-state boundary. It verifies that UCI requests preserve
// complete legal move history, historical positions and repetition state.
#include "graphics/game.hpp"
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void require(bool condition, const std::string &message) {
	if (!condition) {
		throw std::runtime_error(message);
	}
}

} // namespace

int main() {
	try {
		gadidae::graphics::GameState game;
		const std::string initial = std::string("fen ") + chess::constants::STARTPOS;
		require(game.uci_position() == initial, "initial UCI position mismatch");

		for (const char *uci : {"g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1", "f6g8"}) {
			game.make_uci(uci);
		}
		const std::string first_cycle = initial + " moves g1f3 g8f6 f3g1 f6g8";
		require(game.position_at(4).uci_position() == first_cycle, "historical UCI position included the wrong move range");
		require(game.uci_position() == first_cycle + " g1f3 g8f6 f3g1 f6g8", "live UCI position omitted move history");
		require(game.board().isRepetition(2), "preserved move history did not reach threefold repetition");

		std::cout << "graphicstests passed" << std::endl;
		return 0;
	} catch (const std::exception &error) {
		std::cerr << "graphicstests failed: " << error.what() << std::endl;
		return 1;
	}
}
