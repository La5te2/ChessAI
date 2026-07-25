// Implements the native OpenGL Gadidae interface. Simulator and Stadium share
// one renderer and chess state while communicating with engines only through UCI.
#include "graphics/application.hpp"
#include "graphics/uci.hpp"
#include <glad/gl.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <chess.hpp>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <misc/cpp/imgui_stdlib.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <commdlg.h>
#endif

namespace gadidae::graphics {
namespace {

using Clock = std::chrono::steady_clock;

/// Reads one environment variable without relying on deprecated Windows CRT APIs.
std::optional<std::string> environment_value(const char *name) {
#ifdef _WIN32
	char *buffer = nullptr;
	std::size_t length = 0;
	if(_dupenv_s(&buffer, &length, name) != 0 || buffer == nullptr) {
		return std::nullopt;
	}
	const std::string value(buffer);
	std::free(buffer);
	return value;
#else
	if(const char *value = std::getenv(name)) {
		return std::string(value);
	}
	return std::nullopt;
#endif
}

constexpr ImU32 color32(float red, float green, float blue, float alpha = 1.0F) {
	return IM_COL32(static_cast<int>(red * 255.0F),
					static_cast<int>(green * 255.0F),
					static_cast<int>(blue * 255.0F),
					static_cast<int>(alpha * 255.0F));
}

/// Converts a packed ImGui color to a normalized editable RGBA vector.
ImVec4 color_vector(ImU32 color) {
	return {
		static_cast<float>((color >> IM_COL32_R_SHIFT) & 0xFF) / 255.0F,
		static_cast<float>((color >> IM_COL32_G_SHIFT) & 0xFF) / 255.0F,
		static_cast<float>((color >> IM_COL32_B_SHIFT) & 0xFF) / 255.0F,
		static_cast<float>((color >> IM_COL32_A_SHIFT) & 0xFF) / 255.0F,
	};
}

/// Converts an editable RGBA vector into the packed format used by ImDrawList.
ImU32 packed_color(const ImVec4 &color) {
	return ImGui::ColorConvertFloat4ToU32(color);
}

/// Finds a readable system UI font without making it a runtime requirement.
std::optional<std::filesystem::path> system_font_path() {
#ifdef _WIN32
	const std::array candidates = {
		std::filesystem::path("C:/Windows/Fonts/seguisb.ttf"),
		std::filesystem::path("C:/Windows/Fonts/segoeui.ttf"),
	};
#else
	const std::array candidates = {
		std::filesystem::path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
		std::filesystem::path("/usr/share/fonts/TTF/DejaVuSans.ttf"),
	};
#endif
	for(const auto &candidate : candidates) {
		if(std::filesystem::exists(candidate)) {
			return candidate;
		}
	}
	return std::nullopt;
}

/// Returns a writable per-user settings path on Windows and Linux.
std::filesystem::path settings_path() {
#ifdef _WIN32
	if(const auto appdata = environment_value("APPDATA")) {
		return std::filesystem::path(*appdata) / "Gadidae" / "gui.json";
	}
#else
	if(const auto xdg = environment_value("XDG_CONFIG_HOME")) {
		return std::filesystem::path(*xdg) / "Gadidae" / "gui.json";
	}
	if(const auto home = environment_value("HOME")) {
		return std::filesystem::path(*home) / ".config" / "Gadidae" / "gui.json";
	}
#endif
	return std::filesystem::current_path() / "gadidae-gui.json";
}

/// Escapes a PGN tag value according to the PGN string syntax.
std::string escape_pgn(std::string value) {
	std::string escaped;
	for(const char ch : value) {
		if(ch == '\\' || ch == '"') {
			escaped.push_back('\\');
		}
		escaped.push_back(ch);
	}
	return escaped;
}

/// Wraps PGN movetext at a stable column width without splitting tokens.
std::string wrap_movetext(const std::string &text, std::size_t width = 80) {
	std::istringstream input(text);
	std::ostringstream output;
	std::size_t column = 0;
	for(std::string token; input >> token;) {
		const std::size_t required = token.size() + (column == 0 ? 0 : 1);
		if(column > 0 && column + required > width) {
			output << '\n';
			column = 0;
		}
		if(column > 0) {
			output << ' ';
			++column;
		}
		output << token;
		column += token.size();
	}
	return output.str();
}

/// Owns a legal chess position and the SAN/UCI history needed by both GUI modes.
class GameState {
public:
	GameState() {
		reset("startpos");
	}

	/// Replaces the game with a validated start position and clears history.
	void reset(const std::string &fen) {
		start_fen_ = fen.empty() || fen == "startpos"
						 ? std::string(chess::constants::STARTPOS)
						 : fen;
		board_ = chess::Board(start_fen_);
		moves_.clear();
		san_.clear();
	}

	/// Returns the current board.
	const chess::Board &board() const {
		return board_;
	}

	/// Returns a mutable board for protocol conversion helpers.
	chess::Board &board() {
		return board_;
	}

	/// Returns the initial FEN from which PGN numbering and replay begin.
	const std::string &start_fen() const {
		return start_fen_;
	}

	/// Generates all legal moves from the current position.
	std::vector<chess::Move> legal_moves() const {
		chess::Movelist moves;
		chess::movegen::legalmoves(moves, board_);
		return {moves.begin(), moves.end()};
	}

	/// Generates legal moves starting on one selected square.
	std::vector<chess::Move> legal_moves_from(int square) const {
		std::vector<chess::Move> result;
		for(const auto &move : legal_moves()) {
			if(move.from().index() == square) {
				result.push_back(move);
			}
		}
		return result;
	}

	/// Applies a legal move and records its SAN before mutating the board.
	void make_move(const chess::Move &move) {
		const auto legal = legal_moves();
		if(std::find(legal.begin(), legal.end(), move) == legal.end()) {
			throw std::invalid_argument("illegal move");
		}
		san_.push_back(chess::uci::moveToSan(board_, move));
		moves_.push_back(move);
		board_.makeMove(move);
	}

	/// Converts and applies a UCI move in the current position.
	void make_uci(const std::string &uci) {
		make_move(chess::uci::uciToMove(board_, uci));
	}

	/// Removes the most recent move while preserving chess-library repetition state.
	bool undo() {
		if(moves_.empty()) {
			return false;
		}
		board_.unmakeMove(moves_.back());
		moves_.pop_back();
		san_.pop_back();
		return true;
	}

	/// Reports whether a rules-defined terminal condition has been reached.
	bool over() const {
		return board_.isGameOver().first != chess::GameResultReason::NONE;
	}

	/// Formats a PGN result from the absolute winner.
	std::string result() const {
		const auto outcome = board_.isGameOver();
		if(outcome.first == chess::GameResultReason::NONE) {
			return "*";
		}
		if(outcome.second == chess::GameResult::DRAW) {
			return "1/2-1/2";
		}
		const bool side_wins = outcome.second == chess::GameResult::WIN;
		const bool white_wins =
			side_wins == (board_.sideToMove() == chess::Color::WHITE);
		return white_wins ? "1-0" : "0-1";
	}

	/// Converts the rules-defined terminal reason into a PGN-friendly label.
	std::string termination() const {
		switch(board_.isGameOver().first) {
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

	/// Returns the number of played half-moves.
	std::size_t plies() const {
		return moves_.size();
	}

	/// Returns the last move for board highlighting.
	std::optional<chess::Move> last_move() const {
		if(moves_.empty()) {
			return std::nullopt;
		}
		return moves_.back();
	}

	/// Formats compact movetext without PGN headers.
	std::string movetext() const {
		std::ostringstream text;
		for(std::size_t index = 0; index < san_.size(); ++index) {
			if(index % 2 == 0) {
				if(index > 0) {
					text << ' ';
				}
				text << (index / 2 + 1) << ". ";
			} else {
				text << ' ';
			}
			text << san_[index];
		}
		if(!san_.empty()) {
			text << ' ';
		}
		text << result();
		return wrap_movetext(text.str());
	}

	/// Creates a complete, wrapped PGN document with optional engine names.
	std::string pgn(const std::string &white = "White",
					const std::string &black = "Black") const {
		std::ostringstream output;
		output << "[Event \"Gadidae\"]\n"
			   << "[Site \"?\"]\n"
			   << "[Date \"????.??.??\"]\n"
			   << "[Round \"?\"]\n"
			   << "[White \"" << escape_pgn(white) << "\"]\n"
			   << "[Black \"" << escape_pgn(black) << "\"]\n"
			   << "[Result \"" << result() << "\"]\n";
		if(start_fen_ != chess::constants::STARTPOS) {
			output << "[SetUp \"1\"]\n[FEN \"" << escape_pgn(start_fen_) << "\"]\n";
		}
		if(over()) {
			output << "[Termination \"" << escape_pgn(termination()) << "\"]\n";
		}
		output << '\n' << movetext() << '\n';
		return output.str();
	}

	/// Imports the main line of a PGN document and ignores comments and variations.
	void import_pgn(const std::string &document) {
		std::string fen = "startpos";
		std::istringstream lines(document);
		std::string movetext_source;
		for(std::string line; std::getline(lines, line);) {
			if(line.rfind("[FEN \"", 0) == 0) {
				const auto end = line.rfind("\"]");
				if(end != std::string::npos && end > 6) {
					fen = line.substr(6, end - 6);
				}
			} else if(line.empty() || line.front() != '[') {
				movetext_source += line + '\n';
			}
		}
		reset(fen);

		std::string clean;
		int variation_depth = 0;
		bool brace_comment = false;
		bool line_comment = false;
		for(const char ch : movetext_source) {
			if(line_comment) {
				if(ch == '\n') {
					line_comment = false;
					clean.push_back(' ');
				}
				continue;
			}
			if(brace_comment) {
				if(ch == '}') {
					brace_comment = false;
					clean.push_back(' ');
				}
				continue;
			}
			if(ch == ';') {
				line_comment = true;
			} else if(ch == '{') {
				brace_comment = true;
			} else if(ch == '(') {
				++variation_depth;
			} else if(ch == ')' && variation_depth > 0) {
				--variation_depth;
			} else if(variation_depth == 0) {
				clean.push_back(ch);
			}
		}

		std::istringstream tokens(clean);
		for(std::string token; tokens >> token;) {
			if(token[0] == '$' || token == "1-0" || token == "0-1" ||
			   token == "1/2-1/2" || token == "*") {
				continue;
			}
			const auto dot = token.find_last_of('.');
			if(dot != std::string::npos) {
				token = token.substr(dot + 1);
				if(token.empty()) {
					continue;
				}
			}
			while(!token.empty() && (token.back() == '!' || token.back() == '?')) {
				token.pop_back();
			}
			if(token.empty()) {
				continue;
			}
			make_move(chess::uci::parseSan(board_, token));
		}
	}

private:
	chess::Board board_;
	std::string start_fen_;
	std::vector<chess::Move> moves_;
	std::vector<std::string> san_;
};

/// Mutable board colors and presentation preferences stored with the GUI config.
struct Appearance {
	ImVec4 light = color_vector(color32(0.91F, 0.93F, 0.91F));
	ImVec4 dark = color_vector(color32(0.29F, 0.45F, 0.39F));
	ImVec4 selected = color_vector(color32(0.93F, 0.73F, 0.25F));
	ImVec4 last_move = color_vector(color32(0.66F, 0.75F, 0.36F));
	ImVec4 legal = color_vector(color32(0.20F, 0.24F, 0.23F, 0.34F));
	bool coordinates = true;
};

/// Selects application chrome independently from chessboard colors.
enum class Theme { Dark, Light };

/// Converts one engine configuration to its persisted JSON representation.
nlohmann::json engine_json(const EngineConfig &config) {
	return {
		{"path", config.path.string()},
		{"name", config.name},
		{"device", config.device},
		{"arguments", config.arguments},
		{"options", config.options},
		{"movetime_ms", config.movetime_ms},
		{"node_limit", config.node_limit},
		{"multipv", config.multipv},
		{"progress_interval_ms", config.progress_interval_ms},
	};
}

/// Applies recognized JSON values without requiring every setting to be present.
void read_engine_json(const nlohmann::json &json, EngineConfig &config) {
	if(json.contains("path")) {
		config.path = json.value("path", "");
	}
	config.name = json.value("name", config.name);
	config.device = json.value("device", config.device);
	config.arguments = json.value("arguments", config.arguments);
	config.options = json.value("options", config.options);
	config.movetime_ms = json.value("movetime_ms", config.movetime_ms);
	config.node_limit = json.value("node_limit", config.node_limit);
	config.multipv = json.value("multipv", config.multipv);
	config.progress_interval_ms =
		json.value("progress_interval_ms", config.progress_interval_ms);
}

/// Produces a side-to-move UCI score string for one analysis row.
std::string score_text(const AnalysisLine &line) {
	if(line.mate) {
		return line.score == 0 ? "#" : "#" + std::to_string(line.score);
	}
	std::ostringstream score;
	score << std::showpos << std::fixed << std::setprecision(2)
		  << static_cast<double>(line.score) / 100.0;
	return score.str();
}

/// Converts a UCI principal variation to SAN from a supplied root position.
std::string pv_san(const std::string &fen, const std::vector<std::string> &pv,
				   std::size_t maximum = 10) {
	try {
		chess::Board board(fen);
		std::ostringstream text;
		for(std::size_t index = 0; index < pv.size() && index < maximum; ++index) {
			const auto move = chess::uci::uciToMove(board, pv[index]);
			chess::Movelist legal;
			chess::movegen::legalmoves(legal, board);
			if(std::find(legal.begin(), legal.end(), move) == legal.end()) {
				throw std::invalid_argument("PV contains an illegal move");
			}
			if(index > 0) {
				text << ' ';
			}
			text << chess::uci::moveToSan(board, move);
			board.makeMove(move);
		}
		return text.str();
	} catch(...) {
		std::ostringstream text;
		for(std::size_t index = 0; index < pv.size() && index < maximum; ++index) {
			if(index > 0) {
				text << ' ';
			}
			text << pv[index];
		}
		return text.str();
	}
}

/// Opens a native Windows file dialog and returns an empty value on cancellation.
std::optional<std::filesystem::path> file_dialog(bool save,
												 const wchar_t *filter,
												 const wchar_t *extension) {
#ifdef _WIN32
	std::array<wchar_t, 32768> path{};
	OPENFILENAMEW dialog{};
	dialog.lStructSize = sizeof(dialog);
	dialog.lpstrFile = path.data();
	dialog.nMaxFile = static_cast<DWORD>(path.size());
	dialog.lpstrFilter = filter;
	dialog.lpstrDefExt = extension;
	dialog.Flags = OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR |
				   (save ? OFN_OVERWRITEPROMPT : OFN_FILEMUSTEXIST);
	const BOOL selected =
		save ? GetSaveFileNameW(&dialog) : GetOpenFileNameW(&dialog);
	if(selected) {
		return std::filesystem::path(path.data());
	}
#else
	static_cast<void>(save);
	static_cast<void>(filter);
	static_cast<void>(extension);
#endif
	return std::nullopt;
}

/// Draws a filled outlined polygon used by the procedural piece set.
void polygon(ImDrawList *draw, const std::vector<ImVec2> &points,
			 ImU32 fill, ImU32 outline, float thickness) {
	draw->AddConvexPolyFilled(points.data(), static_cast<int>(points.size()), fill);
	draw->AddPolyline(points.data(), static_cast<int>(points.size()), outline,
					  ImDrawFlags_Closed, thickness);
}

/// Draws one crisp vector chess piece from simple geometry at any board scale.
void draw_piece(ImDrawList *draw, const chess::Piece &piece,
				ImVec2 center, float size) {
	const bool white = piece.color() == chess::Color::WHITE;
	const ImU32 fill = white ? color32(0.96F, 0.97F, 0.96F)
							 : color32(0.12F, 0.15F, 0.16F);
	const ImU32 outline = white ? color32(0.15F, 0.19F, 0.19F)
								: color32(0.90F, 0.92F, 0.90F);
	const float stroke = std::max(1.2F, size * 0.026F);
	const float x = center.x;
	const float y = center.y;
	const float radius = size * 0.105F;
	const auto base = [&](float width, float top, float bottom) {
		const std::vector<ImVec2> points = {
			{x - width * 0.38F, y + top},
			{x + width * 0.38F, y + top},
			{x + width * 0.50F, y + bottom},
			{x - width * 0.50F, y + bottom},
		};
		polygon(draw, points, fill, outline, stroke);
	};

	switch(piece.type().internal()) {
	case chess::PieceType::underlying::PAWN:
		draw->AddCircleFilled({x, y - size * 0.19F}, radius, fill, 32);
		draw->AddCircle({x, y - size * 0.19F}, radius, outline, 32, stroke);
		polygon(draw,
				{{x - size * 0.10F, y - size * 0.09F},
				 {x + size * 0.10F, y - size * 0.09F},
				 {x + size * 0.18F, y + size * 0.23F},
				 {x - size * 0.18F, y + size * 0.23F}},
				fill, outline, stroke);
		base(size * 0.52F, size * 0.20F, size * 0.30F);
		break;
	case chess::PieceType::underlying::ROOK:
		polygon(draw,
				{{x - size * 0.24F, y - size * 0.28F},
				 {x - size * 0.24F, y - size * 0.13F},
				 {x - size * 0.17F, y - size * 0.13F},
				 {x - size * 0.17F, y + size * 0.21F},
				 {x + size * 0.17F, y + size * 0.21F},
				 {x + size * 0.17F, y - size * 0.13F},
				 {x + size * 0.24F, y - size * 0.13F},
				 {x + size * 0.24F, y - size * 0.28F},
				 {x + size * 0.10F, y - size * 0.28F},
				 {x + size * 0.10F, y - size * 0.20F},
				 {x - size * 0.02F, y - size * 0.20F},
				 {x - size * 0.02F, y - size * 0.28F}},
				fill, outline, stroke);
		base(size * 0.62F, size * 0.19F, size * 0.30F);
		break;
	case chess::PieceType::underlying::KNIGHT:
		polygon(draw,
				{{x - size * 0.23F, y + size * 0.22F},
				 {x - size * 0.17F, y - size * 0.06F},
				 {x - size * 0.02F, y - size * 0.27F},
				 {x + size * 0.17F, y - size * 0.16F},
				 {x + size * 0.23F, y + size * 0.08F},
				 {x + size * 0.11F, y + size * 0.22F}},
				fill, outline, stroke);
		draw->AddCircleFilled({x + size * 0.075F, y - size * 0.13F},
							 size * 0.025F, outline, 12);
		base(size * 0.63F, size * 0.20F, size * 0.30F);
		break;
	case chess::PieceType::underlying::BISHOP:
		polygon(draw,
				{{x, y - size * 0.31F},
				 {x + size * 0.14F, y - size * 0.11F},
				 {x + size * 0.08F, y + size * 0.03F},
				 {x + size * 0.16F, y + size * 0.22F},
				 {x - size * 0.16F, y + size * 0.22F},
				 {x - size * 0.08F, y + size * 0.03F},
				 {x - size * 0.14F, y - size * 0.11F}},
				fill, outline, stroke);
		draw->AddLine({x - size * 0.02F, y - size * 0.24F},
					  {x + size * 0.07F, y - size * 0.10F}, outline, stroke);
		base(size * 0.61F, size * 0.20F, size * 0.30F);
		break;
	case chess::PieceType::underlying::QUEEN:
		polygon(draw,
				{{x - size * 0.25F, y - size * 0.18F},
				 {x - size * 0.13F, y - size * 0.29F},
				 {x, y - size * 0.16F},
				 {x + size * 0.13F, y - size * 0.29F},
				 {x + size * 0.25F, y - size * 0.18F},
				 {x + size * 0.15F, y + size * 0.21F},
				 {x - size * 0.15F, y + size * 0.21F}},
				fill, outline, stroke);
		for(const float offset : {-0.13F, 0.0F, 0.13F}) {
			draw->AddCircleFilled({x + offset * size, y - size * 0.29F},
								 size * 0.035F, fill, 16);
			draw->AddCircle({x + offset * size, y - size * 0.29F},
							size * 0.035F, outline, 16, stroke);
		}
		base(size * 0.66F, size * 0.19F, size * 0.30F);
		break;
	case chess::PieceType::underlying::KING:
		draw->AddLine({x, y - size * 0.36F}, {x, y - size * 0.17F},
					  outline, stroke * 1.3F);
		draw->AddLine({x - size * 0.08F, y - size * 0.28F},
					  {x + size * 0.08F, y - size * 0.28F}, outline,
					  stroke * 1.3F);
		polygon(draw,
				{{x, y - size * 0.19F},
				 {x + size * 0.16F, y - size * 0.04F},
				 {x + size * 0.11F, y + size * 0.21F},
				 {x - size * 0.11F, y + size * 0.21F},
				 {x - size * 0.16F, y - size * 0.04F}},
				fill, outline, stroke);
		base(size * 0.66F, size * 0.19F, size * 0.30F);
		break;
	default:
		break;
	}
}

/// Draws a labeled board and returns the clicked square when one was pressed.
std::optional<int> draw_board(const GameState &game, float size, bool flipped,
							  const Appearance &appearance,
							  std::optional<int> selected,
							  const std::vector<chess::Move> &targets,
							  const char *identifier) {
	ImGui::PushID(identifier);
	const ImVec2 origin = ImGui::GetCursorScreenPos();
	ImGui::InvisibleButton("board", {size, size});
	const bool clicked = ImGui::IsItemClicked(ImGuiMouseButton_Left);
	const ImVec2 mouse = ImGui::GetIO().MousePos;
	auto *draw = ImGui::GetWindowDrawList();
	const float square_size = size / 8.0F;
	const auto last = game.last_move();

	for(int screen_rank = 0; screen_rank < 8; ++screen_rank) {
		for(int screen_file = 0; screen_file < 8; ++screen_file) {
			const int file = flipped ? 7 - screen_file : screen_file;
			const int rank = flipped ? screen_rank : 7 - screen_rank;
			const int square = rank * 8 + file;
			const ImVec2 minimum = {
				origin.x + screen_file * square_size,
				origin.y + screen_rank * square_size,
			};
			const ImVec2 maximum = {minimum.x + square_size, minimum.y + square_size};
			ImU32 color = packed_color((file + rank) % 2 == 0 ? appearance.dark
																  : appearance.light);
			if(last && (last->from().index() == square || last->to().index() == square)) {
				color = packed_color(appearance.last_move);
			}
			if(selected && *selected == square) {
				color = packed_color(appearance.selected);
			}
			draw->AddRectFilled(minimum, maximum, color);

			const bool legal_target =
				std::any_of(targets.begin(), targets.end(), [&](const auto &move) {
					return move.to().index() == square;
				});
			if(legal_target) {
				const bool occupied =
					game.board().at(chess::Square(square)) != chess::Piece::NONE;
				if(occupied) {
					draw->AddCircle({minimum.x + square_size * 0.5F,
									 minimum.y + square_size * 0.5F},
									square_size * 0.43F,
									packed_color(appearance.legal), 32,
									std::max(3.0F, square_size * 0.07F));
				} else {
					draw->AddCircleFilled({minimum.x + square_size * 0.5F,
										  minimum.y + square_size * 0.5F},
										 square_size * 0.12F,
										 packed_color(appearance.legal), 24);
				}
			}

			const auto piece = game.board().at(chess::Square(square));
			if(piece != chess::Piece::NONE) {
				draw_piece(draw,
						   piece,
						   {minimum.x + square_size * 0.5F,
							minimum.y + square_size * 0.49F},
						   square_size);
			}

			if(appearance.coordinates) {
				const bool label_rank = screen_file == 0;
				const bool label_file = screen_rank == 7;
				const ImU32 label_color =
					packed_color((file + rank) % 2 == 0 ? appearance.light
																	 : appearance.dark);
				if(label_rank) {
					const std::string label(1, static_cast<char>('1' + rank));
					draw->AddText({minimum.x + 4.0F, minimum.y + 2.0F}, label_color,
								  label.c_str());
				}
				if(label_file) {
					const std::string label(1, static_cast<char>('a' + file));
					const auto dimensions = ImGui::CalcTextSize(label.c_str());
					draw->AddText({maximum.x - dimensions.x - 4.0F,
								   maximum.y - dimensions.y - 2.0F},
								  label_color, label.c_str());
				}
			}
		}
	}
	draw->AddRect(origin, {origin.x + size, origin.y + size},
				  color32(0.08F, 0.10F, 0.11F), 0.0F, 0,
				  std::max(1.0F, size * 0.003F));

	std::optional<int> result;
	if(clicked && mouse.x >= origin.x && mouse.x < origin.x + size &&
	   mouse.y >= origin.y && mouse.y < origin.y + size) {
		const int screen_file = static_cast<int>((mouse.x - origin.x) / square_size);
		const int screen_rank = static_cast<int>((mouse.y - origin.y) / square_size);
		const int file = flipped ? 7 - screen_file : screen_file;
		const int rank = flipped ? screen_rank : 7 - screen_rank;
		result = rank * 8 + file;
	}
	ImGui::PopID();
	return result;
}

/// Renders the live MultiPV table shared by Simulator and Stadium.
void analysis_table(const AnalysisSnapshot &snapshot, const std::string &fen,
					float height) {
	if(ImGui::BeginChild("analysis-lines", {0.0F, height},
						 ImGuiChildFlags_Borders)) {
		if(!snapshot.error.empty()) {
			ImGui::TextColored({0.95F, 0.32F, 0.30F, 1.0F}, "%s",
							   snapshot.error.c_str());
		} else if(snapshot.lines.empty()) {
			ImGui::TextDisabled(snapshot.searching ? "Waiting for analysis..."
													  : "Analysis is idle");
		} else if(ImGui::BeginTable(
					  "multipv", 5,
					  ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp |
						  ImGuiTableFlags_ScrollY)) {
			ImGui::TableSetupColumn("#", ImGuiTableColumnFlags_WidthFixed, 28.0F);
			ImGui::TableSetupColumn("Move", ImGuiTableColumnFlags_WidthFixed, 72.0F);
			ImGui::TableSetupColumn("Score", ImGuiTableColumnFlags_WidthFixed, 58.0F);
			ImGui::TableSetupColumn("Depth", ImGuiTableColumnFlags_WidthFixed, 48.0F);
			ImGui::TableSetupColumn("PV");
			ImGui::TableHeadersRow();
			for(const auto &line : snapshot.lines) {
				ImGui::TableNextRow();
				ImGui::TableSetColumnIndex(0);
				ImGui::Text("%d", line.multipv);
				ImGui::TableSetColumnIndex(1);
				ImGui::TextUnformatted(line.pv.empty() ? "-" : line.pv.front().c_str());
				ImGui::TableSetColumnIndex(2);
				ImGui::TextUnformatted(score_text(line).c_str());
				ImGui::TableSetColumnIndex(3);
				ImGui::Text("%d", line.depth);
				ImGui::TableSetColumnIndex(4);
				ImGui::TextUnformatted(pv_san(fen, line.pv).c_str());
			}
			ImGui::EndTable();
		}
	}
	ImGui::EndChild();
}

/// Applies restrained modern colors and compact desktop control spacing.
void apply_style(Theme theme) {
	if(theme == Theme::Light) {
		ImGui::StyleColorsLight();
	} else {
		ImGui::StyleColorsDark();
	}
	auto &style = ImGui::GetStyle();
	style.WindowRounding = 0.0F;
	style.ChildRounding = 5.0F;
	style.FrameRounding = 4.0F;
	style.PopupRounding = 6.0F;
	style.ScrollbarRounding = 4.0F;
	style.TabRounding = 4.0F;
	style.WindowPadding = {12.0F, 12.0F};
	style.FramePadding = {9.0F, 6.0F};
	style.ItemSpacing = {8.0F, 7.0F};
	if(theme == Theme::Light) {
		style.Colors[ImGuiCol_WindowBg] = {0.93F, 0.95F, 0.95F, 1.0F};
		style.Colors[ImGuiCol_ChildBg] = {0.975F, 0.98F, 0.98F, 1.0F};
		style.Colors[ImGuiCol_FrameBg] = {0.84F, 0.87F, 0.87F, 1.0F};
		style.Colors[ImGuiCol_FrameBgHovered] = {0.76F, 0.83F, 0.81F, 1.0F};
		style.Colors[ImGuiCol_Button] = {0.79F, 0.83F, 0.82F, 1.0F};
		style.Colors[ImGuiCol_ButtonHovered] = {0.55F, 0.72F, 0.66F, 1.0F};
		style.Colors[ImGuiCol_ButtonActive] = {0.36F, 0.62F, 0.53F, 1.0F};
		style.Colors[ImGuiCol_Header] = {0.61F, 0.76F, 0.70F, 1.0F};
		style.Colors[ImGuiCol_HeaderHovered] = {0.48F, 0.69F, 0.61F, 1.0F};
		style.Colors[ImGuiCol_CheckMark] = {0.11F, 0.48F, 0.35F, 1.0F};
		style.Colors[ImGuiCol_SliderGrab] = {0.16F, 0.53F, 0.40F, 1.0F};
	} else {
		style.Colors[ImGuiCol_WindowBg] = {0.075F, 0.085F, 0.09F, 1.0F};
		style.Colors[ImGuiCol_ChildBg] = {0.095F, 0.105F, 0.11F, 1.0F};
		style.Colors[ImGuiCol_FrameBg] = {0.14F, 0.155F, 0.16F, 1.0F};
		style.Colors[ImGuiCol_FrameBgHovered] = {0.19F, 0.22F, 0.22F, 1.0F};
		style.Colors[ImGuiCol_Button] = {0.16F, 0.19F, 0.19F, 1.0F};
		style.Colors[ImGuiCol_ButtonHovered] = {0.24F, 0.38F, 0.34F, 1.0F};
		style.Colors[ImGuiCol_ButtonActive] = {0.20F, 0.47F, 0.38F, 1.0F};
		style.Colors[ImGuiCol_Header] = {0.19F, 0.38F, 0.33F, 1.0F};
		style.Colors[ImGuiCol_HeaderHovered] = {0.23F, 0.46F, 0.39F, 1.0F};
		style.Colors[ImGuiCol_CheckMark] = {0.38F, 0.80F, 0.64F, 1.0F};
		style.Colors[ImGuiCol_SliderGrab] = {0.38F, 0.75F, 0.61F, 1.0F};
	}
}

/// Main application state and frame renderer.
class Application {
public:
	/// Loads persisted settings, then applies supported command-line overrides.
	Application(int argc, char **argv) {
		load_settings();
		parse_arguments(argc, argv);
		apply_style(theme_);
	}

	/// Stops every engine before the OpenGL context is destroyed.
	~Application() {
		simulator_engine_.close();
		white_engine_.close();
		black_engine_.close();
	}

	/// Draws one frame and optionally advances engines and match state.
	void frame(bool update_state = true) {
		if(update_state) {
			update_stadium();
		}
		const auto &io = ImGui::GetIO();
		ImGui::SetNextWindowPos({0.0F, 0.0F});
		ImGui::SetNextWindowSize(io.DisplaySize);
		ImGui::Begin("GadidaeRoot", nullptr,
					 ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
						 ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoSavedSettings);
		render_top_bar();
		ImGui::Separator();
		if(mode_ == Mode::Simulator) {
			render_simulator(update_state);
		} else {
			render_stadium();
		}
		render_settings();
		render_error_popup();
		ImGui::End();
	}

private:
	enum class Mode { Simulator, Stadium };

	/// Reads supported CLI options without introducing architecture-specific flags.
	void parse_arguments(int argc, char **argv) {
		for(int index = 1; index < argc; ++index) {
			const std::string argument = argv[index];
			const auto value = [&]() -> std::string {
				if(index + 1 >= argc) {
					throw std::invalid_argument("missing value after " + argument);
				}
				return argv[++index];
			};
			if(argument == "--mode") {
				mode_ = value() == "stadium" ? Mode::Stadium : Mode::Simulator;
			} else if(argument == "--theme") {
				theme_ = value() == "light" ? Theme::Light : Theme::Dark;
			} else if(argument == "--uci") {
				simulator_config_.path = value();
			} else if(argument == "--arguments") {
				simulator_config_.arguments = value();
			} else if(argument == "--device") {
				simulator_config_.device = value();
			} else if(argument == "--movetime-ms") {
				simulator_config_.movetime_ms = std::stoi(value());
			} else if(argument == "--node-limit") {
				simulator_config_.node_limit = std::stoull(value());
			} else if(argument == "--multipv") {
				simulator_config_.multipv = std::stoi(value());
			} else if(argument == "--fen") {
				simulator_.reset(value());
			} else if(argument == "--white-uci") {
				white_config_.path = value();
			} else if(argument == "--black-uci") {
				black_config_.path = value();
			}
		}
	}

	/// Loads user settings while treating a malformed file as recoverable.
	void load_settings() {
		const auto path = settings_path();
		if(!std::filesystem::exists(path)) {
			return;
		}
		try {
			std::ifstream input(path);
			const auto json = nlohmann::json::parse(input);
			if(json.contains("simulator")) {
				read_engine_json(json["simulator"], simulator_config_);
			}
			if(json.contains("white")) {
				read_engine_json(json["white"], white_config_);
			}
			if(json.contains("black")) {
				read_engine_json(json["black"], black_config_);
			}
			delay_ms_ = json.value("delay_ms", delay_ms_);
			max_plies_ = json.value("max_plies", max_plies_);
			flipped_ = json.value("flipped", flipped_);
			const auto appearance = json.value("appearance", nlohmann::json::object());
			auto read_color = [&](const char *name, ImVec4 &target) {
				if(appearance.contains(name) && appearance[name].is_array() &&
				   appearance[name].size() == 4) {
					target = {appearance[name][0].get<float>(),
							  appearance[name][1].get<float>(),
							  appearance[name][2].get<float>(),
							  appearance[name][3].get<float>()};
				}
			};
			read_color("light", appearance_.light);
			read_color("dark", appearance_.dark);
			read_color("selected", appearance_.selected);
			read_color("last_move", appearance_.last_move);
			appearance_.coordinates =
				appearance.value("coordinates", appearance_.coordinates);
			theme_ = json.value("theme", std::string("dark")) == "light"
						 ? Theme::Light
						 : Theme::Dark;
		} catch(...) {
			status_ = "Settings file was ignored because it could not be parsed";
		}
	}

	/// Atomically replaces the per-user JSON settings file.
	void save_settings() {
		const auto color_json = [](const ImVec4 &color) {
			return nlohmann::json::array({color.x, color.y, color.z, color.w});
		};
		const nlohmann::json json = {
			{"simulator", engine_json(simulator_config_)},
			{"white", engine_json(white_config_)},
			{"black", engine_json(black_config_)},
			{"theme", theme_ == Theme::Light ? "light" : "dark"},
			{"delay_ms", delay_ms_},
			{"max_plies", max_plies_},
			{"flipped", flipped_},
			{"appearance",
			 {{"light", color_json(appearance_.light)},
			  {"dark", color_json(appearance_.dark)},
			  {"selected", color_json(appearance_.selected)},
			  {"last_move", color_json(appearance_.last_move)},
			  {"coordinates", appearance_.coordinates}}},
		};
		const auto path = settings_path();
		std::filesystem::create_directories(path.parent_path());
		const auto temporary = path.string() + ".tmp";
		{
			std::ofstream output(temporary, std::ios::trunc);
			output << std::setw(2) << json << '\n';
		}
		std::error_code error;
		std::filesystem::remove(path, error);
		std::filesystem::rename(temporary, path);
	}

	/// Draws the stable mode selector without reconstructing either mode.
	void render_top_bar() {
		ImGui::AlignTextToFramePadding();
		ImGui::TextUnformatted("GADIDAE");
		ImGui::SameLine(116.0F);
		const auto mode_button = [&](const char *label, Mode mode) {
			const bool active = mode_ == mode;
			if(active) {
				ImGui::PushStyleColor(ImGuiCol_Button, {0.20F, 0.47F, 0.38F, 1.0F});
			}
			if(ImGui::Button(label, {104.0F, 0.0F}) && !active) {
				if(mode_ == Mode::Stadium) {
					stop_stadium();
				}
				simulator_engine_.stop_search();
				mode_ = mode;
				selected_square_.reset();
				legal_targets_.clear();
			}
			if(active) {
				ImGui::PopStyleColor();
			}
		};
		mode_button("Simulator", Mode::Simulator);
		ImGui::SameLine();
		mode_button("Stadium", Mode::Stadium);
		ImGui::SameLine();
		ImGui::SetCursorPosX(ImGui::GetWindowWidth() -
							 ImGui::CalcTextSize(status_.c_str()).x - 18.0F);
		ImGui::TextDisabled("%s", status_.c_str());
	}

	/// Renders the interactive position analyser.
	void render_simulator(bool update_state) {
		if(update_state) {
			ensure_simulator_analysis();
		}
		const float available_height = ImGui::GetContentRegionAvail().y;
		const float available_width = ImGui::GetContentRegionAvail().x;
		const float right_width = std::clamp(available_width * 0.39F, 410.0F, 610.0F);
		const float board_size =
			std::max(320.0F, std::min(available_height - 55.0F,
									 available_width - right_width - 16.0F));

		ImGui::BeginChild("simulator-left", {board_size, 0.0F});
		if(const auto square = draw_board(simulator_, board_size, flipped_, appearance_,
										 selected_square_, legal_targets_, "simulator")) {
			handle_simulator_square(*square);
		}
		render_simulator_bottom();
		ImGui::EndChild();
		ImGui::SameLine();
		ImGui::BeginChild("simulator-right", {0.0F, 0.0F});
		render_simulator_toolbar();
		ImGui::TextUnformatted("Suggested moves");
		analysis_table(simulator_display_, simulator_.board().getFen(),
					   available_height * 0.32F);
		ImGui::Spacing();
		ImGui::TextUnformatted("Engine analysis");
		render_engine_summary(simulator_display_, available_height * 0.18F);
		ImGui::Spacing();
		ImGui::TextUnformatted("Board state");
		render_board_state(simulator_, available_height * 0.32F);
		ImGui::EndChild();
	}

	/// Starts or refreshes simulator analysis only when its FEN changes.
	void ensure_simulator_analysis() {
		if(!analysis_open_ || simulator_config_.path.empty()) {
			return;
		}
		const std::string fen = simulator_.board().getFen();
		try {
			if(!simulator_engine_.ready()) {
				simulator_engine_.start(simulator_config_);
				invalidate_simulator_analysis();
			}
			if(simulator_analysis_fen_ != fen) {
				simulator_engine_.analyse(
					fen, simulator_config_.movetime_ms == 0 &&
							 simulator_config_.node_limit == 0);
				simulator_analysis_fen_ = fen;
				simulator_last_display_ = Clock::time_point{};
			}
			const auto now = Clock::now();
			if(simulator_last_display_ == Clock::time_point{} ||
			   now - simulator_last_display_ >=
				   std::chrono::milliseconds(
					   std::max(50, simulator_config_.progress_interval_ms)) ||
			   simulator_engine_.snapshot().finished) {
				simulator_display_ = simulator_engine_.snapshot();
				simulator_last_display_ = now;
			}
		} catch(const std::exception &error) {
			show_error(error.what());
			analysis_open_ = false;
			simulator_engine_.close();
		}
	}

	/// Discards analysis tied to the previous position before the UI renders a new FEN.
	void invalidate_simulator_analysis() {
		simulator_engine_.stop_search();
		simulator_analysis_fen_.clear();
		simulator_display_ = {};
		simulator_last_display_ = Clock::time_point{};
	}

	/// Handles piece selection, legal destinations, and default queen promotion.
	void handle_simulator_square(int square) {
		const auto piece = simulator_.board().at(chess::Square(square));
		if(selected_square_) {
			std::vector<chess::Move> matching;
			for(const auto &move : legal_targets_) {
				if(move.to().index() == square) {
					matching.push_back(move);
				}
			}
			if(!matching.empty()) {
				auto selected = matching.front();
				for(const auto &move : matching) {
					if(move.typeOf() == chess::Move::PROMOTION &&
					   move.promotionType() == chess::PieceType::QUEEN) {
						selected = move;
					}
				}
				simulator_.make_move(selected);
				selected_square_.reset();
				legal_targets_.clear();
				invalidate_simulator_analysis();
				return;
			}
		}
		if(piece != chess::Piece::NONE &&
		   piece.color() == simulator_.board().sideToMove()) {
			selected_square_ = square;
			legal_targets_ = simulator_.legal_moves_from(square);
		} else {
			selected_square_.reset();
			legal_targets_.clear();
		}
	}

	/// Draws compact board manipulation controls below Simulator.
	void render_simulator_bottom() {
		ImGui::Spacing();
		ImGui::SetNextItemWidth(-230.0F);
		ImGui::InputTextWithHint("##fen", "FEN or startpos", &simulator_fen_input_);
		ImGui::SameLine();
		if(ImGui::Button("Reset")) {
			try {
				simulator_.reset(simulator_fen_input_.empty() ? "startpos"
															 : simulator_fen_input_);
				selected_square_.reset();
				legal_targets_.clear();
				invalidate_simulator_analysis();
			} catch(const std::exception &error) {
				show_error(error.what());
			}
		}
		ImGui::SameLine();
		if(ImGui::Button("Undo")) {
			simulator_.undo();
			selected_square_.reset();
			legal_targets_.clear();
			invalidate_simulator_analysis();
		}
		ImGui::SameLine();
		if(ImGui::Button("Flip")) {
			flipped_ = !flipped_;
		}
	}

	/// Draws Simulator import/export, analysis, and settings commands.
	void render_simulator_toolbar() {
		if(ImGui::Button("Import PGN")) {
			if(const auto path =
				   file_dialog(false, L"PGN files\0*.pgn\0All files\0*.*\0", L"pgn")) {
				try {
					std::ifstream input(*path);
					std::ostringstream document;
					document << input.rdbuf();
					simulator_.import_pgn(document.str());
					selected_square_.reset();
					legal_targets_.clear();
					invalidate_simulator_analysis();
				} catch(const std::exception &error) {
					show_error(error.what());
				}
			}
		}
		ImGui::SameLine();
		if(ImGui::Button("Save PGN")) {
			save_pgn(simulator_, "White", "Black");
		}
		ImGui::SameLine();
		if(ImGui::Button(analysis_open_ ? "Close" : "Open")) {
			analysis_open_ = !analysis_open_;
			if(!analysis_open_) {
				simulator_engine_.stop_search();
				simulator_display_ = {};
			} else {
				invalidate_simulator_analysis();
			}
		}
		ImGui::SameLine();
		if(ImGui::Button("Settings")) {
			settings_mode_ = Mode::Simulator;
			simulator_edit_ = simulator_config_;
			appearance_edit_ = appearance_;
			theme_edit_ = theme_;
			settings_open_ = true;
		}
	}

	/// Renders stable search metadata separately from the changing MultiPV rows.
	void render_engine_summary(const AnalysisSnapshot &snapshot, float height) {
		if(ImGui::BeginChild("engine-summary", {0.0F, height},
							 ImGuiChildFlags_Borders)) {
			ImGui::Text("Engine: %s",
						snapshot.engine_name.empty() ? "-" : snapshot.engine_name.c_str());
			ImGui::Text("State: %s", snapshot.searching ? "thinking"
												 : snapshot.finished ? "complete" : "idle");
			if(!snapshot.lines.empty()) {
				const auto &line = snapshot.lines.front();
				ImGui::Text("Score: %s (side to move)", score_text(line).c_str());
				ImGui::Text("Depth: %d / %d", line.depth, line.seldepth);
				ImGui::Text("Nodes: %llu   NPS: %llu   Time: %d ms",
							static_cast<unsigned long long>(line.nodes),
							static_cast<unsigned long long>(line.nps), line.elapsed_ms);
			}
		}
		ImGui::EndChild();
	}

	/// Renders selectable FEN and wrapped PGN text.
	void render_board_state(const GameState &game, float height) {
		if(ImGui::BeginChild("board-state", {0.0F, height},
							 ImGuiChildFlags_Borders)) {
			ImGui::TextDisabled("FEN");
			ImGui::TextWrapped("%s", game.board().getFen().c_str());
			ImGui::Separator();
			ImGui::TextDisabled("PGN");
			ImGui::TextWrapped("%s", game.movetext().c_str());
		}
		ImGui::EndChild();
	}

	/// Renders one visible UCI match with live analysis for the side to move.
	void render_stadium() {
		const float available_height = ImGui::GetContentRegionAvail().y;
		const float available_width = ImGui::GetContentRegionAvail().x;
		const float right_width = std::clamp(available_width * 0.39F, 410.0F, 610.0F);
		const float board_size =
			std::max(320.0F, std::min(available_height - 55.0F,
									 available_width - right_width - 16.0F));
		ImGui::BeginChild("stadium-left", {board_size, 0.0F});
		draw_board(stadium_, board_size, flipped_, appearance_, std::nullopt, {},
				   "stadium");
		ImGui::Spacing();
		ImGui::SetNextItemWidth(-80.0F);
		ImGui::InputTextWithHint("##stadium-fen", "Start FEN or startpos",
								 &stadium_fen_input_);
		ImGui::SameLine();
		if(ImGui::Button("Flip##stadium")) {
			flipped_ = !flipped_;
		}
		ImGui::EndChild();
		ImGui::SameLine();
		ImGui::BeginChild("stadium-right", {0.0F, 0.0F});
		render_stadium_toolbar();
		const bool white_turn = stadium_.board().sideToMove() == chess::Color::WHITE;
		ImGui::Text("%s to move",
					white_turn ? stadium_white_name().c_str()
							   : stadium_black_name().c_str());
		analysis_table(stadium_display_, stadium_.board().getFen(),
					   available_height * 0.37F);
		ImGui::Spacing();
		ImGui::TextUnformatted("Match");
		if(ImGui::BeginChild("match-state", {0.0F, available_height * 0.18F},
							 ImGuiChildFlags_Borders)) {
			ImGui::Text("%s  vs  %s", stadium_white_name().c_str(),
						stadium_black_name().c_str());
			ImGui::Text("State: %s", stadium_status_.c_str());
			ImGui::Text("Ply: %zu / %d", stadium_.plies(), max_plies_);
			ImGui::Text("Result: %s", stadium_.result().c_str());
			if(!stadium_.termination().empty()) {
				ImGui::Text("Termination: %s", stadium_.termination().c_str());
			}
		}
		ImGui::EndChild();
		ImGui::Spacing();
		ImGui::TextUnformatted("Moves");
		render_board_state(stadium_, available_height * 0.26F);
		ImGui::EndChild();
	}

	/// Draws Stadium lifecycle and settings commands.
	void render_stadium_toolbar() {
		if(!stadium_running_) {
			if(ImGui::Button("Start")) {
				start_stadium();
			}
		} else {
			if(ImGui::Button(stadium_paused_ ? "Resume" : "Pause")) {
				stadium_paused_ = !stadium_paused_;
				if(stadium_paused_) {
					active_stadium_engine().stop_search();
					stadium_turn_started_ = false;
					stadium_status_ = "Paused";
				} else {
					next_stadium_turn_ = Clock::now();
					stadium_status_ = "Running";
				}
			}
			ImGui::SameLine();
			if(ImGui::Button("Stop")) {
				stop_stadium();
			}
		}
		ImGui::SameLine();
		if(ImGui::Button("Save PGN##stadium")) {
			save_pgn(stadium_, stadium_white_name(), stadium_black_name());
		}
		ImGui::SameLine();
		if(ImGui::Button("Settings##stadium")) {
			settings_mode_ = Mode::Stadium;
			white_edit_ = white_config_;
			black_edit_ = black_config_;
			appearance_edit_ = appearance_;
			theme_edit_ = theme_;
			delay_edit_ = delay_ms_;
			max_plies_edit_ = max_plies_;
			settings_open_ = true;
		}
	}

	/// Launches both UCI engines and initializes a fresh visible match.
	void start_stadium() {
		try {
			if(white_config_.path.empty() || black_config_.path.empty()) {
				throw std::invalid_argument("Stadium requires two UCI engine paths");
			}
			stop_stadium();
			stadium_.reset(stadium_fen_input_.empty() ? "startpos" : stadium_fen_input_);
			white_engine_.start(white_config_);
			black_engine_.start(black_config_);
			stadium_running_ = true;
			stadium_paused_ = false;
			stadium_turn_started_ = false;
			stadium_display_ = {};
			next_stadium_turn_ = Clock::now();
			stadium_status_ = "Running";
		} catch(const std::exception &error) {
			stop_stadium();
			show_error(error.what());
		}
	}

	/// Stops the match and guarantees both UCI child processes are gone.
	void stop_stadium() {
		white_engine_.close();
		black_engine_.close();
		stadium_running_ = false;
		stadium_paused_ = false;
		stadium_turn_started_ = false;
		if(stadium_status_ == "Running" || stadium_status_ == "Paused") {
			stadium_status_ = "Stopped";
		}
	}

	/// Advances Stadium's non-blocking turn state machine once per frame.
	void update_stadium() {
		if(!stadium_running_ || stadium_paused_) {
			return;
		}
		try {
			if(stadium_.over()) {
				stadium_status_ = stadium_.termination();
				stop_stadium();
				return;
			}
			if(stadium_.plies() >= static_cast<std::size_t>(max_plies_)) {
				stadium_status_ = "Maximum plies reached";
				stop_stadium();
				return;
			}
			const auto now = Clock::now();
			if(!stadium_turn_started_) {
				if(now < next_stadium_turn_) {
					return;
				}
				stadium_root_fen_ = stadium_.board().getFen();
				stadium_generation_ =
					active_stadium_engine().analyse(stadium_root_fen_, false);
				stadium_turn_started_ = true;
				stadium_last_display_ = Clock::time_point{};
				return;
			}
			const auto snapshot = active_stadium_engine().snapshot();
			const auto interval =
				std::max(50, active_stadium_config().progress_interval_ms);
			if(stadium_last_display_ == Clock::time_point{} ||
			   now - stadium_last_display_ >= std::chrono::milliseconds(interval) ||
			   snapshot.finished) {
				stadium_display_ = snapshot;
				stadium_last_display_ = now;
			}
			if(snapshot.generation != stadium_generation_ || !snapshot.finished) {
				return;
			}
			if(snapshot.bestmove.empty()) {
				throw std::runtime_error("UCI engine returned no legal bestmove");
			}
			stadium_.make_uci(snapshot.bestmove);
			stadium_turn_started_ = false;
			next_stadium_turn_ = now + std::chrono::milliseconds(delay_ms_);
		} catch(const std::exception &error) {
			show_error(error.what());
			stadium_status_ = "Error";
			stop_stadium();
		}
	}

	/// Returns the engine whose color matches the current position.
	UciEngine &active_stadium_engine() {
		return stadium_.board().sideToMove() == chess::Color::WHITE ? white_engine_
																	 : black_engine_;
	}

	/// Returns search settings for the current side to move.
	const EngineConfig &active_stadium_config() const {
		return stadium_.board().sideToMove() == chess::Color::WHITE ? white_config_
																	 : black_config_;
	}

	/// Returns a stable white player label.
	std::string stadium_white_name() const {
		if(!white_config_.name.empty()) {
			return white_config_.name;
		}
		const auto reported = white_engine_.display_name();
		return reported.empty() ? "White" : reported;
	}

	/// Returns a stable black player label.
	std::string stadium_black_name() const {
		if(!black_config_.name.empty()) {
			return black_config_.name;
		}
		const auto reported = black_engine_.display_name();
		return reported.empty() ? "Black" : reported;
	}

	/// Presents one engine's generic process and UCI controls.
	void engine_editor(const char *identifier, EngineConfig &config) {
		ImGui::PushID(identifier);
		std::string path = config.path.string();
		ImGui::TextUnformatted("Path");
		ImGui::SetNextItemWidth(-82.0F);
		if(ImGui::InputText("##path", &path)) {
			config.path = path;
		}
		ImGui::SameLine();
		if(ImGui::Button("Browse")) {
			if(const auto selected =
				   file_dialog(false, L"Executables\0*.exe\0All files\0*.*\0", L"exe")) {
				config.path = *selected;
			}
		}
		ImGui::TextUnformatted("Name");
		ImGui::SetNextItemWidth(-1.0F);
		ImGui::InputText("##name", &config.name);
		ImGui::TextUnformatted("Device");
		const char *devices[] = {"auto", "cpu", "cuda"};
		int device = config.device == "cpu" ? 1 : config.device == "cuda" ? 2 : 0;
		if(ImGui::Combo("##device", &device, devices, 3)) {
			config.device = devices[device];
		}
		ImGui::Spacing();
		ImGui::SeparatorText("Arguments");
		ImGui::TextUnformatted("Process arguments");
		ImGui::SetNextItemWidth(-1.0F);
		ImGui::InputText("##arguments", &config.arguments);
		ImGui::InputInt("Move time (ms)", &config.movetime_ms);
		ImGui::InputScalar("Node limit", ImGuiDataType_U64, &config.node_limit);
		ImGui::InputInt("Analysis lines", &config.multipv);
		ImGui::InputInt("Display update (ms)", &config.progress_interval_ms);
		config.movetime_ms = std::max(0, config.movetime_ms);
		config.multipv = std::max(1, config.multipv);
		config.progress_interval_ms = std::max(50, config.progress_interval_ms);
		ImGui::TextUnformatted("UCI options (JSON)");
		ImGui::InputTextMultiline("##options", &config.options, {-1.0F, 90.0F});
		ImGui::PopID();
	}

	/// Renders appearance controls shared by both modes.
	void appearance_editor(Appearance &appearance, Theme &theme) {
		const char *themes[] = {"Dark", "Light"};
		int selected_theme = theme == Theme::Light ? 1 : 0;
		ImGui::TextUnformatted("Theme");
		if(ImGui::Combo("##theme", &selected_theme, themes, 2)) {
			theme = selected_theme == 1 ? Theme::Light : Theme::Dark;
		}
		ImGui::SeparatorText("Board");
		if(ImGui::Button("Forest")) {
			appearance.light = color_vector(color32(0.91F, 0.93F, 0.91F));
			appearance.dark = color_vector(color32(0.29F, 0.45F, 0.39F));
		}
		ImGui::SameLine();
		if(ImGui::Button("Graphite")) {
			appearance.light = color_vector(color32(0.76F, 0.79F, 0.80F));
			appearance.dark = color_vector(color32(0.28F, 0.32F, 0.35F));
		}
		ImGui::SameLine();
		if(ImGui::Button("Ocean")) {
			appearance.light = color_vector(color32(0.79F, 0.86F, 0.86F));
			appearance.dark = color_vector(color32(0.22F, 0.43F, 0.50F));
		}
		ImGui::ColorEdit4("Light squares", &appearance.light.x,
						  ImGuiColorEditFlags_NoInputs);
		ImGui::ColorEdit4("Dark squares", &appearance.dark.x,
						  ImGuiColorEditFlags_NoInputs);
		ImGui::ColorEdit4("Selection", &appearance.selected.x,
						  ImGuiColorEditFlags_NoInputs);
		ImGui::ColorEdit4("Last move", &appearance.last_move.x,
						  ImGuiColorEditFlags_NoInputs);
		ImGui::Checkbox("Coordinates", &appearance.coordinates);
	}

	/// Renders modal settings without rebuilding either mode's visible layout.
	void render_settings() {
		if(settings_open_) {
			ImGui::OpenPopup("Settings");
			settings_open_ = false;
		}
		ImGui::SetNextWindowSize({720.0F, 620.0F}, ImGuiCond_Appearing);
		if(!ImGui::BeginPopupModal("Settings", nullptr,
								   ImGuiWindowFlags_NoSavedSettings)) {
			return;
		}
		if(settings_mode_ == Mode::Simulator) {
			if(ImGui::BeginTabBar("sim-settings")) {
				if(ImGui::BeginTabItem("Engine")) {
					engine_editor("simulator-edit", simulator_edit_);
					ImGui::EndTabItem();
				}
				if(ImGui::BeginTabItem("Appearance")) {
					appearance_editor(appearance_edit_, theme_edit_);
					ImGui::EndTabItem();
				}
				ImGui::EndTabBar();
			}
		} else if(ImGui::BeginTabBar("stadium-settings")) {
			if(ImGui::BeginTabItem("Engine 1")) {
				engine_editor("white-edit", white_edit_);
				ImGui::EndTabItem();
			}
			if(ImGui::BeginTabItem("Engine 2")) {
				engine_editor("black-edit", black_edit_);
				ImGui::EndTabItem();
			}
			if(ImGui::BeginTabItem("Match")) {
				ImGui::InputInt("Display delay (ms)", &delay_edit_);
				ImGui::InputInt("Maximum plies", &max_plies_edit_);
				delay_edit_ = std::max(0, delay_edit_);
				max_plies_edit_ = std::max(1, max_plies_edit_);
				ImGui::EndTabItem();
			}
			if(ImGui::BeginTabItem("Appearance")) {
				appearance_editor(appearance_edit_, theme_edit_);
				ImGui::EndTabItem();
			}
			ImGui::EndTabBar();
		}
		ImGui::SetCursorPosY(ImGui::GetWindowHeight() - 48.0F);
		ImGui::Separator();
		if(ImGui::Button("Apply", {100.0F, 0.0F})) {
			try {
				const auto primary_options = nlohmann::json::parse(
					settings_mode_ == Mode::Simulator ? simulator_edit_.options
													 : white_edit_.options);
				if(settings_mode_ == Mode::Stadium) {
					const auto secondary_options =
						nlohmann::json::parse(black_edit_.options);
					static_cast<void>(secondary_options);
				}
				static_cast<void>(primary_options);
				if(settings_mode_ == Mode::Simulator) {
					simulator_engine_.close();
					simulator_config_ = simulator_edit_;
					invalidate_simulator_analysis();
				} else {
					stop_stadium();
					white_config_ = white_edit_;
					black_config_ = black_edit_;
					delay_ms_ = delay_edit_;
					max_plies_ = max_plies_edit_;
			}
			appearance_ = appearance_edit_;
			theme_ = theme_edit_;
			apply_style(theme_);
			save_settings();
				ImGui::CloseCurrentPopup();
			} catch(const std::exception &error) {
				show_error(error.what());
			}
		}
		ImGui::SameLine();
		if(ImGui::Button("Cancel", {100.0F, 0.0F})) {
			ImGui::CloseCurrentPopup();
		}
		ImGui::EndPopup();
	}

	/// Saves a complete game through the platform file picker.
	void save_pgn(const GameState &game, const std::string &white,
				  const std::string &black) {
		if(const auto path =
			   file_dialog(true, L"PGN files\0*.pgn\0All files\0*.*\0", L"pgn")) {
			try {
				std::ofstream output(*path, std::ios::trunc);
				if(!output) {
					throw std::runtime_error("could not open PGN output");
				}
				output << game.pgn(white, black);
				status_ = "PGN saved";
			} catch(const std::exception &error) {
				show_error(error.what());
			}
		}
	}

	/// Defers an error message to a modal rendered on the UI thread.
	void show_error(std::string message) {
		error_message_ = std::move(message);
		error_open_ = true;
	}

	/// Renders the shared error modal.
	void render_error_popup() {
		if(error_open_) {
			ImGui::OpenPopup("Error");
			error_open_ = false;
		}
		if(ImGui::BeginPopupModal("Error", nullptr,
								  ImGuiWindowFlags_AlwaysAutoResize)) {
			ImGui::TextWrapped("%s", error_message_.c_str());
			if(ImGui::Button("Close", {100.0F, 0.0F})) {
				ImGui::CloseCurrentPopup();
			}
			ImGui::EndPopup();
		}
	}

	Mode mode_ = Mode::Simulator;
	Mode settings_mode_ = Mode::Simulator;
	GameState simulator_;
	GameState stadium_;
	UciEngine simulator_engine_;
	UciEngine white_engine_;
	UciEngine black_engine_;
	EngineConfig simulator_config_;
	EngineConfig white_config_;
	EngineConfig black_config_;
	EngineConfig simulator_edit_;
	EngineConfig white_edit_;
	EngineConfig black_edit_;
	Appearance appearance_;
	Appearance appearance_edit_;
	Theme theme_ = Theme::Dark;
	Theme theme_edit_ = Theme::Dark;
	bool flipped_ = false;
	bool analysis_open_ = true;
	bool settings_open_ = false;
	bool error_open_ = false;
	std::optional<int> selected_square_;
	std::vector<chess::Move> legal_targets_;
	std::string simulator_fen_input_ = "startpos";
	std::string simulator_analysis_fen_;
	AnalysisSnapshot simulator_display_;
	Clock::time_point simulator_last_display_{};
	bool stadium_running_ = false;
	bool stadium_paused_ = false;
	bool stadium_turn_started_ = false;
	std::uint64_t stadium_generation_ = 0;
	std::string stadium_fen_input_ = "startpos";
	std::string stadium_root_fen_;
	std::string stadium_status_ = "Ready";
	AnalysisSnapshot stadium_display_;
	Clock::time_point stadium_last_display_{};
	Clock::time_point next_stadium_turn_{};
	int delay_ms_ = 250;
	int max_plies_ = 240;
	int delay_edit_ = 250;
	int max_plies_edit_ = 240;
	std::string status_ = "Ready";
	std::string error_message_;
};

std::string latest_glfw_error;

/// Records GLFW initialization errors before an application window exists.
void glfw_error(int, const char *description) {
	latest_glfw_error = description == nullptr ? "unknown GLFW error" : description;
}

/// Carries a re-entrancy-safe draw function into GLFW's modal refresh callback.
struct RefreshContext {
	std::function<void(bool)> render;
	bool rendering = false;
	Clock::time_point last_render{};
};

/// Draws at most one frame per interval and prevents recursive callbacks.
void refresh_window(GLFWwindow *window, bool update_state) {
	auto *context =
		static_cast<RefreshContext *>(glfwGetWindowUserPointer(window));
	if(context == nullptr || context->rendering) {
		return;
	}
	const auto now = Clock::now();
	const auto minimum_interval = update_state
		? std::chrono::milliseconds(8)
		: std::chrono::milliseconds(33);
	if(context->last_render != Clock::time_point{} &&
	   now - context->last_render < minimum_interval) {
		return;
	}
	context->rendering = true;
	context->render(update_state);
	context->last_render = Clock::now();
	context->rendering = false;
}

/// Redraws cached UI state while Windows owns its modal move or resize loop.
void refresh_window_modal(GLFWwindow *window) {
	refresh_window(window, false);
}

} // namespace

int run_application(int argc, char **argv) {
	glfwSetErrorCallback(glfw_error);
	if(!glfwInit()) {
		throw std::runtime_error(
			latest_glfw_error.empty() ? "could not initialize GLFW"
									 : "could not initialize GLFW: " + latest_glfw_error);
	}
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
#endif
	GLFWwindow *window =
		glfwCreateWindow(1280, 820, "Gadidae", nullptr, nullptr);
	if(window == nullptr) {
		glfwTerminate();
		throw std::runtime_error(
			latest_glfw_error.empty() ? "could not create the OpenGL window"
									 : "could not create the OpenGL window: " +
										   latest_glfw_error);
	}
	glfwMakeContextCurrent(window);
	glfwSetWindowSizeLimits(window, 960, 640, GLFW_DONT_CARE, GLFW_DONT_CARE);
	glfwSwapInterval(0);
	if(gladLoadGL(glfwGetProcAddress) == 0) {
		glfwDestroyWindow(window);
		glfwTerminate();
		throw std::runtime_error("could not load OpenGL 3.3");
	}

	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	auto &io = ImGui::GetIO();
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
	io.IniFilename = nullptr;
	if(const auto font = system_font_path()) {
		io.Fonts->AddFontFromFileTTF(font->string().c_str(), 17.0F);
	}
	apply_style(Theme::Dark);
	ImGui_ImplGlfw_InitForOpenGL(window, true);
	ImGui_ImplOpenGL3_Init("#version 330 core");

	int result = 0;
	try {
		Application application(argc, argv);
		const auto render_frame = [&](bool update_state) {
			ImGui_ImplOpenGL3_NewFrame();
			ImGui_ImplGlfw_NewFrame();
			ImGui::NewFrame();
			application.frame(update_state);
			ImGui::Render();
			int width = 0;
			int height = 0;
			glfwGetFramebufferSize(window, &width, &height);
			glViewport(0, 0, width, height);
			const auto clear = ImGui::GetStyle().Colors[ImGuiCol_WindowBg];
			glClearColor(clear.x, clear.y, clear.z, clear.w);
			glClear(GL_COLOR_BUFFER_BIT);
			ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
			glfwSwapBuffers(window);
		};
		RefreshContext refresh{render_frame};
		glfwSetWindowUserPointer(window, &refresh);
		glfwSetWindowRefreshCallback(window, refresh_window_modal);
		auto next_frame = Clock::now();
		while(!glfwWindowShouldClose(window)) {
			glfwPollEvents();
			refresh_window(window, true);
			next_frame += std::chrono::milliseconds(8);
			const auto now = Clock::now();
			if(next_frame > now) {
				std::this_thread::sleep_until(next_frame);
			} else {
				next_frame = now;
			}
		}
		glfwSetWindowRefreshCallback(window, nullptr);
		glfwSetWindowUserPointer(window, nullptr);
	} catch(const std::exception &error) {
		std::cerr << "GUI runtime error: " << error.what() << '\n';
		result = 1;
	}

	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();
	glfwDestroyWindow(window);
	glfwTerminate();
	return result;
}

} // namespace gadidae::graphics
