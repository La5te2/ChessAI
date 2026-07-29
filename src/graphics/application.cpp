// Implements the native OpenGL Gadidae interface. Simulator and Stadium share
// one renderer and chess state while communicating with engines only through UCI.
#include "graphics/application.hpp"
#include "graphics/game.hpp"
#include "graphics/pieces.hpp"
#include "graphics/registry.hpp"
#include "graphics/simulator.hpp"
#include "graphics/stadium.hpp"
#include "graphics/uci.hpp"
#include <glad/gl.h>
#define GLFW_INCLUDE_NONE
#ifdef _WIN32
#define NOMINMAX
#endif
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
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#ifdef _WIN32
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
		std::filesystem::path("C:/Windows/Fonts/segoeui.ttf"),
		std::filesystem::path("C:/Windows/Fonts/seguisb.ttf"),
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

/// Returns the former per-user path used only for one-time config migration.
std::filesystem::path legacy_settings_path() {
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

/// Places portable GUI settings beside the running executable.
std::filesystem::path executable_settings_path(const char *argument_zero) {
	std::error_code error;
	if(argument_zero != nullptr && argument_zero[0] != '\0') {
		auto executable =
			std::filesystem::absolute(std::filesystem::path(argument_zero), error);
		if(!error) {
			return executable.parent_path() / "gui.json";
		}
	}
	return std::filesystem::current_path() / "gui.json";
}

/// Moves the former per-user config into the portable executable directory.
void migrate_legacy_settings(const std::filesystem::path &destination) {
	if(std::filesystem::exists(destination)) {
		return;
	}
	const auto source = legacy_settings_path();
	if(source == destination || !std::filesystem::is_regular_file(source)) {
		return;
	}
	std::error_code error;
	std::filesystem::create_directories(destination.parent_path(), error);
	if(error) {
		return;
	}
	std::filesystem::copy_file(source, destination,
							   std::filesystem::copy_options::none, error);
	if(!error) {
		std::filesystem::remove(source, error);
	}
}

/// Mutable board colors and presentation preferences stored with the GUI config.
struct Appearance {
	ImVec4 light = color_vector(color32(0.91F, 0.93F, 0.91F));
	ImVec4 dark = color_vector(color32(0.29F, 0.45F, 0.39F));
	ImVec4 selected = color_vector(color32(0.93F, 0.73F, 0.25F));
	ImVec4 last_move = color_vector(color32(0.66F, 0.75F, 0.36F));
	ImVec4 legal = color_vector(color32(0.20F, 0.24F, 0.23F, 0.34F));
	bool coordinates = true;
	float font_size = 20.0F;
	bool auto_font_scale = true;
	std::string piece_style = "vector";
};

struct BoardLayout {
	float board_size = 0.0F;
	float minimum_board_size = 0.0F;
	float maximum_board_size = 0.0F;
	float splitter_width = 8.0F;
};

/// Pixel heights assigned to the three vertically resizable right-side panels.
struct RightPanelHeights {
	std::array<float, 3> values{};
	float content_height = 0.0F;

	float operator[](std::size_t index) const {
		return values[index];
	}
};

/// Fits a user-sized square board beside a readable right panel.
BoardLayout board_layout(float available_width, float available_height,
						 float board_fraction) {
	constexpr float splitter_width = 8.0F;
	const float right_minimum = std::min(360.0F, available_width * 0.42F);
	const float bottom_controls =
		ImGui::GetFrameHeight() + ImGui::GetStyle().ItemSpacing.y * 2.0F +
		ImGui::GetStyle().WindowPadding.y * 2.0F + 4.0F;
	const float maximum_board_size =
		std::max(1.0F, std::min(available_height - bottom_controls,
							   available_width - right_minimum - splitter_width));
	const float minimum_board_size = std::min(280.0F, maximum_board_size);
	const float board_size =
		std::clamp(available_width * board_fraction, minimum_board_size,
				   maximum_board_size);
	return {board_size, minimum_board_size, maximum_board_size, splitter_width};
}

/// Selects application chrome independently from chessboard colors.
enum class Theme { Dark, Light };

/// Formats a match clock with tenths during the final minute.
std::string clock_text(std::int64_t milliseconds) {
	milliseconds = std::max<std::int64_t>(0, milliseconds);
	const auto total_seconds = milliseconds / 1000;
	const auto minutes = total_seconds / 60;
	const auto seconds = total_seconds % 60;
	std::ostringstream output;
	output << minutes << ':' << std::setfill('0') << std::setw(2) << seconds;
	if(milliseconds < 60'000) {
		output << '.' << (milliseconds % 1000) / 100;
	}
	return output.str();
}

/// Draws one compact boolean switch with an explicit label.
bool toggle_switch(const char *label, bool &value) {
	ImGui::PushID(label);
	ImGui::AlignTextToFramePadding();
	ImGui::TextUnformatted(label);
	ImGui::SameLine();
	const float height = ImGui::GetFrameHeight();
	const float width = height * 1.75F;
	const auto position = ImGui::GetCursorScreenPos();
	const bool pressed = ImGui::InvisibleButton("##toggle", {width, height});
	if(pressed) {
		value = !value;
	}
	const auto background = ImGui::GetColorU32(
		value ? ImGuiCol_SliderGrabActive : ImGuiCol_FrameBg);
	const auto knob = ImGui::GetColorU32(ImGuiCol_Text);
	auto *draw = ImGui::GetWindowDrawList();
	draw->AddRectFilled(position, {position.x + width, position.y + height},
						background, height * 0.5F);
	const float radius = height * 0.38F;
	const float center_x =
		value ? position.x + width - height * 0.5F
			  : position.x + height * 0.5F;
	draw->AddCircleFilled({center_x, position.y + height * 0.5F}, radius,
						  knob, 20);
	ImGui::PopID();
	return pressed;
}

/// Draws one step button whose symbol is centered by geometry instead of font metrics.
bool centered_step_button(const char *identifier, bool increment) {
	const float side = ImGui::GetFrameHeight();
	const bool pressed = ImGui::Button(identifier, {side, side});
	const ImVec2 minimum = ImGui::GetItemRectMin();
	const ImVec2 maximum = ImGui::GetItemRectMax();
	const ImVec2 center = {
		(minimum.x + maximum.x) * 0.5F,
		(minimum.y + maximum.y) * 0.5F,
	};
	const float half_extent = side * 0.12F;
	const float thickness = std::max(1.5F, side * 0.045F);
	const auto color = ImGui::GetColorU32(ImGuiCol_Text);
	auto *draw = ImGui::GetWindowDrawList();
	draw->AddLine({center.x - half_extent, center.y},
				  {center.x + half_extent, center.y}, color, thickness);
	if(increment) {
		draw->AddLine({center.x, center.y - half_extent},
					  {center.x, center.y + half_extent}, color, thickness);
	}
	return pressed;
}

/// Edits one signed integer with geometrically centered decrement and increment controls.
bool step_input(const char *label, int &value) {
	ImGui::PushID(label);
	const auto &style = ImGui::GetStyle();
	const float button_extent = ImGui::GetFrameHeight();
	const float input_width =
		std::max(1.0F, ImGui::CalcItemWidth() -
						   2.0F * (button_extent + style.ItemInnerSpacing.x));
	ImGui::SetNextItemWidth(input_width);
	bool changed =
		ImGui::InputScalar("##value", ImGuiDataType_S32, &value, nullptr, nullptr);
	ImGui::SameLine(0.0F, style.ItemInnerSpacing.x);
	if(centered_step_button("##decrement", false) &&
	   value > std::numeric_limits<int>::min()) {
		--value;
		changed = true;
	}
	ImGui::SameLine(0.0F, style.ItemInnerSpacing.x);
	if(centered_step_button("##increment", true) &&
	   value < std::numeric_limits<int>::max()) {
		++value;
		changed = true;
	}
	ImGui::SameLine(0.0F, style.ItemInnerSpacing.x);
	ImGui::TextUnformatted(label);
	ImGui::PopID();
	return changed;
}

/// Edits one unsigned integer with geometrically centered decrement and increment controls.
bool step_input(const char *label, std::uint64_t &value) {
	ImGui::PushID(label);
	const auto &style = ImGui::GetStyle();
	const float button_extent = ImGui::GetFrameHeight();
	const float input_width =
		std::max(1.0F, ImGui::CalcItemWidth() -
						   2.0F * (button_extent + style.ItemInnerSpacing.x));
	ImGui::SetNextItemWidth(input_width);
	bool changed =
		ImGui::InputScalar("##value", ImGuiDataType_U64, &value, nullptr, nullptr);
	ImGui::SameLine(0.0F, style.ItemInnerSpacing.x);
	if(centered_step_button("##decrement", false) && value > 0) {
		--value;
		changed = true;
	}
	ImGui::SameLine(0.0F, style.ItemInnerSpacing.x);
	if(centered_step_button("##increment", true) &&
	   value < std::numeric_limits<std::uint64_t>::max()) {
		++value;
		changed = true;
	}
	ImGui::SameLine(0.0F, style.ItemInnerSpacing.x);
	ImGui::TextUnformatted(label);
	ImGui::PopID();
	return changed;
}

/// Draws a compact clipboard command using two overlapping document outlines.
bool copy_button(const char *identifier) {
	ImGui::PushID(identifier);
	const float side = ImGui::GetTextLineHeight();
	const bool pressed = ImGui::InvisibleButton("##copy", {side, side});
	const ImVec2 minimum = ImGui::GetItemRectMin();
	const ImVec2 maximum = ImGui::GetItemRectMax();
	const bool hovered = ImGui::IsItemHovered();
	if(hovered) {
		const ImVec4 background = ImGui::GetStyleColorVec4(ImGuiCol_WindowBg);
		const ImVec4 foreground = ImGui::GetStyleColorVec4(ImGuiCol_Text);
		constexpr float foreground_mix = 0.12F;
		const ImVec4 neutral = {
			background.x * (1.0F - foreground_mix) +
				foreground.x * foreground_mix,
			background.y * (1.0F - foreground_mix) +
				foreground.y * foreground_mix,
			background.z * (1.0F - foreground_mix) +
				foreground.z * foreground_mix,
			1.0F,
		};
		ImGui::GetWindowDrawList()->AddRectFilled(
			minimum, maximum, ImGui::GetColorU32(neutral), 2.0F);
	}
	const auto color =
		ImGui::GetColorU32(hovered ? ImGuiCol_Text : ImGuiCol_TextDisabled);
	const float unit = side / 16.0F;
	auto *draw = ImGui::GetWindowDrawList();
	draw->AddRect({minimum.x + 6.0F * unit, minimum.y + 2.0F * unit},
				  {minimum.x + 14.0F * unit, minimum.y + 10.0F * unit}, color,
				  1.0F, 0, std::max(1.0F, unit));
	draw->AddRect({minimum.x + 2.0F * unit, minimum.y + 6.0F * unit},
				  {minimum.x + 10.0F * unit, minimum.y + 14.0F * unit}, color,
				  1.0F, 0, std::max(1.0F, unit));
	ImGui::PopID();
	return pressed;
}

/// Commands issued by the shared ImGui application menu.
enum class MenuCommand {
	ImportPgn,
	SavePgn,
	SetFen,
	Reset,
	Undo,
	Flip,
	Start,
	Pause,
	Stop,
	SimulatorMode,
	StadiumMode,
	Settings,
	ImportEngine,
	Matches,
};

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

/// Validates additional setoption values while reserving fields managed by the GUI.
nlohmann::json additional_uci_options(const std::string &text) {
	const auto options = nlohmann::json::parse(text.empty() ? "{}" : text);
	if(!options.is_object()) {
		throw std::invalid_argument("UCI options must be a JSON object");
	}
	for(auto iterator = options.begin(); iterator != options.end(); ++iterator) {
		std::string key = iterator.key();
		std::transform(key.begin(), key.end(), key.begin(), [](unsigned char character) {
			return static_cast<char>(std::tolower(character));
		});
		if(key == "device" || key == "multipv") {
			throw std::invalid_argument(
				iterator.key() +
				" is managed by its dedicated field and must be removed from UCI options");
		}
	}
	return options;
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

/// Draws one built-in procedural chess piece at any board scale.
void draw_vector_piece(ImDrawList *draw, const chess::Piece &piece,
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

/// Selects the embedded SVG atlas or falls back to the built-in Vector style.
void draw_piece(ImDrawList *draw, std::string_view style,
				const chess::Piece &piece,
				ImVec2 center, float size) {
	if(style != "vector" &&
	   draw_compiled_piece(draw, style, piece, center, size * 0.94F)) {
		return;
	}
	draw_vector_piece(draw, piece, center, size);
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
						   appearance.piece_style,
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
					  ImGuiTableFlags_RowBg | ImGuiTableFlags_Resizable |
						  ImGuiTableFlags_BordersInnerV |
						  ImGuiTableFlags_SizingStretchProp |
						  ImGuiTableFlags_ScrollY)) {
			const auto column_width = [](const char *sample, float minimum) {
				const float content = ImGui::CalcTextSize(sample).x;
				const float padding = ImGui::GetStyle().CellPadding.x * 2.0F;
				return std::max(minimum, content + padding + 8.0F);
			};
			ImGui::TableSetupColumn("#", ImGuiTableColumnFlags_WidthFixed,
									column_width("##", 32.0F));
			ImGui::TableSetupColumn("Move", ImGuiTableColumnFlags_WidthFixed,
									column_width("O-O-O+", 80.0F));
			ImGui::TableSetupColumn("Score", ImGuiTableColumnFlags_WidthFixed,
									column_width("-M123", 72.0F));
			ImGui::TableSetupColumn("Depth", ImGuiTableColumnFlags_WidthFixed,
									column_width("Depth", 64.0F));
			ImGui::TableSetupColumn("PV", ImGuiTableColumnFlags_WidthStretch);
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
	/// Loads portable settings, then applies supported command-line overrides.
	Application(int argc, char **argv)
		: settings_path_(
			  executable_settings_path(argc > 0 ? argv[0] : nullptr)) {
		migrate_legacy_settings(settings_path_);
		load_settings();
		parse_arguments(argc, argv);
		apply_style(theme_);
	}

	/// Stops every engine before the OpenGL context is destroyed.
	~Application() {
		stadiums_.stop_all();
	}

	/// Executes one menu command through the same application actions on every platform.
	void handle_menu_command(MenuCommand command) {
		switch(command) {
		case MenuCommand::ImportPgn:
			if(mode_ == Mode::Simulator) {
				import_simulator_pgn();
			}
			break;
		case MenuCommand::ImportEngine:
			import_uci_engine();
			break;
		case MenuCommand::SavePgn:
			if(mode_ == Mode::Simulator) {
				save_pgn(simulator_.game(), "White", "Black");
			} else {
				save_pgn(stadium().game(), stadium().white_name(),
						 stadium().black_name(), stadium().result(),
						 stadium().termination());
			}
			break;
		case MenuCommand::SetFen:
			if(mode_ != Mode::Stadium || !stadium().running()) {
				open_position_editor();
			}
			break;
		case MenuCommand::Reset:
			if(mode_ != Mode::Stadium || !stadium().running()) {
				reset_active_board();
			}
			break;
		case MenuCommand::Undo:
			if(mode_ == Mode::Simulator) {
				undo_simulator();
			}
			break;
		case MenuCommand::Flip:
			flipped_ = !flipped_;
			break;
		case MenuCommand::Start:
			if(mode_ == Mode::Simulator) {
				toggle_simulator_analysis();
			} else if(!stadium().running()) {
				start_stadium();
			}
			break;
		case MenuCommand::Pause:
			if(mode_ == Mode::Stadium) {
				toggle_stadium_pause();
			}
			break;
		case MenuCommand::Stop:
			if(mode_ == Mode::Simulator) {
				if(simulator_.analysis_open()) {
					toggle_simulator_analysis();
				}
			} else if(stadium().running()) {
				stop_stadium();
			}
			break;
		case MenuCommand::SimulatorMode:
			set_mode(Mode::Simulator);
			break;
		case MenuCommand::StadiumMode:
			set_mode(Mode::Stadium);
			break;
		case MenuCommand::Settings:
			open_settings(mode_);
			break;
		case MenuCommand::Matches:
			if(mode_ == Mode::Stadium) {
				matches_open_ = true;
			}
			break;
		}
	}

	/// Draws one frame and optionally advances engines and match state.
	void frame(bool update_state = true) {
		if(update_state) {
			update_stadium();
		}
		auto &io = ImGui::GetIO();
		const float window_ratio =
			std::max(0.25F, std::min(io.DisplaySize.x / 1280.0F,
									 io.DisplaySize.y / 820.0F));
		const float viewport_scale = appearance_.auto_font_scale
			? std::clamp(std::sqrt(window_ratio), 0.95F, 1.10F)
			: 1.0F;
		io.FontGlobalScale =
			std::clamp(appearance_.font_size * viewport_scale / 32.0F, 0.44F, 1.25F);
		ImGui::SetNextWindowPos({0.0F, 0.0F});
		ImGui::SetNextWindowSize(io.DisplaySize);
		auto window_flags =
			ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
			ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoSavedSettings |
			ImGuiWindowFlags_MenuBar;
		const auto root_padding = ImGui::GetStyle().WindowPadding;
		ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {root_padding.x, 1.0F});
		ImGui::Begin("GadidaeRoot", nullptr, window_flags);
		ImGui::PopStyleVar();
		render_menu_bar();
		if(mode_ == Mode::Simulator) {
			render_simulator(update_state);
		} else {
			render_stadium();
		}
		render_position_editor();
		render_matches();
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
			} else if(argument == "--font-size") {
				appearance_.font_size = std::clamp(std::stof(value()), 14.0F, 36.0F);
			} else if(argument == "--piece-style") {
				const auto style = value();
				if(!piece_style_available(style)) {
					throw std::invalid_argument(
						"--piece-style is not compiled into this executable");
				}
				appearance_.piece_style = style;
			} else if(argument == "--uci") {
				simulator_.config().path = value();
			} else if(argument == "--arguments") {
				simulator_.config().arguments = value();
			} else if(argument == "--device") {
				simulator_.config().device = value();
			} else if(argument == "--movetime-ms") {
				simulator_.config().movetime_ms = std::stoi(value());
			} else if(argument == "--node-limit") {
				simulator_.config().node_limit = std::stoull(value());
			} else if(argument == "--multipv") {
				simulator_.config().multipv = std::stoi(value());
			} else if(argument == "--fen") {
				simulator_.set_start_fen(value());
				simulator_.reset();
			} else if(argument == "--white-uci") {
				stadium().white_config().path = value();
			} else if(argument == "--white-name") {
				stadium().white_config().name = value();
			} else if(argument == "--black-uci") {
				stadium().black_config().path = value();
			} else if(argument == "--black-name") {
				stadium().black_config().name = value();
			}
		}
	}

	/// Loads user settings while treating a malformed file as recoverable.
	void load_settings() {
		if(!std::filesystem::exists(settings_path_)) {
			return;
		}
		try {
			std::ifstream input(settings_path_);
			const auto json = nlohmann::json::parse(input);
			if(json.contains("simulator")) {
				read_engine_json(json["simulator"], simulator_.config());
			}
			if(json.contains("white")) {
				read_engine_json(json["white"], stadium().white_config());
			}
			if(json.contains("black")) {
				read_engine_json(json["black"], stadium().black_config());
			}
			if(json.contains("engines") && json["engines"].is_array()) {
				std::vector<EngineConfig> engines;
				for(const auto &entry : json["engines"]) {
					EngineConfig engine;
					read_engine_json(entry, engine);
					engines.push_back(std::move(engine));
				}
				registry_.replace(std::move(engines));
			}
			stadium().set_match_limits(
				json.value("delay_ms", stadium().display_delay_ms()),
				json.value("max_plies", stadium().max_plies()));
			stadium().set_name(json.value("match_name", std::string{}));
			stadium().set_clock(
				json.value("clock_initial_ms",
						   stadium().clock_initial_ms()),
				json.value("clock_increment_ms",
						   stadium().clock_increment_ms()));
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
			read_color("legal", appearance_.legal);
			appearance_.coordinates =
				appearance.value("coordinates", appearance_.coordinates);
			appearance_.font_size =
				std::clamp(appearance.value("font_size", appearance_.font_size),
						   14.0F, 36.0F);
			appearance_.auto_font_scale =
				appearance.value("auto_font_scale", appearance_.auto_font_scale);
			const auto style =
				appearance.value("piece_style", appearance_.piece_style);
			if(piece_style_available(style)) {
				appearance_.piece_style = style;
			}
			theme_ = json.value("theme", std::string("dark")) == "light"
						 ? Theme::Light
						 : Theme::Dark;
		} catch(...) {
			// Keep defaults when a stale or partial settings file cannot be read.
		}
	}

	/// Atomically replaces the portable JSON settings file.
	void save_settings() {
		const auto color_json = [](const ImVec4 &color) {
			return nlohmann::json::array({color.x, color.y, color.z, color.w});
		};
		nlohmann::json engines = nlohmann::json::array();
		for(const auto &engine : registry_.engines()) {
			engines.push_back(engine_json(engine));
		}
		const nlohmann::json json = {
			{"simulator", engine_json(simulator_.config())},
			{"white", engine_json(stadium().white_config())},
			{"black", engine_json(stadium().black_config())},
			{"engines", std::move(engines)},
			{"theme", theme_ == Theme::Light ? "light" : "dark"},
			{"delay_ms", stadium().display_delay_ms()},
			{"max_plies", stadium().max_plies()},
			{"match_name", stadium().name()},
			{"clock_initial_ms", stadium().clock_initial_ms()},
			{"clock_increment_ms", stadium().clock_increment_ms()},
			{"flipped", flipped_},
			{"appearance",
			 {{"light", color_json(appearance_.light)},
			  {"dark", color_json(appearance_.dark)},
			  {"selected", color_json(appearance_.selected)},
			  {"last_move", color_json(appearance_.last_move)},
			  {"legal", color_json(appearance_.legal)},
			  {"coordinates", appearance_.coordinates},
			  {"font_size", appearance_.font_size},
			  {"auto_font_scale", appearance_.auto_font_scale},
			  {"piece_style", appearance_.piece_style}}},
		};
		std::filesystem::create_directories(settings_path_.parent_path());
		const auto temporary = settings_path_.string() + ".tmp";
		{
			std::ofstream output(temporary, std::ios::trunc);
			output << std::setw(2) << json << '\n';
		}
		std::error_code error;
		std::filesystem::remove(settings_path_, error);
		std::filesystem::rename(temporary, settings_path_);
	}

	/// Draws conventional application menus while keeping the work area uncluttered.
	void render_menu_bar() {
		if(!ImGui::BeginMenuBar()) {
			return;
		}
		const auto command_item = [](const char *label) {
			const ImVec2 size = ImGui::CalcTextSize(label);
			return ImGui::Selectable(
				label, false, ImGuiSelectableFlags_None, size);
		};
		const auto popup_padding = ImGui::GetStyle().WindowPadding;
		ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding,
							{8.0F, popup_padding.y});
		if(ImGui::BeginMenu("File")) {
			if(command_item("Import Engine")) {
				handle_menu_command(MenuCommand::ImportEngine);
			}
			if(mode_ == Mode::Simulator && command_item("Import PGN")) {
				handle_menu_command(MenuCommand::ImportPgn);
			}
			if(command_item("Save PGN")) {
				handle_menu_command(MenuCommand::SavePgn);
			}
			ImGui::EndMenu();
		}
		if(ImGui::BeginMenu("Board")) {
			const bool position_locked =
				mode_ == Mode::Stadium && stadium().running();
			ImGui::BeginDisabled(position_locked);
			if(command_item("FEN")) {
				handle_menu_command(MenuCommand::SetFen);
			}
			if(command_item("Reset")) {
				handle_menu_command(MenuCommand::Reset);
			}
			ImGui::EndDisabled();
			if(mode_ == Mode::Simulator) {
				ImGui::BeginDisabled(simulator_.game().plies() == 0);
				if(command_item("Undo")) {
					handle_menu_command(MenuCommand::Undo);
				}
				ImGui::EndDisabled();
			}
			ImGui::Separator();
			if(command_item("Flip")) {
				handle_menu_command(MenuCommand::Flip);
			}
			ImGui::EndMenu();
		}
		if(ImGui::BeginMenu("Mode")) {
			if(ImGui::MenuItem("Simulator", nullptr,
							   mode_ == Mode::Simulator)) {
				handle_menu_command(MenuCommand::SimulatorMode);
			}
			if(ImGui::MenuItem("Stadium", nullptr, mode_ == Mode::Stadium)) {
				handle_menu_command(MenuCommand::StadiumMode);
			}
			ImGui::EndMenu();
		}
		if(ImGui::BeginMenu("Run")) {
			if(mode_ == Mode::Simulator) {
				const bool open = simulator_.analysis_open();
				if(ImGui::MenuItem("Open", nullptr, open) && !open) {
					handle_menu_command(MenuCommand::Start);
				}
				if(ImGui::MenuItem("Close", nullptr, !open) && open) {
					handle_menu_command(MenuCommand::Stop);
				}
			} else {
				ImGui::BeginDisabled(stadium().running());
				if(command_item("Start")) {
					handle_menu_command(MenuCommand::Start);
				}
				ImGui::EndDisabled();
				ImGui::BeginDisabled(!stadium().running());
				if(command_item(stadium().paused() ? "Resume" : "Pause")) {
					handle_menu_command(MenuCommand::Pause);
				}
				if(command_item("Stop")) {
					handle_menu_command(MenuCommand::Stop);
				}
				ImGui::EndDisabled();
			}
			ImGui::EndMenu();
		}
		if(ImGui::BeginMenu("Tools")) {
			if(mode_ == Mode::Stadium && command_item("Matches")) {
				handle_menu_command(MenuCommand::Matches);
			}
			if(command_item("Settings")) {
				handle_menu_command(MenuCommand::Settings);
			}
			ImGui::EndMenu();
		}
		ImGui::PopStyleVar();
		ImGui::EndMenuBar();
	}

	/// Switches only the visible workspace while background sessions keep running.
	void set_mode(Mode mode) {
		if(mode_ == mode) {
			return;
		}
		mode_ = mode;
		selected_square_.reset();
		legal_targets_.clear();
	}

	/// Renders the interactive position analyser.
	void render_simulator(bool update_state) {
		if(update_state) {
			ensure_simulator_analysis();
		}
		const auto &visible = simulator_.visible_game();
		const float available_height = ImGui::GetContentRegionAvail().y;
		const float available_width = ImGui::GetContentRegionAvail().x;
		const auto layout =
			board_layout(available_width, available_height, simulator_board_fraction_);

		ImGui::PushStyleColor(ImGuiCol_ChildBg,
							  ImGui::GetStyleColorVec4(ImGuiCol_WindowBg));
		ImGui::BeginChild("simulator-left", {layout.board_size, 0.0F});
		ImGui::PopStyleColor();
		const bool viewing_history = simulator_.viewing_history();
		if(const auto square =
			   draw_board(visible, layout.board_size, flipped_, appearance_,
						  viewing_history ? std::nullopt : selected_square_,
						  viewing_history ? std::vector<chess::Move>{} : legal_targets_,
						  "simulator");
		   square && !viewing_history) {
			handle_simulator_square(*square);
		}
		if(const auto ply = render_history_slider(
			   "simulator-history", simulator_.game(), simulator_.viewed_ply())) {
			simulator_.view_ply(*ply);
			selected_square_.reset();
			legal_targets_.clear();
		}
		ImGui::EndChild();
		ImGui::SameLine(0.0F, 0.0F);
		render_board_splitter("simulator-board-splitter", layout, available_width,
							  available_height, simulator_board_fraction_);
		ImGui::SameLine(0.0F, 0.0F);
		ImGui::BeginChild("simulator-right", {0.0F, 0.0F},
						  ImGuiChildFlags_None,
						  ImGuiWindowFlags_NoScrollbar |
							  ImGuiWindowFlags_NoScrollWithMouse);
		const auto panel_heights =
			right_panel_heights(ImGui::GetContentRegionAvail().y,
								simulator_panel_ratios_);
		ImGui::TextUnformatted("Suggested moves");
		analysis_table(simulator_.display(),
					   simulator_.visible_game().board().getFen(),
					   panel_heights[0]);
		render_panel_splitter("simulator-panel-splitter-1", simulator_panel_ratios_,
							  0, panel_heights.content_height);
		ImGui::TextUnformatted("Engine analysis");
		render_engine_summary(simulator_.display(), panel_heights[1]);
		render_panel_splitter("simulator-panel-splitter-2", simulator_panel_ratios_,
							  1, panel_heights.content_height);
		ImGui::TextUnformatted("Board state");
		render_board_state(simulator_.visible_game(), panel_heights[2]);
		ImGui::EndChild();
	}

	/// Starts or refreshes simulator analysis only when its FEN changes.
	void ensure_simulator_analysis() {
		try {
			simulator_.update_analysis();
		} catch(const std::exception &error) {
			show_error(error.what());
		}
	}

	/// Handles piece selection, legal destinations, and default queen promotion.
	void handle_simulator_square(int square) {
		const auto piece = simulator_.game().board().at(chess::Square(square));
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
				return;
			}
		}
		if(piece != chess::Piece::NONE &&
		   piece.color() == simulator_.game().board().sideToMove()) {
			selected_square_ = square;
			legal_targets_ = simulator_.game().legal_moves_from(square);
		} else {
			selected_square_.reset();
			legal_targets_.clear();
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
			const std::string fen = game.board().getFen();
			const std::string pgn = game.movetext();
			ImGui::TextDisabled("FEN");
			ImGui::SameLine(0.0F, 4.0F);
			if(copy_button("fen")) {
				ImGui::SetClipboardText(fen.c_str());
			}
			ImGui::TextWrapped("%s", fen.c_str());
			ImGui::Separator();
			ImGui::TextDisabled("PGN");
			ImGui::SameLine(0.0F, 4.0F);
			if(copy_button("pgn")) {
				ImGui::SetClipboardText(pgn.c_str());
			}
			ImGui::TextWrapped("%s", pgn.c_str());
		}
		ImGui::EndChild();
	}

	/// Renders one visible UCI match with live analysis for the side to move.
	void render_stadium() {
		const auto &visible = stadium().visible_game();
		const bool human_input =
			!stadium().viewed_ply() && stadium().human_to_move();
		const float available_height = ImGui::GetContentRegionAvail().y;
		const float available_width = ImGui::GetContentRegionAvail().x;
		const auto layout =
			board_layout(available_width, available_height, stadium_board_fraction_);
		ImGui::PushStyleColor(ImGuiCol_ChildBg,
							  ImGui::GetStyleColorVec4(ImGuiCol_WindowBg));
		ImGui::BeginChild("stadium-left", {layout.board_size, 0.0F});
		ImGui::PopStyleColor();
		if(const auto square = draw_board(
			   visible, layout.board_size, flipped_, appearance_,
			   human_input ? selected_square_ : std::nullopt,
			   human_input ? legal_targets_ : std::vector<chess::Move>{},
			   "stadium");
		   square && human_input) {
			handle_stadium_square(*square);
		}
		if(const auto ply = render_history_slider(
			   "stadium-history", stadium().game(), stadium().viewed_ply())) {
			stadium().view_ply(*ply);
			selected_square_.reset();
			legal_targets_.clear();
		}
		ImGui::EndChild();
		ImGui::SameLine(0.0F, 0.0F);
		render_board_splitter("stadium-board-splitter", layout, available_width,
							  available_height, stadium_board_fraction_);
		ImGui::SameLine(0.0F, 0.0F);
		ImGui::BeginChild("stadium-right", {0.0F, 0.0F},
						  ImGuiChildFlags_None,
						  ImGuiWindowFlags_NoScrollbar |
							  ImGuiWindowFlags_NoScrollWithMouse);
		const auto panel_heights =
			right_panel_heights(ImGui::GetContentRegionAvail().y,
								stadium_panel_ratios_);
		const bool white_turn =
			stadium().game().board().sideToMove() == chess::Color::WHITE;
		ImGui::Text("%s to move",
					white_turn ? stadium().white_name().c_str()
							   : stadium().black_name().c_str());
		analysis_table(stadium().display(), stadium().game().board().getFen(),
					   panel_heights[0]);
		render_panel_splitter("stadium-panel-splitter-1", stadium_panel_ratios_, 0,
							  panel_heights.content_height);
		ImGui::TextUnformatted("Match");
		if(ImGui::BeginChild("match-state", {0.0F, panel_heights[1]},
							 ImGuiChildFlags_Borders)) {
			ImGui::TextUnformatted(stadium().display_name().c_str());
			ImGui::Text("%s  vs  %s", stadium().white_name().c_str(),
						stadium().black_name().c_str());
			if(stadium().clock_enabled()) {
				const auto white_clock = clock_text(stadium().white_remaining_ms());
				const auto black_clock = clock_text(stadium().black_remaining_ms());
				ImGui::Text("Clock: %s  /  %s", white_clock.c_str(),
							black_clock.c_str());
			}
			ImGui::Text("State: %s", stadium().status().c_str());
			ImGui::Text("Ply: %zu / %d", stadium().game().plies(),
						stadium().max_plies());
			ImGui::Text("Result: %s", stadium().result().c_str());
			if(!stadium().termination().empty()) {
				ImGui::Text("Termination: %s",
							stadium().termination().c_str());
			}
		}
		ImGui::EndChild();
		render_panel_splitter("stadium-panel-splitter-2", stadium_panel_ratios_, 1,
							  panel_heights.content_height);
		ImGui::TextUnformatted("Moves");
		render_board_state(stadium().visible_game(), panel_heights[2]);
		ImGui::EndChild();
	}

	/// Applies board clicks to the active Human seat with queen promotion as default.
	void handle_stadium_square(int square) {
		const auto piece = stadium().game().board().at(chess::Square(square));
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
				try {
					stadium().make_human_move(selected);
				} catch(const std::exception &error) {
					show_error(error.what());
				}
				selected_square_.reset();
				legal_targets_.clear();
				return;
			}
		}
		if(piece != chess::Piece::NONE &&
		   piece.color() == stadium().game().board().sideToMove()) {
			selected_square_ = square;
			legal_targets_ = stadium().game().legal_moves_from(square);
		} else {
			selected_square_.reset();
			legal_targets_.clear();
		}
	}

	/// Launches both UCI engines and initializes a fresh visible match.
	void start_stadium() {
		try {
			validate_match_side("White", stadium().white_config(),
								stadium().white_config().path.empty());
			validate_match_side("Black", stadium().black_config(),
								stadium().black_config().path.empty());
			stadium().start();
			selected_square_.reset();
			legal_targets_.clear();
		} catch(const std::exception &error) {
			stadium().stop();
			show_error(error.what());
		}
	}

	/// Stops the match and guarantees both UCI child processes are gone.
	void stop_stadium() {
		stadium().stop();
		selected_square_.reset();
		legal_targets_.clear();
	}

	/// Advances every background Stadium session independently of the visible mode.
	void update_stadium() {
		stadiums_.update_all();
		for(auto &error : stadiums_.take_errors()) {
			show_error(std::move(error));
		}
	}

	/// Returns the Stadium session selected for presentation and direct commands.
	StadiumSession &stadium() {
		return stadiums_.active();
	}

	const StadiumSession &stadium() const {
		return stadiums_.active();
	}

	/// Resizes the square board and right panel within a fixed application window.
	void render_board_splitter(const char *identifier, const BoardLayout &layout,
							   float available_width, float available_height,
							   float &board_fraction) {
		ImGui::PushID(identifier);
		ImGui::InvisibleButton("##splitter",
							   {layout.splitter_width, available_height});
		const bool hovered = ImGui::IsItemHovered();
		const bool active = ImGui::IsItemActive();
		if(hovered || active) {
			ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);
		}
		if(active) {
			const float requested =
				std::clamp(layout.board_size + ImGui::GetIO().MouseDelta.x,
						   layout.minimum_board_size, layout.maximum_board_size);
			board_fraction = requested / std::max(1.0F, available_width);
		}
		const ImVec2 minimum = ImGui::GetItemRectMin();
		const ImVec2 maximum = ImGui::GetItemRectMax();
		const float x = (minimum.x + maximum.x) * 0.5F;
		ImGui::GetWindowDrawList()->AddLine(
			{x, minimum.y + 4.0F}, {x, maximum.y - 4.0F},
			ImGui::GetColorU32(active ? ImGuiCol_SeparatorActive
									 : hovered ? ImGuiCol_SeparatorHovered
											   : ImGuiCol_Separator),
			active ? 2.0F : 1.0F);
		ImGui::PopID();
	}

	/// Converts normalized panel ratios into child heights after headers and splitters.
	RightPanelHeights right_panel_heights(
		float available_height, const std::array<float, 3> &ratios) const {
		constexpr float splitter_height = 8.0F;
		const float headers = ImGui::GetTextLineHeightWithSpacing() * 3.0F;
		const float content_height =
			std::max(3.0F, available_height - headers - splitter_height * 2.0F);
		const float sum = std::max(0.001F, ratios[0] + ratios[1] + ratios[2]);
		return {
			.values = {
				content_height * ratios[0] / sum,
				content_height * ratios[1] / sum,
				content_height * ratios[2] / sum,
			},
			.content_height = content_height,
		};
	}

	/// Resizes two adjacent right-side panels while preserving their combined height.
	void render_panel_splitter(const char *identifier,
							   std::array<float, 3> &ratios, std::size_t first,
							   float content_height) {
		constexpr float splitter_height = 8.0F;
		ImGui::PushID(identifier);
		ImGui::InvisibleButton("##splitter", {-1.0F, splitter_height});
		const bool hovered = ImGui::IsItemHovered();
		const bool active = ImGui::IsItemActive();
		if(hovered || active) {
			ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeNS);
		}
		if(active && first + 1 < ratios.size()) {
			const float pair = ratios[first] + ratios[first + 1];
			const float minimum =
				std::min(pair * 0.45F, 80.0F / std::max(1.0F, content_height));
			const float resized =
				std::clamp(ratios[first] +
							   ImGui::GetIO().MouseDelta.y /
								   std::max(1.0F, content_height),
						   minimum, pair - minimum);
			ratios[first + 1] = pair - resized;
			ratios[first] = resized;
		}
		const ImVec2 minimum = ImGui::GetItemRectMin();
		const ImVec2 maximum = ImGui::GetItemRectMax();
		const float y = (minimum.y + maximum.y) * 0.5F;
		ImGui::GetWindowDrawList()->AddLine(
			{minimum.x + 4.0F, y}, {maximum.x - 4.0F, y},
			ImGui::GetColorU32(active ? ImGuiCol_SeparatorActive
									 : hovered ? ImGuiCol_SeparatorHovered
											   : ImGuiCol_Separator),
			active ? 2.0F : 1.0F);
		ImGui::PopID();
	}

	/// Moves a read-only history pointer while preserving the complete live game.
	std::optional<std::size_t> render_history_slider(
		const char *identifier, const GameState &live,
		std::optional<std::size_t> view_ply) {
		ImGui::Spacing();
		const int maximum = static_cast<int>(live.plies());
		int position =
			view_ply ? std::min(static_cast<int>(*view_ply), maximum) : maximum;
		ImGui::PushID(identifier);
		const float width = std::max(1.0F, ImGui::GetContentRegionAvail().x);
		const float height = ImGui::GetFrameHeight();
		ImGui::InvisibleButton("##position", {width, height},
							   ImGuiButtonFlags_MouseButtonLeft);
		const ImVec2 minimum = ImGui::GetItemRectMin();
		const ImVec2 maximum_point = ImGui::GetItemRectMax();
		const float knob_radius = std::max(6.0F, height * 0.24F);
		const float track_left = minimum.x + knob_radius;
		const float track_right = maximum_point.x - knob_radius;
		const float pointer_ratio = std::clamp(
			(ImGui::GetIO().MousePos.x - track_left) /
				std::max(1.0F, track_right - track_left),
			0.0F, 1.0F);
		bool changed = false;
		if(maximum > 0 && ImGui::IsItemActive()) {
			const int selected =
				std::clamp(static_cast<int>(std::lround(pointer_ratio * maximum)), 0,
						   maximum);
			if(selected != position) {
				position = selected;
				changed = true;
			}
		}
		if(ImGui::IsItemHovered()) {
			ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
		}

		const auto &style = ImGui::GetStyle();
		const float center_y = (minimum.y + maximum_point.y) * 0.5F;
		const float position_ratio = maximum > 0
			? static_cast<float>(position) / static_cast<float>(maximum)
			: 1.0F;
		const float visual_ratio =
			maximum > 0 && ImGui::IsItemActive() ? pointer_ratio : position_ratio;
		const float knob_x =
			track_left + (track_right - track_left) * visual_ratio;
		auto *draw = ImGui::GetWindowDrawList();
		const ImU32 track_color = ImGui::GetColorU32(ImGuiCol_FrameBg);
		const ImU32 fill_color = ImGui::GetColorU32(
			maximum == 0 ? ImGuiCol_TextDisabled : ImGuiCol_SliderGrab);
		const ImU32 knob_color = ImGui::GetColorU32(
			ImGui::IsItemActive() ? ImGuiCol_SliderGrabActive : ImGuiCol_SliderGrab);
		const float track_half = std::max(2.0F, height * 0.09F);
		draw->AddRectFilled({track_left, center_y - track_half},
							{track_right, center_y + track_half}, track_color,
							track_half);
		draw->AddRectFilled({track_left, center_y - track_half},
							{knob_x, center_y + track_half}, fill_color,
							track_half);
		draw->AddCircleFilled({knob_x, center_y}, knob_radius, knob_color, 24);
		draw->AddCircle({knob_x, center_y}, knob_radius,
						ImGui::GetColorU32(ImGuiCol_Border), 24,
						std::max(1.0F, style.FrameBorderSize));
		ImGui::PopID();
		if(!changed) {
			return std::nullopt;
		}
		return static_cast<std::size_t>(position);
	}

	/// Opens the settings editor with mode-specific engine and match values.
	void open_settings(Mode mode) {
		settings_mode_ = mode;
		appearance_edit_ = appearance_;
		theme_edit_ = theme_;
		if(mode == Mode::Simulator) {
			simulator_edit_ = simulator_.config();
		} else {
			white_edit_ = stadium().white_config();
			black_edit_ = stadium().black_config();
			white_human_edit_ = white_edit_.path.empty();
			black_human_edit_ = black_edit_.path.empty();
			match_name_edit_ = stadium().name();
			clock_minutes_edit_ =
				static_cast<int>(stadium().clock_initial_ms() / 60'000);
			clock_increment_seconds_edit_ =
				static_cast<int>(stadium().clock_increment_ms() / 1000);
			delay_edit_ = stadium().display_delay_ms();
			max_plies_edit_ = stadium().max_plies();
		}
		settings_open_ = true;
	}

	/// Adds one executable to the reusable UCI engine registry.
	void import_uci_engine() {
		if(const auto path =
			   file_dialog(false, L"Executables\0*.exe\0All files\0*.*\0", L"exe")) {
			try {
				EngineConfig engine;
				engine.path = *path;
				engine.name = path->stem().string();
				registry_.add(std::move(engine));
				save_settings();
			} catch(const std::exception &error) {
				show_error(error.what());
			}
		}
	}

	/// Imports one PGN main line into Simulator and returns the view to live.
	void import_simulator_pgn() {
		if(const auto path =
			   file_dialog(false, L"PGN files\0*.pgn\0All files\0*.*\0", L"pgn")) {
			try {
				std::ifstream input(*path);
				std::ostringstream document;
				document << input.rdbuf();
				simulator_.import_pgn(document.str());
				selected_square_.reset();
				legal_targets_.clear();
			} catch(const std::exception &error) {
				show_error(error.what());
			}
		}
	}

	/// Removes one Simulator move and resets its non-destructive history pointer.
	void undo_simulator() {
		if(simulator_.undo()) {
			selected_square_.reset();
			legal_targets_.clear();
		}
	}

	/// Starts or stops live Simulator analysis.
	void toggle_simulator_analysis() {
		if(simulator_.analysis_open()) {
			simulator_.stop_analysis();
		} else {
			simulator_.start_analysis();
		}
	}

	/// Pauses or resumes Stadium without destroying either engine process.
	void toggle_stadium_pause() {
		stadium().toggle_pause();
	}

	/// Opens a focused FEN editor for the active mode.
	void open_position_editor() {
		position_mode_ = mode_;
		position_edit_ = mode_ == Mode::Simulator
							 ? simulator_.start_fen()
							 : stadium().start_fen();
		position_editor_open_ = true;
	}

	/// Resets the active board from its configured start FEN.
	void reset_active_board() {
		try {
			if(mode_ == Mode::Simulator) {
				simulator_.reset();
				selected_square_.reset();
				legal_targets_.clear();
			} else {
				stadium().reset();
				selected_square_.reset();
				legal_targets_.clear();
			}
		} catch(const std::exception &error) {
			show_error(error.what());
		}
	}

	/// Validates and stores the start FEN without changing the current board.
	void render_position_editor() {
		if(position_editor_open_) {
			ImGui::OpenPopup("FEN");
			position_editor_open_ = false;
		}
		ImGui::SetNextWindowSize({680.0F, 210.0F}, ImGuiCond_Appearing);
		if(!ImGui::BeginPopupModal("FEN", nullptr,
								   ImGuiWindowFlags_NoResize |
									   ImGuiWindowFlags_NoSavedSettings)) {
			return;
		}
		ImGui::TextUnformatted(position_mode_ == Mode::Simulator
								  ? "Simulator start position"
								  : "Stadium start position");
		ImGui::SetNextItemWidth(-1.0F);
		const bool submitted = ImGui::InputTextWithHint(
			"##position-fen", "FEN or startpos", &position_edit_,
			ImGuiInputTextFlags_EnterReturnsTrue);
		if(ImGui::Button("Apply", {100.0F, 0.0F}) || submitted) {
			try {
				const std::string fen = position_edit_.empty() ? "startpos" : position_edit_;
				GameState validation;
				validation.reset(fen);
				static_cast<void>(validation);
				if(position_mode_ == Mode::Simulator) {
					simulator_.set_start_fen(fen);
				} else {
					stadium().set_start_fen(fen);
				}
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

	/// Presents one engine's generic process and UCI controls.
	void engine_editor(const char *identifier, EngineConfig &config,
					   bool include_name = true) {
		ImGui::PushID(identifier);
		ImGui::SeparatorText("Engine");
		if(!registry_.engines().empty()) {
			const std::string preview = config.name.empty()
				? config.path.filename().string()
				: config.name;
			if(ImGui::BeginCombo("Imported engine",
								 preview.empty() ? "Select engine"
												 : preview.c_str())) {
				for(const auto &engine : registry_.engines()) {
					const bool selected = engine.path == config.path;
					const std::string label = engine.name.empty()
						? engine.path.filename().string()
						: engine.name;
					if(ImGui::Selectable(label.c_str(), selected)) {
						config = engine;
					}
					if(selected) {
						ImGui::SetItemDefaultFocus();
					}
				}
				ImGui::EndCombo();
			}
		}
		std::string path = config.path.string();
		ImGui::TextUnformatted("Executable");
		const float browse_width =
			ImGui::CalcTextSize("Browse").x + ImGui::GetStyle().FramePadding.x * 2.0F;
		const float path_width =
			std::max(80.0F, ImGui::GetContentRegionAvail().x - browse_width -
							 ImGui::GetStyle().ItemSpacing.x);
		ImGui::SetNextItemWidth(path_width);
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
		if(include_name) {
			ImGui::TextUnformatted("Display name");
			ImGui::SetNextItemWidth(-1.0F);
			ImGui::InputText("##name", &config.name);
		}
		ImGui::TextUnformatted("Device (managed UCI option)");
		const char *devices[] = {"auto", "cpu", "cuda"};
		int device = config.device == "cpu" ? 1 : config.device == "cuda" ? 2 : 0;
		if(ImGui::Combo("##device", &device, devices, 3)) {
			config.device = devices[device];
		}
		ImGui::Spacing();
		ImGui::SeparatorText("Search");
		step_input("Move time (ms)", config.movetime_ms);
		step_input("Node / simulation limit", config.node_limit);
		step_input("Analysis lines (MultiPV)", config.multipv);
		step_input("Display update (ms)", config.progress_interval_ms);
		config.movetime_ms = std::max(0, config.movetime_ms);
		config.multipv = std::max(1, config.multipv);
		config.progress_interval_ms = std::max(50, config.progress_interval_ms);
		ImGui::Spacing();
		ImGui::SeparatorText("UCI options");
		ImGui::TextDisabled(
			"JSON values become setoption commands after launch. Device and MultiPV are managed above.");
		const float options_height =
			std::max(180.0F, ImGui::GetTextLineHeightWithSpacing() * 10.0F);
		ImGui::InputTextMultiline("##options", &config.options,
								  {-1.0F, options_height});
		ImGui::Spacing();
		if(ImGui::CollapsingHeader("Launch arguments")) {
			ImGui::TextDisabled(
				"Optional process arguments passed before the UCI handshake.");
			ImGui::SetNextItemWidth(-1.0F);
			ImGui::InputText("##arguments", &config.arguments);
		}
		ImGui::PopID();
	}

	/// Presents either a Human seat or the complete UCI configuration for one side.
	void match_side_editor(const char *identifier, EngineConfig &config,
						   bool &human) {
		ImGui::PushID(identifier);
		ImGui::TextUnformatted("Name");
		ImGui::SetNextItemWidth(-1.0F);
		ImGui::InputText("##participant-name", &config.name);
		toggle_switch("Human", human);
		ImGui::BeginDisabled(human);
		engine_editor("uci", config, false);
		ImGui::EndDisabled();
		if(human) {
			ImGui::TextDisabled(
				"Moves are entered directly on the board when this side is to move.");
		}
		ImGui::PopID();
	}

	/// Requires stable PGN identity for every configured UCI match participant.
	void validate_match_side(const char *side, const EngineConfig &config,
							 bool human) const {
		if(config.name.empty()) {
			throw std::invalid_argument(std::string(side) +
									   " participant name is required for PGN");
		}
		if(!human && config.path.empty()) {
			throw std::invalid_argument(std::string(side) +
									   " engine executable is required");
		}
	}

	/// Creates a blank Human-versus-Human draft for the next stable match id.
	void begin_new_match() {
		new_match_name_.clear();
		new_white_ = {};
		new_black_ = {};
		new_white_human_ = true;
		new_black_human_ = true;
		new_clock_minutes_ = 0;
		new_increment_seconds_ = 0;
		new_delay_ = 250;
		new_max_plies_ = 240;
		match_create_mode_ = true;
	}

	/// Renders the Stadium-only manager for creating, entering, and closing matches.
	void render_matches() {
		if(matches_open_) {
			ImGui::OpenPopup("Matches");
			matches_open_ = false;
		}
		ImGui::SetNextWindowSize({900.0F, 650.0F}, ImGuiCond_Appearing);
		if(!ImGui::BeginPopupModal("Matches", nullptr,
								   ImGuiWindowFlags_NoSavedSettings)) {
			return;
		}

		if(match_create_mode_) {
			const auto next_id = stadiums_.next_id();
			const auto side_name = [](const EngineConfig &config, bool human) {
				if(!config.name.empty()) {
					return config.name;
				}
				if(human) {
					return std::string("Human");
				}
				const auto stem = config.path.stem().string();
				return stem.empty() ? std::string("Engine") : stem;
			};
			const std::string default_name =
				"#" + std::to_string(next_id) + " " +
				side_name(new_white_, new_white_human_) + " vs. " +
				side_name(new_black_, new_black_human_);
			ImGui::TextUnformatted("New match");
			ImGui::SetNextItemWidth(-1.0F);
			ImGui::InputTextWithHint("##new-match-name", default_name.c_str(),
									 &new_match_name_);
			ImGui::BeginChild("new-match-content", {0.0F, -48.0F},
							  ImGuiChildFlags_None,
							  ImGuiWindowFlags_AlwaysVerticalScrollbar);
			if(ImGui::BeginTabBar("new-match-tabs")) {
				if(ImGui::BeginTabItem("White")) {
					match_side_editor("new-white", new_white_,
									  new_white_human_);
					ImGui::EndTabItem();
				}
				if(ImGui::BeginTabItem("Black")) {
					match_side_editor("new-black", new_black_,
									  new_black_human_);
					ImGui::EndTabItem();
				}
				if(ImGui::BeginTabItem("Match")) {
					step_input("Initial time (minutes)", new_clock_minutes_);
					step_input("Increment (seconds)", new_increment_seconds_);
					ImGui::TextDisabled("An initial time of 0 disables the clock.");
					step_input("Display delay (ms)", new_delay_);
					step_input("Maximum plies", new_max_plies_);
					new_clock_minutes_ = std::max(0, new_clock_minutes_);
					new_increment_seconds_ =
						std::max(0, new_increment_seconds_);
					new_delay_ = std::max(0, new_delay_);
					new_max_plies_ = std::max(1, new_max_plies_);
					ImGui::EndTabItem();
				}
				ImGui::EndTabBar();
			}
			ImGui::EndChild();
			if(ImGui::Button("Create", {100.0F, 0.0F})) {
				try {
					validate_match_side("White", new_white_,
										new_white_human_);
					validate_match_side("Black", new_black_,
										new_black_human_);
					if(!new_white_human_) {
						static_cast<void>(
							additional_uci_options(new_white_.options));
					} else {
						new_white_.path.clear();
					}
					if(!new_black_human_) {
						static_cast<void>(
							additional_uci_options(new_black_.options));
					} else {
						new_black_.path.clear();
					}
					const auto index = stadiums_.create_session();
					auto &session = stadiums_.at(index);
					session.set_name(new_match_name_);
					session.set_configs(new_white_, new_black_);
					session.set_clock(
						static_cast<std::int64_t>(new_clock_minutes_) * 60'000,
						static_cast<std::int64_t>(new_increment_seconds_) * 1000);
					session.set_match_limits(new_delay_, new_max_plies_);
					selected_square_.reset();
					legal_targets_.clear();
					match_create_mode_ = false;
					save_settings();
					ImGui::CloseCurrentPopup();
				} catch(const std::exception &error) {
					show_error(error.what());
				}
			}
			ImGui::SameLine();
			if(ImGui::Button("Back", {100.0F, 0.0F})) {
				match_create_mode_ = false;
			}
			ImGui::EndPopup();
			return;
		}

		ImGui::Text("Matches: %zu", stadiums_.size());
		std::optional<std::size_t> close_index;
		if(ImGui::BeginTable(
			   "match-list", 5,
			   ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
				   ImGuiTableFlags_Resizable | ImGuiTableFlags_SizingStretchProp,
			   {0.0F, -54.0F})) {
			ImGui::TableSetupColumn("Name");
			ImGui::TableSetupColumn("White");
			ImGui::TableSetupColumn("Black");
			ImGui::TableSetupColumn("State");
			ImGui::TableSetupColumn("Actions");
			ImGui::TableHeadersRow();
			for(std::size_t index = 0; index < stadiums_.size(); ++index) {
				auto &session = stadiums_.at(index);
				ImGui::PushID(static_cast<int>(session.id()));
				ImGui::TableNextRow();
				ImGui::TableSetColumnIndex(0);
				ImGui::TextUnformatted(session.display_name().c_str());
				ImGui::TableSetColumnIndex(1);
				ImGui::TextUnformatted(session.white_name().c_str());
				ImGui::TableSetColumnIndex(2);
				ImGui::TextUnformatted(session.black_name().c_str());
				ImGui::TableSetColumnIndex(3);
				ImGui::TextUnformatted(session.status().c_str());
				ImGui::TableSetColumnIndex(4);
				if(ImGui::SmallButton(
					   index == stadiums_.active_index() ? "Current" : "Enter")) {
					stadiums_.select(index);
					selected_square_.reset();
					legal_targets_.clear();
					ImGui::CloseCurrentPopup();
				}
				ImGui::SameLine();
				if(ImGui::SmallButton("Close")) {
					close_index = index;
				}
				ImGui::PopID();
			}
			ImGui::EndTable();
		}
		if(close_index) {
			stadiums_.close(*close_index);
			selected_square_.reset();
			legal_targets_.clear();
		}
		if(ImGui::Button("New", {120.0F, 0.0F})) {
			begin_new_match();
		}
		ImGui::SameLine();
		if(ImGui::Button("Done", {100.0F, 0.0F})) {
			ImGui::CloseCurrentPopup();
		}
		ImGui::EndPopup();
	}

	/// Renders appearance controls shared by both modes.
	void appearance_editor(Appearance &appearance, Theme &theme) {
		const char *themes[] = {"Dark", "Light"};
		int selected_theme = theme == Theme::Light ? 1 : 0;
		ImGui::TextUnformatted("Theme");
		if(ImGui::Combo("##theme", &selected_theme, themes, 2)) {
			theme = selected_theme == 1 ? Theme::Light : Theme::Dark;
		}
		ImGui::SliderFloat("Font size", &appearance.font_size, 14.0F, 36.0F, "%.0f px");
		ImGui::Checkbox("Scale font with window", &appearance.auto_font_scale);
		ImGui::TextUnformatted("Pieces");
		if(ImGui::BeginCombo("##pieces", appearance.piece_style.c_str())) {
			for(std::size_t index = 0; index < piece_style_count(); ++index) {
				const auto name = piece_style_name(index);
				const bool selected = appearance.piece_style == name;
				if(ImGui::Selectable(name.data(), selected)) {
					appearance.piece_style = name;
				}
				if(selected) {
					ImGui::SetItemDefaultFocus();
				}
			}
			ImGui::EndCombo();
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
		ImGui::SameLine();
		if(ImGui::Button("Tournament")) {
			appearance.light = color_vector(color32(0.94F, 0.85F, 0.68F));
			appearance.dark = color_vector(color32(0.64F, 0.43F, 0.25F));
		}
		ImGui::SameLine();
		if(ImGui::Button("Walnut")) {
			appearance.light = color_vector(color32(0.78F, 0.69F, 0.56F));
			appearance.dark = color_vector(color32(0.37F, 0.25F, 0.20F));
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
		ImGui::BeginChild("settings-content", {0.0F, -48.0F}, ImGuiChildFlags_None,
						  ImGuiWindowFlags_AlwaysVerticalScrollbar);
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
			if(ImGui::BeginTabItem("White")) {
				match_side_editor("white-edit", white_edit_,
								  white_human_edit_);
				ImGui::EndTabItem();
			}
			if(ImGui::BeginTabItem("Black")) {
				match_side_editor("black-edit", black_edit_,
								  black_human_edit_);
				ImGui::EndTabItem();
			}
			if(ImGui::BeginTabItem("Match")) {
				ImGui::InputTextWithHint("Name", stadium().display_name().c_str(),
									 &match_name_edit_);
				step_input("Initial time (minutes)", clock_minutes_edit_);
				step_input("Increment (seconds)", clock_increment_seconds_edit_);
				ImGui::TextDisabled("An initial time of 0 disables the clock.");
				step_input("Display delay (ms)", delay_edit_);
				step_input("Maximum plies", max_plies_edit_);
				clock_minutes_edit_ = std::max(0, clock_minutes_edit_);
				clock_increment_seconds_edit_ =
					std::max(0, clock_increment_seconds_edit_);
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
		ImGui::EndChild();
		ImGui::Separator();
		if(ImGui::Button("Apply", {100.0F, 0.0F})) {
			try {
				const auto primary_options =
					settings_mode_ == Mode::Simulator || !white_human_edit_
						? additional_uci_options(
							  settings_mode_ == Mode::Simulator
								  ? simulator_edit_.options
								  : white_edit_.options)
						: nlohmann::json::object();
				if(settings_mode_ == Mode::Stadium) {
					validate_match_side("White", white_edit_,
										white_human_edit_);
					validate_match_side("Black", black_edit_,
										black_human_edit_);
					const auto secondary_options =
						black_human_edit_
							? nlohmann::json::object()
							: additional_uci_options(black_edit_.options);
					static_cast<void>(secondary_options);
				}
				static_cast<void>(primary_options);
				if(settings_mode_ == Mode::Simulator) {
					simulator_.set_config(simulator_edit_);
				} else {
					stadium().stop();
					if(white_human_edit_) {
						white_edit_.path.clear();
					}
					if(black_human_edit_) {
						black_edit_.path.clear();
					}
					stadium().set_configs(white_edit_, black_edit_);
					stadium().set_name(match_name_edit_);
					stadium().set_clock(
						static_cast<std::int64_t>(clock_minutes_edit_) * 60'000,
						static_cast<std::int64_t>(
							clock_increment_seconds_edit_) *
							1000);
					stadium().set_match_limits(delay_edit_, max_plies_edit_);
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
				  const std::string &black,
				  const std::string &result_override = "",
				  const std::string &termination_override = "") {
		if(const auto path =
			   file_dialog(true, L"PGN files\0*.pgn\0All files\0*.*\0", L"pgn")) {
			try {
				std::ofstream output(*path, std::ios::trunc);
				if(!output) {
					throw std::runtime_error("could not open PGN output");
				}
				output << game.pgn(white, black, result_override,
								   termination_override);
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

	std::filesystem::path settings_path_;
	Mode mode_ = Mode::Simulator;
	Mode settings_mode_ = Mode::Simulator;
	Mode position_mode_ = Mode::Simulator;
	SimulatorWorkspace simulator_;
	StadiumWorkspace stadiums_;
	Registry registry_;
	EngineConfig simulator_edit_;
	EngineConfig white_edit_;
	EngineConfig black_edit_;
	EngineConfig new_white_;
	EngineConfig new_black_;
	Appearance appearance_;
	Appearance appearance_edit_;
	Theme theme_ = Theme::Dark;
	Theme theme_edit_ = Theme::Dark;
	float simulator_board_fraction_ = 0.60F;
	float stadium_board_fraction_ = 0.60F;
	std::array<float, 3> simulator_panel_ratios_ = {0.42F, 0.23F, 0.35F};
	std::array<float, 3> stadium_panel_ratios_ = {0.42F, 0.23F, 0.35F};
	bool flipped_ = false;
	bool settings_open_ = false;
	bool matches_open_ = false;
	bool match_create_mode_ = false;
	bool position_editor_open_ = false;
	bool error_open_ = false;
	bool white_human_edit_ = true;
	bool black_human_edit_ = true;
	bool new_white_human_ = true;
	bool new_black_human_ = true;
	std::optional<int> selected_square_;
	std::vector<chess::Move> legal_targets_;
	std::string position_edit_ = "startpos";
	std::string match_name_edit_;
	std::string new_match_name_;
	int clock_minutes_edit_ = 0;
	int clock_increment_seconds_edit_ = 0;
	int new_clock_minutes_ = 0;
	int new_increment_seconds_ = 0;
	int new_delay_ = 250;
	int new_max_plies_ = 240;
	int delay_edit_ = 250;
	int max_plies_edit_ = 240;
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
	Clock::time_point defer_drawing_until{};
};

constexpr auto frame_interval = std::chrono::microseconds(8333);

/// Draws main content and overlays through the same bounded 120 Hz cadence.
void refresh_window(GLFWwindow *window, bool update_state) {
	auto *context =
		static_cast<RefreshContext *>(glfwGetWindowUserPointer(window));
	if(context == nullptr || context->rendering) {
		return;
	}
	int width = 0;
	int height = 0;
	int window_width = 0;
	int window_height = 0;
	glfwGetFramebufferSize(window, &width, &height);
	glfwGetWindowSize(window, &window_width, &window_height);
	if(width <= 0 || height <= 0 || window_width <= 0 || window_height <= 0) {
		return;
	}
	const auto now = Clock::now();
	if(context->last_render != Clock::time_point{} &&
	   now - context->last_render < frame_interval) {
		return;
	}
	context->rendering = true;
	try {
		context->render(update_state);
		context->last_render = Clock::now();
		context->rendering = false;
	} catch(...) {
		context->rendering = false;
		throw;
	}
}

/// Defers OpenGL submissions while Windows continuously moves or resizes the window.
void defer_window_drawing(GLFWwindow *window) {
	auto *context =
		static_cast<RefreshContext *>(glfwGetWindowUserPointer(window));
	if(context != nullptr) {
		context->defer_drawing_until =
			Clock::now() + std::chrono::milliseconds(100);
	}
}

/// Marks logical size changes without drawing inside the native callback.
void window_resized(GLFWwindow *window, int, int) {
	defer_window_drawing(window);
}

/// Marks window moves without drawing inside the native callback.
void window_moved(GLFWwindow *window, int, int) {
	defer_window_drawing(window);
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
	glfwWindowHint(GLFW_SAMPLES, 4);
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
	glEnable(GL_MULTISAMPLE);

	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	auto &io = ImGui::GetIO();
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
	io.IniFilename = nullptr;
	if(const auto font = system_font_path()) {
		io.Fonts->AddFontFromFileTTF(font->string().c_str(), 32.0F);
	} else {
		ImFontConfig font_config;
		font_config.SizePixels = 32.0F;
		io.Fonts->AddFontDefault(&font_config);
	}
	apply_style(Theme::Dark);
	ImGui_ImplGlfw_InitForOpenGL(window, true);
	ImGui_ImplOpenGL3_Init("#version 330 core");

	int result = 0;
	try {
		Application application(argc, argv);
		RefreshContext refresh;
		refresh.render = [&](bool update_state) {
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
			glFinish();
			glfwSwapBuffers(window);
		};
		glfwSetWindowUserPointer(window, &refresh);
		glfwSetWindowSizeCallback(window, window_resized);
		glfwSetWindowPosCallback(window, window_moved);
		auto next_frame = Clock::now();
		while(!glfwWindowShouldClose(window)) {
			glfwPollEvents();
			if(Clock::now() >= refresh.defer_drawing_until) {
				refresh_window(window, true);
			}
			next_frame += frame_interval;
			const auto now = Clock::now();
			if(next_frame > now) {
				std::this_thread::sleep_until(next_frame);
			} else {
				next_frame = now;
			}
		}
		glfwSetWindowPosCallback(window, nullptr);
		glfwSetWindowSizeCallback(window, nullptr);
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
