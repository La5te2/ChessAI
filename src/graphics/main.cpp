// Provides the platform-neutral executable entry point for the native GUI.
#include "graphics/application.hpp"
#include <exception>
#include <iostream>
#include <string>
#include <vector>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <shellapi.h>
#endif

namespace {

#ifdef _WIN32
/// Converts one UTF-16 Windows argument to the UTF-8 encoding used by the core.
std::string utf8_argument(const wchar_t *value) {
	const int length = WideCharToMultiByte(CP_UTF8, 0, value, -1, nullptr, 0, nullptr, nullptr);
	if (length <= 0) {
		throw std::runtime_error("could not decode the Windows command line");
	}
	std::string output(static_cast<std::size_t>(length), '\0');
	WideCharToMultiByte(CP_UTF8, 0, value, -1, output.data(), length, nullptr, nullptr);
	output.pop_back();
	return output;
}
#endif

/// Runs the native interface and presents startup failures on the active platform.
int run_graphics(int argc, char **argv) {
	try {
		return gadidae::graphics::run_application(argc, argv);
	} catch (const std::exception &error) {
#ifdef _WIN32
		const std::string message = std::string("Gadidae graphics error:\n") + error.what();
		MessageBoxA(nullptr, message.c_str(), "Gadidae", MB_OK | MB_ICONERROR);
#else
		std::cerr << "Gadidae GUI error: " << error.what() << '\n';
#endif
		return 1;
	}
}

} // namespace

#ifdef _WIN32
/// Starts the Windows GUI without creating a companion console window.
int WINAPI wWinMain(HINSTANCE, HINSTANCE, PWSTR, int) {
	int argument_count = 0;
	wchar_t **wide_arguments = CommandLineToArgvW(GetCommandLineW(), &argument_count);
	if (wide_arguments == nullptr) {
		MessageBoxA(nullptr, "Could not parse the Windows command line.", "Gadidae", MB_OK | MB_ICONERROR);
		return 1;
	}
	try {
		std::vector<std::string> arguments;
		arguments.reserve(static_cast<std::size_t>(argument_count));
		for (int index = 0; index < argument_count; ++index) {
			arguments.push_back(utf8_argument(wide_arguments[index]));
		}
		std::vector<char *> pointers;
		pointers.reserve(arguments.size());
		for (auto &argument : arguments) {
			pointers.push_back(argument.data());
		}
		LocalFree(wide_arguments);
		wide_arguments = nullptr;
		return run_graphics(static_cast<int>(pointers.size()), pointers.data());
	} catch (const std::exception &error) {
		if (wide_arguments != nullptr) {
			LocalFree(wide_arguments);
		}
		const std::string message = std::string("Gadidae graphics error:\n") + error.what();
		MessageBoxA(nullptr, message.c_str(), "Gadidae", MB_OK | MB_ICONERROR);
		return 1;
	}
}
#else
/// Starts the native interface on Unix-like platforms.
int main(int argc, char **argv) {
	return run_graphics(argc, argv);
}
#endif
