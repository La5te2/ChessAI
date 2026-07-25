// Declares the native Gadidae OpenGL application entry point.
#pragma once

namespace gadidae::graphics {

/// Creates the OpenGL window, runs Simulator or Stadium, and returns an exit code.
int run_application(int argc, char **argv);

} // namespace gadidae::graphics
