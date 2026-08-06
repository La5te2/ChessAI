cmake_minimum_required(VERSION 3.22)

project(GadidaeCpp LANGUAGES C CXX)

option(GADIDAE_BUILD_GADUS "Build the Gadus architecture" ON)
option(GADIDAE_BUILD_MELANO "Build the Melano architecture" ON)
option(GADIDAE_BUILD_ELEGINUS "Build the Eleginus architecture" ON)
option(GADIDAE_BUILD_ELEGINUS_TRAINING "Build Eleginus LibTorch training tools" ON)
option(GADIDAE_BUILD_GRAPHICS "Build the GLFW/OpenGL desktop application" ON)

if(NOT GADIDAE_BUILD_GADUS AND NOT GADIDAE_BUILD_MELANO AND
	NOT GADIDAE_BUILD_ELEGINUS AND NOT GADIDAE_BUILD_GRAPHICS)
	message(FATAL_ERROR "At least one Gadidae architecture must be enabled")
endif()

set(GADIDAE_NEEDS_TORCH OFF)
if(GADIDAE_BUILD_GADUS OR GADIDAE_BUILD_MELANO OR
	(GADIDAE_BUILD_ELEGINUS AND GADIDAE_BUILD_ELEGINUS_TRAINING))
	set(GADIDAE_NEEDS_TORCH ON)
endif()
set(GADIDAE_NEEDS_HDF5 OFF)
if(GADIDAE_BUILD_GADUS OR GADIDAE_BUILD_MELANO OR
	(GADIDAE_BUILD_ELEGINUS AND GADIDAE_BUILD_ELEGINUS_TRAINING))
	set(GADIDAE_NEEDS_HDF5 ON)
endif()
set(GADIDAE_NEEDS_ZLIB OFF)
if(GADIDAE_BUILD_GADUS OR GADIDAE_BUILD_MELANO OR GADIDAE_BUILD_GRAPHICS OR
	(GADIDAE_BUILD_ELEGINUS AND GADIDAE_BUILD_ELEGINUS_TRAINING))
	set(GADIDAE_NEEDS_ZLIB ON)
endif()

if(MSVC AND CMAKE_GENERATOR MATCHES "Ninja")
	set(CMAKE_CL_SHOWINCLUDES_PREFIX "注意: 包含文件:  ")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# LibTorch enables CUDA only to validate its imported runtime. No project target compiles .cu
# sources, so keep that probe on the broadly supported CUDA C++17 dialect.
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(GADIDAE_API_DIR "${CMAKE_SOURCE_DIR}/api" CACHE PATH "Local dependency directory")
set(GADIDAE_TORCH_DIR "${GADIDAE_API_DIR}/libtorch" CACHE PATH "LibTorch root")
set(GADIDAE_HDF5_DIR "${GADIDAE_API_DIR}/hdf5" CACHE PATH "HDF5 install root")
set(GADIDAE_ZLIB_DIR "${GADIDAE_API_DIR}/zlib" CACHE PATH "zlib install root")

function(gadidaelockvalue name output)
	file(STRINGS "${GADIDAE_API_DIR}/versions.env" lines REGEX "^${name}=")
	list(LENGTH lines count)
	if(NOT count EQUAL 1)
		message(FATAL_ERROR "Dependency lock must define ${name} exactly once")
	endif()
	list(GET lines 0 line)
	string(REGEX REPLACE "^[^=]+=" "" value "${line}")
	set(${output} "${value}" PARENT_SCOPE)
endfunction()

if(GADIDAE_NEEDS_TORCH)
	gadidaelockvalue(TORCH_VERSION GADIDAE_TORCH_VERSION)
endif()
if(GADIDAE_NEEDS_HDF5)
	gadidaelockvalue(HDF5_VERSION GADIDAE_HDF5_VERSION)
endif()
if(GADIDAE_NEEDS_ZLIB)
	gadidaelockvalue(ZLIB_VERSION GADIDAE_ZLIB_VERSION)
endif()

if(GADIDAE_NEEDS_TORCH AND EXISTS "${GADIDAE_TORCH_DIR}/share/cmake/Torch")
	unset(TORCH_LIBRARY CACHE)
	unset(c10_LIBRARY CACHE)
	unset(kineto_LIBRARY CACHE)
	unset(CAFFE2_NVRTC_LIBRARY CACHE)
	unset(C10_CUDA_LIBRARY CACHE)
	set(Torch_DIR "${GADIDAE_TORCH_DIR}/share/cmake/Torch" CACHE PATH "Torch CMake package" FORCE)
	set(Caffe2_DIR "${GADIDAE_TORCH_DIR}/share/cmake/Caffe2" CACHE PATH "Caffe2 CMake package" FORCE)
	list(PREPEND CMAKE_PREFIX_PATH "${GADIDAE_TORCH_DIR}")
endif()
if(GADIDAE_NEEDS_HDF5 AND EXISTS "${GADIDAE_HDF5_DIR}")
	list(PREPEND CMAKE_PREFIX_PATH "${GADIDAE_HDF5_DIR}")
	set(HDF5_ROOT "${GADIDAE_HDF5_DIR}")
endif()
if(GADIDAE_NEEDS_ZLIB AND EXISTS "${GADIDAE_ZLIB_DIR}")
	list(PREPEND CMAKE_PREFIX_PATH "${GADIDAE_ZLIB_DIR}")
	set(ZLIB_ROOT "${GADIDAE_ZLIB_DIR}")
endif()

if(GADIDAE_NEEDS_TORCH)
	find_package(Torch ${GADIDAE_TORCH_VERSION} EXACT REQUIRED)
endif()
if(GADIDAE_NEEDS_HDF5)
	find_package(HDF5 ${GADIDAE_HDF5_VERSION} EXACT REQUIRED COMPONENTS C)
endif()
if(GADIDAE_NEEDS_ZLIB)
	find_package(ZLIB ${GADIDAE_ZLIB_VERSION} EXACT REQUIRED)
endif()
find_package(Threads REQUIRED)

if(MSVC AND GADIDAE_NEEDS_ZLIB)
	find_library(
		GADIDAE_ZLIB_STATIC
		NAMES zlibstatic
		PATHS "${GADIDAE_ZLIB_DIR}/lib"
		NO_DEFAULT_PATH
		REQUIRED
	)
endif()

enable_testing()

if(MSVC AND GADIDAE_NEEDS_TORCH)
	file(GLOB TORCH_DLLS "${GADIDAE_TORCH_DIR}/lib/*.dll")
	if(NOT TORCH_DLLS)
		message(FATAL_ERROR "No LibTorch runtime DLLs found in ${GADIDAE_TORCH_DIR}/lib")
	endif()
	add_custom_target(torchruntime ALL COMMENT "Copying LibTorch runtime")
	foreach(architecture IN ITEMS gadus melano)
		string(TOUPPER "${architecture}" architecture_upper)
		if(GADIDAE_BUILD_${architecture_upper})
			add_custom_command(TARGET torchruntime POST_BUILD
				COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/${architecture}"
				COMMAND ${CMAKE_COMMAND} -E copy_if_different
					${TORCH_DLLS} "${CMAKE_BINARY_DIR}/${architecture}"
			)
		endif()
	endforeach()
	if(GADIDAE_BUILD_ELEGINUS AND GADIDAE_BUILD_ELEGINUS_TRAINING)
		add_custom_command(TARGET torchruntime POST_BUILD
			COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/eleginus"
			COMMAND ${CMAKE_COMMAND} -E copy_if_different
				${TORCH_DLLS} "${CMAKE_BINARY_DIR}/eleginus"
		)
	endif()
endif()

if(GADIDAE_BUILD_GADUS)

add_library(gadus
	src/gadus/checkpoint.cpp
	src/gadus/args.cpp
	src/gadus/dataset.cpp
	src/gadus/evolution.cpp
	src/gadus/game.cpp
	src/gadus/match.cpp
	src/gadus/model.cpp
	src/gadus/searcher.cpp
)
set_target_properties(gadus PROPERTIES
	ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/gadus"
	LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/gadus"
	PDB_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/gadus"
)

target_include_directories(gadus
	PUBLIC "${CMAKE_SOURCE_DIR}/include"
)
target_include_directories(gadus
	SYSTEM PUBLIC
		"${GADIDAE_API_DIR}/chess"
		"${GADIDAE_API_DIR}/nlohmann/include"
		${HDF5_INCLUDE_DIRS}
)
target_precompile_headers(gadus PRIVATE <chess.hpp> <nlohmann/json.hpp>)
if(MSVC)
	target_link_libraries(
		gadus
		PUBLIC ${TORCH_LIBRARIES} ${HDF5_LIBRARIES} "${GADIDAE_ZLIB_STATIC}" Threads::Threads Shlwapi
	)
else()
	target_link_libraries(gadus PUBLIC ${TORCH_LIBRARIES} ${HDF5_LIBRARIES} ZLIB::ZLIB Threads::Threads)
endif()
target_compile_definitions(gadus PRIVATE H5_USE_110_API)

if(MSVC)
	target_compile_options(gadus PRIVATE /MP /W4 /permissive- /utf-8)
else()
	target_compile_options(gadus PRIVATE -Wall -Wextra -Wpedantic)
endif()

function(gadusexecutable name source)
	set(target "gadus${name}")
	add_executable(${target} ${source})
	set_target_properties(${target} PROPERTIES
		OUTPUT_NAME "${name}"
		RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/gadus"
		ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/gadus"
		LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/gadus"
		PDB_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/gadus"
	)
	target_link_libraries(${target} PRIVATE gadus)
	if(MSVC)
		add_dependencies(${target} torchruntime)
	endif()
endfunction()

gadusexecutable(preprocess src/gadus/preprocess.cpp)
gadusexecutable(train src/gadus/train.cpp)
gadusexecutable(search src/gadus/search.cpp)
gadusexecutable(arena src/gadus/arena.cpp)
gadusexecutable(fcpi src/gadus/fcpi.cpp)
gadusexecutable(uci src/gadus/uci.cpp)

add_executable(gadustests src/gadus/tests.cpp)
set_target_properties(gadustests PROPERTIES
	RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/gadus"
	ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/gadus"
	LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/gadus"
	PDB_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/gadus"
)
target_link_libraries(gadustests PRIVATE gadus)
if(MSVC)
	add_dependencies(gadustests torchruntime)
endif()
add_test(NAME gadustests COMMAND gadustests)

endif()

if(GADIDAE_BUILD_MELANO)

add_library(melano
	src/melano/checkpoint.cpp
	src/melano/args.cpp
	src/melano/dataset.cpp
	src/melano/game.cpp
	src/melano/match.cpp
	src/melano/model.cpp
	src/melano/searcher.cpp
)
set_target_properties(melano PROPERTIES
	ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/melano"
	LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/melano"
	PDB_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/melano"
)
target_include_directories(melano PUBLIC "${CMAKE_SOURCE_DIR}/include")
target_include_directories(melano SYSTEM PUBLIC
	"${GADIDAE_API_DIR}/chess"
	"${GADIDAE_API_DIR}/nlohmann/include"
	${HDF5_INCLUDE_DIRS}
)
target_precompile_headers(melano PRIVATE <chess.hpp> <nlohmann/json.hpp>)
if(MSVC)
	target_link_libraries(
		melano
		PUBLIC ${TORCH_LIBRARIES} ${HDF5_LIBRARIES} "${GADIDAE_ZLIB_STATIC}" Threads::Threads Shlwapi
	)
	target_compile_options(melano PRIVATE /MP /W4 /permissive- /utf-8)
else()
	target_link_libraries(melano PUBLIC ${TORCH_LIBRARIES} ${HDF5_LIBRARIES} ZLIB::ZLIB Threads::Threads)
	target_compile_options(melano PRIVATE -Wall -Wextra -Wpedantic)
endif()
target_compile_definitions(melano PRIVATE H5_USE_110_API)

function(melanoexecutable name source)
	set(target "melano${name}")
	add_executable(${target} ${source})
	set_target_properties(${target} PROPERTIES
		OUTPUT_NAME "${name}"
		RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/melano"
		ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/melano"
		LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/melano"
		PDB_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/melano"
	)
	target_link_libraries(${target} PRIVATE melano)
	if(MSVC)
		add_dependencies(${target} torchruntime)
	endif()
endfunction()

melanoexecutable(preprocess src/melano/preprocess.cpp)
melanoexecutable(train src/melano/train.cpp)
melanoexecutable(search src/melano/search.cpp)
melanoexecutable(arena src/melano/arena.cpp)
melanoexecutable(uci src/melano/uci.cpp)

add_executable(melanotests src/melano/tests.cpp)
set_target_properties(melanotests PROPERTIES
	RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/melano"
	ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/melano"
	LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/melano"
	PDB_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/melano"
)
target_link_libraries(melanotests PRIVATE melano)
if(MSVC)
	add_dependencies(melanotests torchruntime)
endif()
add_test(NAME melanotests COMMAND melanotests)

endif()

if(GADIDAE_BUILD_ELEGINUS)
	add_library(eleginus_runtime STATIC
		src/eleginus/game.cpp
		src/eleginus/nnue.cpp
		src/eleginus/runtime.cpp
		src/eleginus/search.cpp
	)
	set_target_properties(eleginus_runtime PROPERTIES
		ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/eleginus"
		LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/eleginus"
		PDB_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/eleginus"
	)
	target_include_directories(eleginus_runtime PUBLIC "${CMAKE_SOURCE_DIR}/include")
	target_include_directories(eleginus_runtime SYSTEM PUBLIC "${GADIDAE_API_DIR}/chess")
	target_precompile_headers(eleginus_runtime PRIVATE <chess.hpp>)
	if(MSVC)
		target_compile_options(eleginus_runtime PRIVATE /MP /W4 /permissive- /utf-8)
	else()
		target_compile_options(eleginus_runtime PRIVATE -Wall -Wextra -Wpedantic)
	endif()

	function(eleginusruntimeexecutable name source)
		set(target "eleginus${name}")
		add_executable(${target} ${source})
		set_target_properties(${target} PROPERTIES
			OUTPUT_NAME "${name}"
			RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/eleginus"
			ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/eleginus"
			LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/eleginus"
			PDB_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/eleginus"
		)
		target_link_libraries(${target} PRIVATE eleginus_runtime)
	endfunction()

	eleginusruntimeexecutable(search src/eleginus/searcher.cpp)
	eleginusruntimeexecutable(uci src/eleginus/uci.cpp)

	if(GADIDAE_BUILD_ELEGINUS_TRAINING)
		add_library(eleginus_training STATIC
			src/eleginus/args.cpp
			src/eleginus/checkpoint.cpp
			src/eleginus/dataset.cpp
			src/eleginus/model.cpp
		)
		set_target_properties(eleginus_training PROPERTIES
			ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/eleginus"
			LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/eleginus"
			PDB_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/eleginus"
		)
		target_include_directories(eleginus_training PUBLIC "${CMAKE_SOURCE_DIR}/include")
		target_include_directories(eleginus_training SYSTEM PUBLIC
			"${GADIDAE_API_DIR}/chess"
			${HDF5_INCLUDE_DIRS}
		)
		target_precompile_headers(eleginus_training PRIVATE <chess.hpp>)
		if(MSVC)
			target_link_libraries(eleginus_training PUBLIC eleginus_runtime ${TORCH_LIBRARIES}
				${HDF5_LIBRARIES} "${GADIDAE_ZLIB_STATIC}" Threads::Threads Shlwapi)
		else()
			target_link_libraries(eleginus_training PUBLIC eleginus_runtime ${TORCH_LIBRARIES}
				${HDF5_LIBRARIES} ZLIB::ZLIB Threads::Threads)
		endif()
		target_compile_definitions(eleginus_training PRIVATE H5_USE_110_API)
		if(MSVC)
			target_compile_options(eleginus_training PRIVATE /MP /W4 /permissive- /utf-8)
		else()
			target_compile_options(eleginus_training PRIVATE -Wall -Wextra -Wpedantic)
		endif()

		function(eleginustrainingexecutable name source)
			set(target "eleginus${name}")
			add_executable(${target} ${source})
			set_target_properties(${target} PROPERTIES
				OUTPUT_NAME "${name}"
				RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/eleginus"
				ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/eleginus"
				LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/eleginus"
				PDB_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/eleginus"
			)
			target_link_libraries(${target} PRIVATE eleginus_training)
			if(MSVC)
				add_dependencies(${target} torchruntime)
			endif()
		endfunction()

		eleginustrainingexecutable(preprocess src/eleginus/preprocess.cpp)
		eleginustrainingexecutable(train src/eleginus/train.cpp)
		eleginustrainingexecutable(embed src/eleginus/embed.cpp)
		eleginustrainingexecutable(tests src/eleginus/tests.cpp)
		add_test(NAME eleginustests COMMAND eleginustests)
	endif()
endif()

if(GADIDAE_BUILD_GRAPHICS)
# The native GUI owns no model implementation. It talks to Gadus, Melano, and
# third-party engines through UCI and therefore links only graphics dependencies.
add_executable(graphicstests
	src/graphics/tests.cpp
	src/graphics/game.cpp
)
target_include_directories(graphicstests PRIVATE "${CMAKE_SOURCE_DIR}/include")
target_include_directories(graphicstests SYSTEM PRIVATE "${GADIDAE_API_DIR}/chess")
if(MSVC)
	target_compile_options(graphicstests PRIVATE /W4 /permissive- /utf-8)
else()
	target_compile_options(graphicstests PRIVATE -Wall -Wextra -Wpedantic)
endif()
add_test(NAME graphicstests COMMAND graphicstests)

set(GADIDAE_PIECE_ARCHIVE "${CMAKE_SOURCE_DIR}/src/graphics/pieces.gpack")
if(NOT EXISTS "${GADIDAE_PIECE_ARCHIVE}")
	message(FATAL_ERROR "Missing compressed piece archive: ${GADIDAE_PIECE_ARCHIVE}")
endif()
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/generated")
if(WIN32)
	enable_language(RC)
	file(TO_NATIVE_PATH "${GADIDAE_PIECE_ARCHIVE}" GADIDAE_PIECE_ARCHIVE_NATIVE)
	string(REPLACE "\\" "\\\\" GADIDAE_PIECE_ARCHIVE_NATIVE "${GADIDAE_PIECE_ARCHIVE_NATIVE}")
	configure_file(
		"${CMAKE_SOURCE_DIR}/src/graphics/pieces.rc.in"
		"${CMAKE_BINARY_DIR}/generated/pieces.rc"
		@ONLY
	)
	set(GADIDAE_PIECE_ARCHIVE_SOURCE "${CMAKE_BINARY_DIR}/generated/pieces.rc")
else()
	enable_language(ASM)
	configure_file(
		"${CMAKE_SOURCE_DIR}/src/graphics/pieces.S.in"
		"${CMAKE_BINARY_DIR}/generated/pieces.S"
		@ONLY
	)
	set(GADIDAE_PIECE_ARCHIVE_SOURCE "${CMAKE_BINARY_DIR}/generated/pieces.S")
endif()
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
add_subdirectory("${GADIDAE_API_DIR}/glfw" "${CMAKE_BINARY_DIR}/dependencies/glfw" EXCLUDE_FROM_ALL)

set(FT_DISABLE_ZLIB ON CACHE BOOL "" FORCE)
set(FT_DISABLE_BZIP2 ON CACHE BOOL "" FORCE)
set(FT_DISABLE_PNG ON CACHE BOOL "" FORCE)
set(FT_DISABLE_HARFBUZZ ON CACHE BOOL "" FORCE)
set(FT_DISABLE_BROTLI ON CACHE BOOL "" FORCE)
# FreeType checks the ambient ZLIB_FOUND value even when system zlib is
# disabled. Hide the top-level package result so its built-in implementation
# is compiled into freetype instead of leaking zlib.dll into the GUI runtime.
set(GADIDAE_SAVED_ZLIB_FOUND "${ZLIB_FOUND}")
set(ZLIB_FOUND FALSE)
add_subdirectory("${GADIDAE_API_DIR}/freetype" "${CMAKE_BINARY_DIR}/dependencies/freetype" EXCLUDE_FROM_ALL)
set(ZLIB_FOUND "${GADIDAE_SAVED_ZLIB_FOUND}")
unset(GADIDAE_SAVED_ZLIB_FOUND)
if(MSVC)
	target_compile_options(freetype PRIVATE /utf-8)
endif()

add_library(gadidaeglad STATIC "${GADIDAE_API_DIR}/glad/src/gl.c")
target_include_directories(gadidaeglad SYSTEM PUBLIC "${GADIDAE_API_DIR}/glad/include")
if(UNIX AND NOT APPLE)
	target_link_libraries(gadidaeglad PUBLIC ${CMAKE_DL_LIBS})
endif()

add_library(gadidaeimgui STATIC
	"${GADIDAE_API_DIR}/imgui/imgui.cpp"
	"${GADIDAE_API_DIR}/imgui/imgui_demo.cpp"
	"${GADIDAE_API_DIR}/imgui/imgui_draw.cpp"
	"${GADIDAE_API_DIR}/imgui/imgui_tables.cpp"
	"${GADIDAE_API_DIR}/imgui/imgui_widgets.cpp"
	"${GADIDAE_API_DIR}/imgui/backends/imgui_impl_glfw.cpp"
	"${GADIDAE_API_DIR}/imgui/backends/imgui_impl_opengl3.cpp"
	"${GADIDAE_API_DIR}/imgui/misc/cpp/imgui_stdlib.cpp"
	"${GADIDAE_API_DIR}/imgui/misc/freetype/imgui_freetype.cpp"
)
target_include_directories(gadidaeimgui SYSTEM PUBLIC
	"${GADIDAE_API_DIR}/imgui"
	"${GADIDAE_API_DIR}/imgui/backends"
	"${GADIDAE_API_DIR}/imgui/misc/cpp"
	"${GADIDAE_API_DIR}/imgui/misc/freetype"
)
target_compile_definitions(gadidaeimgui PUBLIC IMGUI_ENABLE_FREETYPE)
target_link_libraries(gadidaeimgui PUBLIC glfw freetype)

add_executable(gadidaegraphics
	src/graphics/main.cpp
	src/graphics/application.cpp
	src/graphics/game.cpp
	src/graphics/archive.cpp
	src/graphics/pieces.cpp
	src/graphics/registry.cpp
	src/graphics/simulator.cpp
	src/graphics/stadium.cpp
	src/graphics/uci.cpp
	"${GADIDAE_PIECE_ARCHIVE_SOURCE}"
)
set_target_properties(gadidaegraphics PROPERTIES
	OUTPUT_NAME "Gadidae"
	RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/graphics"
	ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/graphics"
	LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/graphics"
	PDB_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/graphics"
)
if(WIN32)
	set_target_properties(gadidaegraphics PROPERTIES WIN32_EXECUTABLE TRUE)
endif()
target_include_directories(gadidaegraphics
	PRIVATE "${CMAKE_SOURCE_DIR}/include"
)
target_include_directories(gadidaegraphics SYSTEM PRIVATE
		"${GADIDAE_API_DIR}/chess"
		"${GADIDAE_API_DIR}/nlohmann/include"
		"${GADIDAE_API_DIR}/glm"
		"${GADIDAE_ZLIB_DIR}/include"
)
target_link_libraries(gadidaegraphics PRIVATE
	gadidaeimgui
	gadidaeglad
	Threads::Threads
)
if(MSVC)
	target_compile_options(gadidaegraphics PRIVATE /MP /W4 /permissive- /utf-8)
	target_link_libraries(gadidaegraphics PRIVATE
		"${GADIDAE_ZLIB_STATIC}" Comdlg32 Shell32 User32)
else()
	target_compile_options(gadidaegraphics PRIVATE -Wall -Wextra -Wpedantic)
	target_link_libraries(gadidaegraphics PRIVATE ZLIB::ZLIB)
endif()
endif()
