cmake_minimum_required(VERSION 3.24)

project(GadidaeCpp LANGUAGES C CXX)

if(MSVC AND CMAKE_GENERATOR MATCHES "Ninja")
	set(CMAKE_CL_SHOWINCLUDES_PREFIX "注意: 包含文件:  ")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(GADIDAE_API_DIR "${CMAKE_SOURCE_DIR}/api" CACHE PATH "Local dependency directory")
set(GADIDAE_TORCH_DIR "${GADIDAE_API_DIR}/libtorch" CACHE PATH "LibTorch root")
set(GADIDAE_HDF5_DIR "${GADIDAE_API_DIR}/hdf5" CACHE PATH "HDF5 install root")
set(GADIDAE_ZLIB_DIR "${GADIDAE_API_DIR}/zlib" CACHE PATH "zlib install root")

if(EXISTS "${GADIDAE_TORCH_DIR}/share/cmake/Torch")
  list(PREPEND CMAKE_PREFIX_PATH "${GADIDAE_TORCH_DIR}")
endif()
if(EXISTS "${GADIDAE_HDF5_DIR}")
	list(PREPEND CMAKE_PREFIX_PATH "${GADIDAE_HDF5_DIR}")
	set(HDF5_ROOT "${GADIDAE_HDF5_DIR}")
endif()
if(EXISTS "${GADIDAE_ZLIB_DIR}")
	list(PREPEND CMAKE_PREFIX_PATH "${GADIDAE_ZLIB_DIR}")
	set(ZLIB_ROOT "${GADIDAE_ZLIB_DIR}")
endif()

find_package(Torch REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C)
find_package(ZLIB REQUIRED)
find_package(Threads REQUIRED)

if(MSVC)
	find_library(
		GADIDAE_ZLIB_STATIC
		NAMES zlibstatic
		PATHS "${GADIDAE_ZLIB_DIR}/lib"
		NO_DEFAULT_PATH
		REQUIRED
	)
endif()

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

target_include_directories(gadus
	PUBLIC "${CMAKE_SOURCE_DIR}/include"
)
target_include_directories(gadus
	SYSTEM PUBLIC
		"${GADIDAE_API_DIR}/chess"
		"${GADIDAE_API_DIR}/nlohmann/include"
		${HDF5_INCLUDE_DIRS}
)
target_precompile_headers(gadus PRIVATE <chess.hpp> <nlohmann/json.hpp> <torch/torch.h>)
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
  )
  target_link_libraries(${target} PRIVATE gadus)
  if(MSVC)
    file(GLOB TORCH_DLLS "${GADIDAE_TORCH_DIR}/lib/*.dll")
    if(TORCH_DLLS)
      add_custom_command(TARGET ${target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${TORCH_DLLS}
          $<TARGET_FILE_DIR:${target}>)
    endif()
  endif()
endfunction()

gadusexecutable(preprocess src/gadus/preprocess.cpp)
gadusexecutable(train src/gadus/train.cpp)
gadusexecutable(search src/gadus/search.cpp)
gadusexecutable(arena src/gadus/arena.cpp)
gadusexecutable(fcpi src/gadus/fcpi.cpp)

enable_testing()
add_executable(gadustests tests/gadustests.cpp)
set_target_properties(gadustests PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/gadus")
target_link_libraries(gadustests PRIVATE gadus)
add_test(NAME gadustests COMMAND gadustests)
