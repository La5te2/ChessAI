cmake_minimum_required(VERSION 3.24)

# Applies source-level compatibility fixes to the pinned LibTorch package.

if(NOT DEFINED API_DIR)
	get_filename_component(API_DIR "${CMAKE_CURRENT_LIST_DIR}" ABSOLUTE)
endif()
if(NOT DEFINED TORCH_DIR)
	set(TORCH_DIR "${API_DIR}/libtorch")
endif()

file(STRINGS "${API_DIR}/versions.env" torch_version_line REGEX "^TORCH_VERSION=")
list(LENGTH torch_version_line torch_version_count)
if(NOT torch_version_count EQUAL 1)
	message(FATAL_ERROR "Dependency lock must define TORCH_VERSION exactly once")
endif()
list(GET torch_version_line 0 torch_version_line)
string(REGEX REPLACE "^[^=]+=" "" torch_version "${torch_version_line}")
if(NOT torch_version STREQUAL "2.13.0")
	message(FATAL_ERROR "LibTorch source patches are defined only for 2.13.0")
endif()

function(replace_exact path original replacement description)
	if(NOT EXISTS "${path}")
		message(FATAL_ERROR "Cannot patch missing file: ${path}")
	endif()
	file(READ "${path}" content)
	string(FIND "${content}" "${replacement}" replacement_position)
	if(NOT replacement_position EQUAL -1)
		return()
	endif()
	string(FIND "${content}" "${original}" original_position)
	if(original_position EQUAL -1)
		message(FATAL_ERROR "LibTorch 2.13.0 patch context changed: ${description}")
	endif()
	string(REPLACE "${original}" "${replacement}" content "${content}")
	file(WRITE "${path}" "${content}")
	message(STATUS "Patched LibTorch: ${description}")
endfunction()

replace_exact(
	"${TORCH_DIR}/include/ATen/core/function_schema.h"
	"      if(name == arguments()[i].name())\n        return i;"
	"      if(name == arguments()[i].name())\n        return static_cast<int>(i);"
	"FunctionSchema argument index result type"
)

replace_exact(
	"${TORCH_DIR}/include/torch/csrc/dynamo/compiled_autograd.h"
	"      size_input_origins.emplace_back(active_node_call_idx.value());"
	"      size_input_origins.emplace_back(\n          static_cast<std::uint32_t>(active_node_call_idx.value()));"
	"compiled autograd size-input origin type"
)

replace_exact(
	"${TORCH_DIR}/include/torch/csrc/dynamo/compiled_autograd.h"
	"        input_origins.emplace_back(active_node_call_idx.value());"
	"        input_origins.emplace_back(\n            static_cast<std::uint32_t>(active_node_call_idx.value()));"
	"compiled autograd tensor-input origin type"
)

replace_exact(
	"${TORCH_DIR}/include/torch/csrc/dynamo/compiled_autograd.h"
	"      args_origins.emplace_back(active_node_call_idx.value());"
	"      args_origins.emplace_back(\n          static_cast<std::uint32_t>(active_node_call_idx.value()));"
	"compiled autograd lifted-value origin type"
)

replace_exact(
	"${TORCH_DIR}/include/torch/csrc/dynamo/compiled_autograd.h"
	"    _node_call.tensor_pre_hooks.emplace_back(fn_id, index);"
	"    _node_call.tensor_pre_hooks.emplace_back(static_cast<int>(fn_id), index);"
	"compiled autograd tensor pre-hook index type"
)

replace_exact(
	"${TORCH_DIR}/include/torch/csrc/dynamo/compiled_autograd.h"
	"    _node_call.cpp_tensor_pre_hooks.emplace_back(hook_id, idx);"
	"    _node_call.cpp_tensor_pre_hooks.emplace_back(\n        static_cast<int>(hook_id), static_cast<int>(idx));"
	"compiled autograd C++ tensor pre-hook index types"
)

replace_exact(
	"${TORCH_DIR}/include/torch/csrc/dynamo/compiled_autograd.h"
	"    _node_call.pre_hooks.emplace_back(fn_id);"
	"    _node_call.pre_hooks.emplace_back(static_cast<int>(fn_id));"
	"compiled autograd pre-hook index type"
)

replace_exact(
	"${TORCH_DIR}/include/torch/csrc/dynamo/compiled_autograd.h"
	"    _node_call.post_hooks.emplace_back(fn_id);"
	"    _node_call.post_hooks.emplace_back(static_cast<int>(fn_id));"
	"compiled autograd post-hook index type"
)

replace_exact(
	"${TORCH_DIR}/include/torch/csrc/dynamo/compiled_autograd.h"
	"    _node_call.post_acc_grad_hooks.emplace_back(fn_id);"
	"    _node_call.post_acc_grad_hooks.emplace_back(static_cast<int>(fn_id));"
	"compiled autograd post-accumulate hook index type"
)

replace_exact(
	"${TORCH_DIR}/include/c10/util/irange.h"
	"    return false; // Horrible hack"
	"#if defined(_MSC_VER) && !defined(__CUDACC__)\n    __assume(0);\n#else\n    return false; // Required by affected NVCC versions.\n#endif"
	"MSVC unreachable integer-iterator tail"
)

replace_exact(
	"${TORCH_DIR}/share/cmake/Torch/TorchConfig.cmake"
	"if(ON)\n  append_torchlib_if_found(kineto)\nendif()"
	"if(EXISTS \"\${TORCH_INSTALL_PREFIX}/lib/kineto.lib\" OR\n   EXISTS \"\${TORCH_INSTALL_PREFIX}/lib/libkineto.a\")\n  append_torchlib_if_found(kineto)\nendif()"
	"optional kineto package lookup"
)
