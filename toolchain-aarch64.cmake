if(NOT DEFINED ENV{XTOOLS})
  message(FATAL_ERROR "请先 export XTOOLS=/path/to/x-tools/aarch64-unknown-linux-gnu")
endif()
set(_oww_xtools "$ENV{XTOOLS}")
if(NOT EXISTS "${_oww_xtools}")
  message(FATAL_ERROR "XTOOLS='${_oww_xtools}' 无效，目录不存在")
endif()
file(TO_CMAKE_PATH "${_oww_xtools}" _oww_xtools)

set(_oww_tool_bin "${_oww_xtools}/bin")
if(NOT EXISTS "${_oww_tool_bin}")
  message(FATAL_ERROR "未找到 ${_oww_tool_bin}，确认交叉工具链目录结构")
endif()

find_program(CMAKE_C_COMPILER
  NAMES aarch64-unknown-linux-gnu-gcc aarch64-linux-gnu-gcc
  PATHS "${_oww_tool_bin}"
  NO_DEFAULT_PATH
  REQUIRED
)
find_program(CMAKE_CXX_COMPILER
  NAMES aarch64-unknown-linux-gnu-g++ aarch64-linux-gnu-g++
  PATHS "${_oww_tool_bin}"
  NO_DEFAULT_PATH
  REQUIRED
)

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)
set(CMAKE_C_COMPILER_TARGET aarch64-unknown-linux-gnu)
set(CMAKE_CXX_COMPILER_TARGET aarch64-unknown-linux-gnu)

set(_oww_default_sysroot "${_oww_xtools}/aarch64-unknown-linux-gnu/sysroot")
if(DEFINED ENV{XTOOLS_SYSROOT})
  set(CMAKE_SYSROOT "$ENV{XTOOLS_SYSROOT}")
elseif(EXISTS "${_oww_default_sysroot}")
  set(CMAKE_SYSROOT "${_oww_default_sysroot}")
else()
  message(WARNING "未检测到 sysroot，若失败请显式传递 -DCMAKE_SYSROOT 或 export XTOOLS_SYSROOT")
endif()

set(CMAKE_FIND_ROOT_PATH ${CMAKE_SYSROOT})
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
