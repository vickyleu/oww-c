cmake_minimum_required(VERSION 3.15)
project(oww_c C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(OWW_ENABLE_SANITIZERS "启用 address/undefined sanitizer (仅本机调试)" OFF)
if(OWW_ENABLE_SANITIZERS AND CMAKE_CROSSCOMPILING)
  message(WARNING "交叉编译时忽略 sanitizer")
  set(OWW_ENABLE_SANITIZERS OFF)
endif()

if(OWW_ENABLE_SANITIZERS)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address,undefined)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-ffunction-sections -fdata-sections)
  add_link_options(-Wl,--gc-sections)
endif()

set(OWW_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

# 允许首选环境变量 ORT_PREFIX，或者直接传 -DONNXR=...
if(NOT DEFINED ONNXR)
  if(DEFINED ENV{ORT_PREFIX} AND NOT "$ENV{ORT_PREFIX}" STREQUAL "")
    set(ONNXR $ENV{ORT_PREFIX})
  endif()
endif()

set(OWW_ONNXR_INCLUDE_DIRS "")
set(OWW_ONNXR_LIBS "")

set(_oww_prefix_candidates)
if(DEFINED ONNXR AND NOT "${ONNXR}" STREQUAL "")
  list(APPEND _oww_prefix_candidates "${ONNXR}")
endif()

foreach(_prefix ${_oww_prefix_candidates})
  set(_include_dir "${_prefix}/include")
  set(_lib_dir "${_prefix}/lib")
  if(EXISTS "${_include_dir}/onnxruntime/core/session/onnxruntime_c_api.h" AND NOT OWW_ONNXR_INCLUDE_DIRS)
    set(OWW_ONNXR_INCLUDE_DIRS "${_include_dir}")
  endif()
  if(EXISTS "${_lib_dir}" AND NOT OWW_ONNXR_LIBS)
    set(_onnxr_primary_candidates
      "${_lib_dir}/libonnxruntime.so"
      "${_lib_dir}/libonnxruntime.so.1"
      "${_lib_dir}/libonnxruntime.so.1.22.1"
      "${_lib_dir}/libonnxruntime.a"
    )
    set(_onnxr_primary_lib "")
    foreach(_cand ${_onnxr_primary_candidates})
      if(EXISTS "${_cand}")
        set(_onnxr_primary_lib "${_cand}")
        break()
      endif()
    endforeach()
    if(NOT _onnxr_primary_lib STREQUAL "")
      set(_onnxr_extra_libs)
      foreach(_suffix IN ITEMS lora providers optimizer session util)
        set(_lib "${_lib_dir}/libonnxruntime_${_suffix}.a")
        if(EXISTS "${_lib}")
          list(APPEND _onnxr_extra_libs "${_lib}")
        endif()
      endforeach()
      set(OWW_ONNXR_LIBS ${_onnxr_primary_lib} ${_onnxr_extra_libs})
    endif()
  endif()
  if(OWW_ONNXR_INCLUDE_DIRS AND OWW_ONNXR_LIBS)
    break()
  endif()
endforeach()

# 若需要，支持分别指定 include / lib 目录
if(NOT OWW_ONNXR_INCLUDE_DIRS)
  set(_include_candidates)
  if(DEFINED ONNXR_INCLUDE_DIR AND NOT "${ONNXR_INCLUDE_DIR}" STREQUAL "")
    list(APPEND _include_candidates "${ONNXR_INCLUDE_DIR}")
  elseif(DEFINED ENV{ORT_INCLUDE_DIR} AND NOT "$ENV{ORT_INCLUDE_DIR}" STREQUAL "")
    list(APPEND _include_candidates "$ENV{ORT_INCLUDE_DIR}")
  endif()
  foreach(_inc ${_include_candidates})
    if(EXISTS "${_inc}/onnxruntime/core/session/onnxruntime_c_api.h")
      set(OWW_ONNXR_INCLUDE_DIRS "${_inc}")
      break()
    endif()
  endforeach()
endif()

if(NOT OWW_ONNXR_LIBS)
  set(_lib_candidates)
  if(DEFINED ONNXR_LIB_DIR AND NOT "${ONNXR_LIB_DIR}" STREQUAL "")
    list(APPEND _lib_candidates "${ONNXR_LIB_DIR}")
  elseif(DEFINED ENV{ORT_LIB_DIR} AND NOT "$ENV{ORT_LIB_DIR}" STREQUAL "")
    list(APPEND _lib_candidates "$ENV{ORT_LIB_DIR}")
  endif()
  foreach(_lib_dir ${_lib_candidates})
    set(_onnxr_primary_candidates
      "${_lib_dir}/libonnxruntime.so"
      "${_lib_dir}/libonnxruntime.so.1"
      "${_lib_dir}/libonnxruntime.so.1.22.1"
      "${_lib_dir}/libonnxruntime.a"
    )
    set(_onnxr_primary_lib "")
    foreach(_cand ${_onnxr_primary_candidates})
      if(EXISTS "${_cand}")
        set(_onnxr_primary_lib "${_cand}")
        break()
      endif()
    endforeach()
    if(NOT _onnxr_primary_lib STREQUAL "")
      set(_onnxr_extra_libs)
      foreach(_suffix IN ITEMS lora providers optimizer session util)
        set(_lib "${_lib_dir}/libonnxruntime_${_suffix}.a")
        if(EXISTS "${_lib}")
          list(APPEND _onnxr_extra_libs "${_lib}")
        endif()
      endforeach()
      set(OWW_ONNXR_LIBS ${_onnxr_primary_lib} ${_onnxr_extra_libs})
      break()
    endif()
  endforeach()
endif()

option(OWW_USE_PKGCONFIG "尝试使用 pkg-config 自动查找 onnxruntime" ON)

# 如果没有显式路径，尝试 pkg-config
if(OWW_USE_PKGCONFIG AND (NOT OWW_ONNXR_LIBS OR NOT OWW_ONNXR_INCLUDE_DIRS))
  find_package(PkgConfig QUIET)
  if(PkgConfig_FOUND)
    pkg_check_modules(ONNXR_PKG QUIET IMPORTED_TARGET onnxruntime)
    if(ONNXR_PKG_FOUND)
      if(NOT OWW_ONNXR_INCLUDE_DIRS)
        set(OWW_ONNXR_INCLUDE_DIRS ${ONNXR_PKG_INCLUDE_DIRS})
      endif()
      if(NOT OWW_ONNXR_LIBS)
        set(OWW_ONNXR_LIBS PkgConfig::ONNXR_PKG)
      endif()
    endif()
  endif()
endif()

if(NOT OWW_ONNXR_INCLUDE_DIRS OR NOT OWW_ONNXR_LIBS)
  message(FATAL_ERROR "未能找到完整的 onnxruntime 开发包。请设置 ONNXR=/prefix 或分别提供 ONNXR_INCLUDE_DIR / ONNXR_LIB_DIR（或导出 ORT_INCLUDE_DIR / ORT_LIB_DIR），也可安装带 pkg-config 的 onnxruntime 开发包。")
endif()

find_package(Threads REQUIRED)

add_library(oww STATIC
  src/oww.cpp
)
target_include_directories(oww
  PUBLIC
    $<BUILD_INTERFACE:${OWW_ROOT}/include>
    $<INSTALL_INTERFACE:include>
    ${OWW_ONNXR_INCLUDE_DIRS}
)
target_link_libraries(oww
  PUBLIC
    ${OWW_ONNXR_LIBS}
    Threads::Threads
)
if(UNIX AND NOT APPLE)
  target_link_libraries(oww PUBLIC m dl)
endif()
set_target_properties(oww PROPERTIES
  C_VISIBILITY_PRESET hidden
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
)
target_compile_options(oww PRIVATE -Wall -Wextra -Wpedantic)

# demo（可关）
option(OWW_BUILD_DEMO "build demo" ON)
if(OWW_BUILD_DEMO)
  add_executable(oww_alsa demo/main_alsa.cpp)
  target_include_directories(oww_alsa PRIVATE ${OWW_ROOT}/include)
  target_link_libraries(oww_alsa PRIVATE oww asound)
endif()

install(TARGETS oww ARCHIVE DESTINATION lib)
install(FILES include/oww.h DESTINATION include)
