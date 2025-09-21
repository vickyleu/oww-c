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
set(_ort_prefix_candidates)
if(DEFINED ONNXR AND NOT "${ONNXR}" STREQUAL "")
  list(APPEND _ort_prefix_candidates "${ONNXR}")
elseif(DEFINED ENV{ORT_PREFIX} AND NOT "$ENV{ORT_PREFIX}" STREQUAL "")
  list(APPEND _ort_prefix_candidates "$ENV{ORT_PREFIX}")
endif()

set(_ort_include_candidates)
if(DEFINED ONNXR_INCLUDE_DIR AND NOT "${ONNXR_INCLUDE_DIR}" STREQUAL "")
  list(APPEND _ort_include_candidates "${ONNXR_INCLUDE_DIR}")
endif()
if(DEFINED ENV{ORT_INCLUDE_DIR} AND NOT "$ENV{ORT_INCLUDE_DIR}" STREQUAL "")
  list(APPEND _ort_include_candidates "$ENV{ORT_INCLUDE_DIR}")
endif()

set(_ort_lib_candidates)
if(DEFINED ONNXR_LIB_DIR AND NOT "${ONNXR_LIB_DIR}" STREQUAL "")
  list(APPEND _ort_lib_candidates "${ONNXR_LIB_DIR}")
endif()
if(DEFINED ENV{ORT_LIB_DIR} AND NOT "$ENV{ORT_LIB_DIR}" STREQUAL "")
  list(APPEND _ort_lib_candidates "$ENV{ORT_LIB_DIR}")
endif()

foreach(_prefix ${_ort_prefix_candidates})
  list(APPEND _ort_include_candidates "${_prefix}/include")
  list(APPEND _ort_lib_candidates "${_prefix}/lib")
endforeach()

list(REMOVE_DUPLICATES _ort_include_candidates)
list(REMOVE_DUPLICATES _ort_lib_candidates)

set(OWW_ONNXR_INCLUDE_DIRS "")
foreach(_inc ${_ort_include_candidates})
  if(EXISTS "${_inc}/onnxruntime/core/session/onnxruntime_c_api.h")
    set(OWW_ONNXR_INCLUDE_DIRS "${_inc}")
    break()
  endif()
endforeach()

set(OWW_ONNXR_LIBS "")
set(_onnxruntime_search_paths)
foreach(_lib ${_ort_lib_candidates})
  if(EXISTS "${_lib}")
    list(APPEND _onnxruntime_search_paths "${_lib}")
  endif()
endforeach()

if(_onnxruntime_search_paths)
  set(_onnxr_primary_lib "")
  foreach(_dir ${_onnxruntime_search_paths})
    foreach(_name IN ITEMS libonnxruntime.so libonnxruntime.dylib libonnxruntime.dll.a libonnxruntime.a)
      if(EXISTS "${_dir}/${_name}")
        set(_onnxr_primary_lib "${_dir}/${_name}")
        break()
      endif()
    endforeach()
    if(_onnxr_primary_lib)
      set(OWW_ONNXR_LIBS ${_onnxr_primary_lib})
      foreach(_suffix IN ITEMS lora providers optimizer session util)
        foreach(_extra_name IN ITEMS
            libonnxruntime_${_suffix}.so
            libonnxruntime_${_suffix}.a)
          if(EXISTS "${_dir}/${_extra_name}")
            list(APPEND OWW_ONNXR_LIBS "${_dir}/${_extra_name}")
            break()
          endif()
        endforeach()
      endforeach()
      break()
    endif()
  endforeach()
endif()

option(OWW_USE_PKGCONFIG "尝试使用 pkg-config 自动查找 onnxruntime" ON)

if((NOT OWW_ONNXR_INCLUDE_DIRS OR NOT OWW_ONNXR_LIBS) AND OWW_USE_PKGCONFIG)
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

if(NOT OWW_ONNXR_INCLUDE_DIRS)
  message(FATAL_ERROR "未找到 onnxruntime 头文件。请设置 ONNXR_INCLUDE_DIR 或 ORT_INCLUDE_DIR 指向包含 onnxruntime/core/session/onnxruntime_c_api.h 的目录，或提供 ONNXR 前缀")
endif()

if(NOT OWW_ONNXR_LIBS)
  message(FATAL_ERROR "未找到 onnxruntime 库。请设置 ONNXR_LIB_DIR 或 ORT_LIB_DIR 指向包含 libonnxruntime.* 的目录，或提供 ONNXR 前缀")
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
