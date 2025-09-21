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

# ONNXR 指向你解压/安装好的 onnxruntime 根目录 (含 include/ lib/)
if(NOT DEFINED ONNXR)
  message(FATAL_ERROR "pass -DONNXR=/path/to/onnxruntime/prefix")
endif()

set(ONNXR_INCLUDE "${ONNXR}/include")
set(ONNXR_LIB_DIR "${ONNXR}/lib")

if(NOT EXISTS "${ONNXR_INCLUDE}/onnxruntime/core/session/onnxruntime_c_api.h")
  message(FATAL_ERROR "${ONNXR_INCLUDE} 缺少 onnxruntime 头文件")
endif()

find_library(ONNXR_LIBRARY
  NAMES onnxruntime libonnxruntime
  HINTS "${ONNXR_LIB_DIR}"
  PATH_SUFFIXES ""
)
if(NOT ONNXR_LIBRARY)
  message(FATAL_ERROR "未在 ${ONNXR_LIB_DIR} 找到 libonnxruntime")
endif()

find_library(ONNXR_LORA NAMES onnxruntime_lora PATHS "${ONNXR_LIB_DIR}" NO_DEFAULT_PATH)
find_library(ONNXR_PROVIDERS NAMES onnxruntime_providers PATHS "${ONNXR_LIB_DIR}" NO_DEFAULT_PATH)
find_library(ONNXR_OPTIMIZER NAMES onnxruntime_optimizer PATHS "${ONNXR_LIB_DIR}" NO_DEFAULT_PATH)
find_library(ONNXR_SESSION NAMES onnxruntime_session PATHS "${ONNXR_LIB_DIR}" NO_DEFAULT_PATH)
find_library(ONNXR_UTIL NAMES onnxruntime_util PATHS "${ONNXR_LIB_DIR}" NO_DEFAULT_PATH)

set(_onnxr_extra_libs)
foreach(_lib IN ITEMS ONNXR_LORA ONNXR_PROVIDERS ONNXR_OPTIMIZER ONNXR_SESSION ONNXR_UTIL)
  if(${_lib})
    list(APPEND _onnxr_extra_libs ${${_lib}})
  endif()
endforeach()

find_package(Threads REQUIRED)

add_library(oww STATIC
  src/oww.cpp
)
target_include_directories(oww
  PUBLIC
    $<BUILD_INTERFACE:${OWW_ROOT}/include>
    $<INSTALL_INTERFACE:include>
    ${ONNXR_INCLUDE}
)
target_link_libraries(oww
  PUBLIC
    ${ONNXR_LIBRARY}
    ${_onnxr_extra_libs}
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
