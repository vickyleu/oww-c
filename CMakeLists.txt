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

if(NOT DEFINED ONNXR)
  message(FATAL_ERROR "pass -DONNXR=/absolute/path/to/onnxruntime/install-a64")
endif()

find_path(ONNXR_INCLUDE_ROOT
  NAMES onnxruntime/core/session/onnxruntime_c_api.h
  HINTS "${ONNXR}/include"
)
if(NOT ONNXR_INCLUDE_ROOT)
  message(FATAL_ERROR "无法在 ${ONNXR} 下面找到 onnxruntime 头文件")
endif()

find_library(ONNXR_LIBRARY
  NAMES onnxruntime
  HINTS "${ONNXR}/lib"
)
if(NOT ONNXR_LIBRARY)
  message(FATAL_ERROR "无法在 ${ONNXR} 下面找到 onnxruntime 库")
endif()

set(OWW_LINK_LIBS ${ONNXR_LIBRARY} pthread m)
if(UNIX AND NOT APPLE)
  list(APPEND OWW_LINK_LIBS dl)
endif()

add_library(oww STATIC
  src/oww.cpp
)
target_include_directories(oww
  PUBLIC
    ${OWW_ROOT}/include
    ${ONNXR_INCLUDE_ROOT}
)
target_link_libraries(oww
  PUBLIC
    ${OWW_LINK_LIBS}
)
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
