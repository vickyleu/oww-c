cmake_minimum_required(VERSION 3.20)
project(oww VERSION 1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 99)

find_package(Threads REQUIRED)

option(OWW_USE_PKGCONFIG "Use pkg-config to find onnxruntime" ON)

if(OWW_USE_PKGCONFIG)
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(ONNXR REQUIRED libonnxruntime)
  set(OWW_ONNXR_LIBS ${ONNXR_LINK_LIBRARIES})
  include_directories(${ONNXR_INCLUDE_DIRS})
  link_directories(${ONNXR_LIBRARY_DIRS})
else()
  if(NOT DEFINED ONNXR_INCLUDE_DIR)
    message(FATAL_ERROR "ONNXR_INCLUDE_DIR must be set")
  endif()
  if(NOT DEFINED ONNXR_LIB_DIR)
    message(FATAL_ERROR "ONNXR_LIB_DIR must be set")
  endif()
  set(OWW_ONNXR_LIBS onnxruntime)
  include_directories(${ONNXR_INCLUDE_DIR})
  link_directories(${ONNXR_LIB_DIR})
endif()

# 提取aubio的所有对象文件
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/aubio_objs)
execute_process(
    COMMAND ${CMAKE_AR} x ${CMAKE_CURRENT_SOURCE_DIR}/external/aubio/lib/libaubio.a
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/aubio_objs
)

# 获取所有提取的.o文件
file(GLOB AUBIO_OBJECTS ${CMAKE_BINARY_DIR}/aubio_objs/*.o)

# 创建oww静态库，包含aubio的对象文件
add_library(oww STATIC src/oww.cpp ${AUBIO_OBJECTS})

target_include_directories(oww 
  PUBLIC include
  PRIVATE external/aubio/include
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

install(TARGETS oww ARCHIVE DESTINATION lib)
install(FILES include/oww.h DESTINATION include)
