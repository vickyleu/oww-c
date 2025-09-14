cmake_minimum_required(VERSION 3.15)
project(oww_c C CXX)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# 必传：ONNXR=/abs/path/to/onnxruntime/install-a64
if(NOT DEFINED ONNXR)
  message(FATAL_ERROR "pass -DONNXR=/absolute/path/to/onnxruntime/install-a64")
endif()

add_library(oww STATIC
  src/oww.cpp
)
target_include_directories(oww PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${ONNXR}/include
)

# demo（可关）
option(OWW_BUILD_DEMO "build demo" ON)
if(OWW_BUILD_DEMO)
  add_executable(oww_alsa demo/main_alsa.cpp)
  target_include_directories(oww_alsa PRIVATE ${ONNXR}/include ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_directories(oww_alsa PRIVATE ${ONNXR}/lib)
  target_link_libraries(oww_alsa oww onnxruntime asound pthread m)
endif()

install(TARGETS oww ARCHIVE DESTINATION lib)
install(FILES include/oww.h DESTINATION include)
