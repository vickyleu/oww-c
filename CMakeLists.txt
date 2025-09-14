cmake_minimum_required(VERSION 3.15)
project(oww_c C CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ONNX Runtime 安装路径
set(ONNXR $ENV{HOME}/build/third_party/onnxruntime/install-a64)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${ONNXR}/include
)

link_directories(${ONNXR}/lib)

add_library(oww STATIC oww.cpp)

target_link_libraries(oww
    ${ONNXR}/lib/libonnxruntime.so
    pthread m
)

install(TARGETS oww ARCHIVE DESTINATION lib)
install(FILES oww.h DESTINATION include)
