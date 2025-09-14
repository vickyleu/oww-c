cmake_minimum_required(VERSION 3.15)
project(oww_c C CXX)
set(CMAKE_CXX_STANDARD 14)
set(ONNXR "/home/vickyleu/build/third_party/onnxruntime/install-a64")  # 改成你的路径

add_library(oww STATIC oww.cpp)
target_include_directories(oww PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${ONNXR}/include)
target_link_libraries(oww PUBLIC ${ONNXR}/lib/libonnxruntime.so pthread m)

# 可选demo
add_executable(demo demo.cpp)
target_link_libraries(demo PRIVATE oww)
