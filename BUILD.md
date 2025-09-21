# 构建指南

本文档说明如何在本机或通过 aarch64 交叉工具链编译 `oww-c`，重点放在 ONNX Runtime 的获取与配置。除非特别说明，以下命令全部在仓库根目录执行（`CMakeLists.txt` 所在目录）。

## 1. 准备工作

- CMake ≥ 3.15，推荐与 Ninja 搭配使用：`sudo apt install cmake ninja-build`
- C/C++ 编译器（主机构建可用 clang/gcc；交叉构建需准备 aarch64 工具链）
- ONNX Runtime 1.22.1（或兼容版本），需包含开发头文件与库
- 可选：`pkg-config`。若已安装带 `.pc` 文件的 onnxruntime 包，可省去手工设置路径

当前 CMake 脚本会按下列优先级寻找 ONNX Runtime：

1. 命令行传入 `-DONNXR=/path/to/onnxruntime`
2. 环境变量 `ORT_PREFIX`
3. `pkg-config --libs --cflags onnxruntime`

找到后会自动填充 include 与 library 路径，省去额外参数。

## 2. 快速开始（本机构建）

以下示例假设你已经安装了能提供 `pkg-config` 信息的 ONNX Runtime。如果没有，请先将其安装到固定路径并在第 2.2 节配置。

```bash
mkdir -p build
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

生成文件：

- 静态库：`build/liboww.a`
- 示例程序（默认开启）：`build/oww_alsa`

若不需要演示程序，可在配置时添加 `-DOWW_BUILD_DEMO=OFF`。

### 2.1 使用 pkg-config 的 onnxruntime

确保 `pkg-config --libs onnxruntime` 能输出正确信息后，即可直接执行上方命令。CMake 会发现 `PkgConfig::ONNXR_PKG` 并完成链接。

### 2.2 手工指定 ONNX Runtime 路径

如果是自编译或压缩包解出的 ONNX Runtime，请设置以下任一方式：

```bash
# 方式一：命令行参数
cmake -S . -B build -G Ninja \
  -DONNXR=$HOME/opt/onnxruntime-1.22.1 \
  -DCMAKE_BUILD_TYPE=Release

# 方式二：环境变量（推荐写在 shell profile）
export ORT_PREFIX=$HOME/opt/onnxruntime-1.22.1
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release

# 方式三：头文件与库分离
cmake -S . -B build -G Ninja \
  -DONNXR_INCLUDE_DIR=$HOME/build/onnxruntime/include \
  -DONNXR_LIB_DIR=$HOME/build/onnxruntime/install-aarch64-1.22.1/lib \
  -DCMAKE_BUILD_TYPE=Release
```

要求：`${prefix}/include/onnxruntime/core/session/onnxruntime_c_api.h` 与 `${prefix}/lib/libonnxruntime.*` 存在。

## 3. aarch64 交叉编译流程

仓库自带 `toolchain-aarch64.cmake`，约定使用 `x-tools` 目录结构。假设交叉工具链安装在 `~/x-tools/aarch64-unknown-linux-gnu`，ONNX Runtime 交叉安装在 `~/build/onnxruntime/install-aarch64-1.22.1`。

```bash
export XTOOLS=$HOME/x-tools/aarch64-unknown-linux-gnu
export PATH="$XTOOLS/bin:$PATH"

# 如果 onnxruntime 分开安装，可只设置需要的变量
export ORT_PREFIX=$HOME/build/onnxruntime/install-aarch64-1.22.1          # 可选，包含 lib 与 include
export ORT_INCLUDE_DIR=$HOME/build/onnxruntime/include                    # 若头文件单独存放
export ORT_LIB_DIR=$HOME/build/onnxruntime/install-aarch64-1.22.1/lib     # 若库单独存放

mkdir -p build-aarch64
cmake -S . -B build-aarch64 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=toolchain-aarch64.cmake

cmake --build build-aarch64
```

提示：请在仓库根目录运行上述命令；不要在 `build-aarch64` 目录里再次执行 `cmake -S .`，否则源目录无法找到。

### 3.1 常见问题

- **提示缺少 `onnxruntime_c_api.h`**：检查 `ORT_PREFIX/include/onnxruntime/core/session/onnxruntime_c_api.h` 是否存在；若没有，请重新执行 ONNX Runtime 的 `cmake --build . --target install`，确保安装阶段复制了头文件。
- **ORT 头/库分离**：可同时设置 `ONNXR_INCLUDE_DIR` 与 `ONNXR_LIB_DIR`（或导出 `ORT_INCLUDE_DIR`、`ORT_LIB_DIR`），例如头文件在 `/path/include`，库在 `/path/install/lib`。
- **提示找不到 C/C++ 编译器**：如果出现 “Could not find compiler set in environment variable CC”，说明环境变量 `CC`/`CXX` 指向的编译器不可用。可以 `unset CC CXX` 后重试，或确保交叉编译器所在目录已加入 `PATH`。
- **找不到 `pkg-config`**：可忽略，下一个优先级即 `ORT_PREFIX`。若既无环境变量又无命令行参数，则会报错提醒。
- **链接报缺少 provider 静态库**：某些 ONNX Runtime 构建会拆分多个 `libonnxruntime_*.a`；CMake 已自动扫描并追加，只要它们位于 `${prefix}/lib/` 即可。

## 4. 其它选项

- `-DOWW_USE_PKGCONFIG=OFF`：显式禁用 pkg-config 探测，仅使用 `ONNXR`/`ORT_PREFIX`。
- `-DOWW_ENABLE_SANITIZERS=ON`：主机调试时启用 ASan/UBSan（交叉编译会自动关闭）。
- `-DOWW_BUILD_DEMO=OFF`：跳过 ALSA 演示程序。

## 5. 安装

执行 `cmake --install build --prefix /desired/prefix` 即可将 `liboww.a` 与 `include/oww.h` 复制到目标目录。

---

若仍遇到配置/链接问题，建议将 CMake 输出的 `onnxruntime` 搜索路径、pkg-config 结果与本教程核对，或直接在 `build` 目录下查阅 `CMakeCache.txt` 验证变量值。
