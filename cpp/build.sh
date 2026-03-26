#!/bin/bash
# Build script for binance_fast C++ module.
# Prerequisites: cmake, g++/clang++, libssl-dev, libcurl4-openssl-dev, pybind11-dev
#
# Install prerequisites (Ubuntu/Debian):
#   apt-get install -y cmake g++ libssl-dev libcurl4-openssl-dev python3-pybind11 pybind11-dev
#   pip install pybind11
#
# Usage:
#   cd /var/www/html/new_example_bot/cpp
#   bash build.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"

echo "=== Building binance_fast C++ module ==="

# Create build directory
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# Configure
cmake "${SCRIPT_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -Dpybind11_DIR="$(python3 -m pybind11 --cmakedir 2>/dev/null || echo '/usr/share/cmake/pybind11')"

# Build
cmake --build . --config Release -j"$(nproc)"

# Copy .so to execution/ directory
SO_FILE=$(find . -name "binance_fast*.so" | head -1)
if [ -n "${SO_FILE}" ]; then
    cp "${SO_FILE}" "${SCRIPT_DIR}/../execution/"
    echo "=== Installed: execution/$(basename ${SO_FILE}) ==="
else
    echo "ERROR: binance_fast.so not found after build"
    exit 1
fi

echo "=== Build complete ==="
