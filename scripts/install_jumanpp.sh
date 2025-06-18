#!/bin/bash

set -e

# Check if running as root (in Docker) or needs sudo
if [ "$EUID" -eq 0 ]; then
    SUDO=""
else
    SUDO="sudo"
fi

# Update and install dependencies
$SUDO apt update -q
$SUDO apt install -qy cmake g++ make wget xz-utils curl

# Create temporary directory
TEMP_DIR=$(mktemp -d)
cd "$TEMP_DIR"

# Download Juman++ source
wget "https://github.com/ku-nlp/jumanpp/releases/download/v2.0.0-rc3/jumanpp-2.0.0-rc3.tar.xz"
tar xvf jumanpp-2.0.0-rc3.tar.xz

# Build and install Juman++
cd jumanpp-2.0.0-rc3

# Fix CMake version requirement in all CMakeLists.txt files
find . -name "CMakeLists.txt" -exec sed -i 's/cmake_minimum_required(VERSION [0-9.]*)/cmake_minimum_required(VERSION 3.5)/' {} \;

mkdir bld && cd bld

# Download Catch2 header
curl -LO https://github.com/catchorg/Catch2/releases/download/v2.13.8/catch.hpp
mv catch.hpp ../libs/

# Configure with CMake
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local

# Build and install
make -j "$(nproc)"
$SUDO make install

# Update library cache
$SUDO ldconfig

# Clean up
cd /
rm -rf "$TEMP_DIR"

echo "Juman++ installation completed."

# Verify installation
jumanpp --version || echo "Warning: jumanpp --version failed, but installation may be successful"