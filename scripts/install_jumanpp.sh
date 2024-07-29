#!/bin/bash

set -e

# Update and install dependencies
sudo apt update -q
sudo apt install -qy cmake g++ make wget xz-utils

# Download Juman++ source
wget "https://github.com/ku-nlp/jumanpp/releases/download/v2.0.0-rc3/jumanpp-2.0.0-rc3.tar.xz"
tar xvf jumanpp-2.0.0-rc3.tar.xz

# Build and install Juman++
cd jumanpp-2.0.0-rc3
mkdir bld && cd bld

curl -LO https://github.com/catchorg/Catch2/releases/download/v2.13.8/catch.hpp
mv catch.hpp ../libs/
cmake .. -DCMAKE_BUILD_TYPE=Release
sudo make install -j "$(nproc)"
sudo make install

# Clean up
cd ../..
rm -rf jumanpp-2.0.0-rc3*

echo "Juman++ installation completed."