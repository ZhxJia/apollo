#!/usr/bin/env bash
mkdir build
cd build 
cmake ..
make all
make install

cd ..
mkdir caffe-output
cp -r build/install/include caffe-output
cp -r build/lib caffe-output
