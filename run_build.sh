#!/usr/bin/env zsh

export NUM_CMAKE_JOBS=4
cmake -B build
cmake --build build --config Release --parallel ${NUM_CMAKE_JOBS}