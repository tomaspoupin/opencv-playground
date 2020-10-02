cmake_minimum_required(VERSION 3.8)

# Change project title
project( PLAYGROUND CXX )

# Optional
# set(CMAKE_CXX_COMPILER "/usr/bin/g++")
set (CMAKE_CXX_STANDARD 14)

# Set executable dir
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

# OpenCV Package
find_package( OpenCV REQUIRED )

add_executable( COLOR_REDUCTION src/color_reduction.cpp )
target_link_libraries( COLOR_REDUCTION ${OpenCV_LIBS} )