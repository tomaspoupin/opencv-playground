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

link_libraries(${OpenCV_LIBS})

# opencv_playground library
add_library(OPENCV_PLAYGROUND 
  src/utils/algorithms/segmentation.cpp
  src/classes/mog.cpp)
target_include_directories(OPENCV_PLAYGROUND PUBLIC include)

add_executable( COLOR_REDUCTION src/basics/color_reduction.cpp )
add_executable( BASIC_FILTERS src/basics/basic_filters.cpp )
add_executable( EDGE_DETECTION src/basics/edge_detection.cpp )

add_executable( BG_SUBSTRACTION src/segmentation/bg_substraction.cpp )
target_link_libraries( BG_SUBSTRACTION PRIVATE OPENCV_PLAYGROUND )

add_executable( TESTS src/tests.cpp )
target_link_libraries( TESTS PRIVATE OPENCV_PLAYGROUND )
