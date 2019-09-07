#  minimum required cmake version: 3.1.0
cmake_minimum_required(VERSION 3.1.0)

project(RealsenseCV)

find_package(OpenCV REQUIRED)

set(DEPENDENCIES realsense2 ${OpenCV_LIBS})

add_subdirectory(grabcuts)
add_subdirectory(mask)
