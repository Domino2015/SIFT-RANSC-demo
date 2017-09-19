cmake_minimum_required(VERSION 3.8)
project(demo)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)
set(SOURCE_FILES main.cpp)
add_executable(demo ${SOURCE_FILES})

target_link_libraries(demo ${OpenCV_LIBS} )