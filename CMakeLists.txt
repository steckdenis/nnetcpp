project(nnetcpp)
add_definitions(-std=c++11)

find_package(PkgConfig REQUIRED)
pkg_check_modules(EIGEN3 eigen3 REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${EIGEN3_INCLUDE_DIRS})

set(nnetcpp_SRCS
    network.cpp
    dense.cpp
    tanh.cpp
)
add_library(nnetcpp ${nnetcpp_SRCS})
