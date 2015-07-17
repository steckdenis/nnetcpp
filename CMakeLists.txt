project(nnetcpp)
add_definitions(-std=c++11)

include(GNUInstallDirs)
find_package(PkgConfig REQUIRED)
pkg_check_modules(EIGEN3 eigen3 REQUIRED)
pkg_check_modules(CPPUNIT REQUIRED cppunit)

include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${EIGEN3_INCLUDE_DIRS})

set(nnetcpp_SRCS
    network.cpp
    dense.cpp
    abstractmergenode.cpp
    mergesum.cpp
    mergeproduct.cpp
    gru.cpp
)
add_library(nnetcpp ${nnetcpp_SRCS})
install(TARGETS nnetcpp ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

add_subdirectory(tests)
