cmake_minimum_required(VERSION 3.22...3.30)
project(memory_bandwidth_test)

add_executable(memory_bandwidth_test)
target_sources(memory_bandwidth_test PRIVATE bandwidth.cpp)

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()
target_link_libraries(memory_bandwidth_test PUBLIC TT::Metalium)
