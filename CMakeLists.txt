cmake_minimum_required(VERSION 3.1)
project(AICA LANGUAGES CXX)

set(AICA_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

#add_subdirectory(deps)

add_subdirectory(src)

add_subdirectory(test)