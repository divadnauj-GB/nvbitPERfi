#cmake_minimum_required(VERSION 3.16)
#project(example DESCRIPTION "examples how to use LogHelper lib")

if (NOT TARGET LogHelper)
    find_package(LogHelper REQUIRED)
endif ()
set(CMAKE_BUILD_TYPE Release)

include_directories(example ${CMAKE_SOURCE_DIR}/include)

# Pure C
add_executable(example_c example_log.c)
target_link_libraries(example_c LogHelper)
add_test(NAME ex_c COMMAND example_c)


# WITH_CXX_BACKEND is for old version of lib log helper
#if (WITH_CXX_BACKEND)
# C++
add_executable(example_cxx example_log.cpp)
add_test(NAME ex_cxx COMMAND example_cxx)
target_link_libraries(example_cxx LogHelper)
#endif ()

# Python example
if (WITH_PYTHON_SUPPORT)
    add_test(NAME example_python.py COMMAND example_python.py)
endif ()