cmake_minimum_required(VERSION 3.10)
project(VrpEngine)

set(CMAKE_CXX_STANDARD 14)

# Find Python
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Find Pybind11
# We can use `pybind11-config --cmakedir` to find it, or assume standard install
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
find_package(pybind11 REQUIRED)

add_executable(vrp_engine src/main.cpp)
target_link_libraries(vrp_engine PRIVATE pybind11::embed)
