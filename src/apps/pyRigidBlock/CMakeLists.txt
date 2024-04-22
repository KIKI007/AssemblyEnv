cmake_minimum_required(VERSION 3.5)
project(py_rigidblock)
find_package(Python 3.12 COMPONENTS Interpreter Development REQUIRED)

set(CRL_TARGET_NAME ${PROJECT_NAME})

file(
    GLOB
    CRL_SOURCES #
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp" #
)

# basic dependencies
list(
    APPEND
    CRL_TARGET_DEPENDENCIES #
    "rigid_block"
)

# target include dirs
list(APPEND CRL_TARGET_INCLUDE_DIRS #
     PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

# target link libs
list(
    APPEND
    CRL_TARGET_LINK_LIBS #
    PUBLIC
    "rigid_block"
)

# compile definitions
list(APPEND
    CRL_COMPILE_DEFINITIONS #
     ""
)

# create target
nanobind_add_module(${CRL_TARGET_NAME} ${CRL_SOURCES})
add_dependencies(${CRL_TARGET_NAME} ${CRL_TARGET_DEPENDENCIES})
target_include_directories(${CRL_TARGET_NAME} ${CRL_TARGET_INCLUDE_DIRS})
target_link_libraries(${CRL_TARGET_NAME} ${CRL_TARGET_LINK_LIBS})
set_target_properties(${CRL_TARGET_NAME} PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${ASEEMBLYENV_TOPLEVEL_FOLDER}/python/AssemblyEnv")
set_target_properties(${CRL_TARGET_NAME} PROPERTIES CXX_VISIBILITY_PRESET hidden)
