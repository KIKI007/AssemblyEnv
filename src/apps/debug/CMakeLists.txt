cmake_minimum_required(VERSION 3.5)
project(debug)

set(CRL_TARGET_NAME ${PROJECT_NAME})

file(
    GLOB
    CRL_SOURCES #
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" #
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
create_app(
        ${PROJECT_NAME}
        "${CRL_SOURCES}" #
        "${CRL_TARGET_DEPENDENCIES}" #
        "${CRL_TARGET_INCLUDE_DIRS}" #
        "${CRL_TARGET_LINK_LIBS}" #
        "${CRL_COMPILE_DEFINITIONS}"
)