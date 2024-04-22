# FetchContent command is available with cmake >= 3.11
include(FetchContent)

# -----------------------------------------------------------------------------
# helper funtion to download a 'fetch content declare'
# -----------------------------------------------------------------------------
macro(fetch what)
    FetchContent_GetProperties("${what}")
    if(NOT ${${what}_POPULATED})
        message(STATUS "fetching ${what} ...")
        FetchContent_Populate(${what})
    endif()
    mark_as_advanced(${${what}_SOURCE_DIR})
endmacro()

#-----------------------------------------------------------------------------
# libccd
option(BUILD_SHARED_LIBS "Build libccd as a shared library" OFF)
FetchContent_Declare(
        libccd
        GIT_REPOSITORY https://github.com/danfis/libccd.git
        GIT_TAG v2.1
)
fetch(libccd)
add_subdirectory(${libccd_SOURCE_DIR} libccd)

#------------------------------------------------------------------------------
# nanobind
FetchContent_Declare(
        nanobind
        GIT_REPOSITORY https://github.com/wjakob/nanobind.git
        GIT_TAG v1.9.2
)
fetch(nanobind)
add_subdirectory(${nanobind_SOURCE_DIR} nanobind)

#-------------------------------------------------------------------------------
# triangle
FetchContent_Declare(
        triangle
        GIT_REPOSITORY https://github.com/libigl/triangle.git
)
fetch(triangle)
add_subdirectory(${triangle_SOURCE_DIR} triangle)