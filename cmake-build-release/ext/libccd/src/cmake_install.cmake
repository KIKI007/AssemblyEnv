# Install script for directory: /Users/ziqwang/Documents/GitHub/AssemblyEnv/cmake-build-release/_deps/libccd-src/src

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES
    "/Users/ziqwang/Documents/GitHub/AssemblyEnv/cmake-build-release/ext/libccd/src/libccd.2.0.dylib"
    "/Users/ziqwang/Documents/GitHub/AssemblyEnv/cmake-build-release/ext/libccd/src/libccd.2.dylib"
    )
  foreach(file
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libccd.2.0.dylib"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libccd.2.dylib"
      )
    if(EXISTS "${file}" AND
       NOT IS_SYMLINK "${file}")
      if(CMAKE_INSTALL_DO_STRIP)
        execute_process(COMMAND "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/strip" -x "${file}")
      endif()
    endif()
  endforeach()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/Users/ziqwang/Documents/GitHub/AssemblyEnv/cmake-build-release/ext/libccd/src/libccd.dylib")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/ccd" TYPE FILE FILES
    "/Users/ziqwang/Documents/GitHub/AssemblyEnv/cmake-build-release/_deps/libccd-src/src/ccd/ccd.h"
    "/Users/ziqwang/Documents/GitHub/AssemblyEnv/cmake-build-release/_deps/libccd-src/src/ccd/compiler.h"
    "/Users/ziqwang/Documents/GitHub/AssemblyEnv/cmake-build-release/_deps/libccd-src/src/ccd/ccd_export.h"
    "/Users/ziqwang/Documents/GitHub/AssemblyEnv/cmake-build-release/_deps/libccd-src/src/ccd/quat.h"
    "/Users/ziqwang/Documents/GitHub/AssemblyEnv/cmake-build-release/_deps/libccd-src/src/ccd/vec3.h"
    "/Users/ziqwang/Documents/GitHub/AssemblyEnv/cmake-build-release/ext/libccd/src/ccd/config.h"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/ccd/ccd-targets.cmake")
    file(DIFFERENT _cmake_export_file_changed FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/ccd/ccd-targets.cmake"
         "/Users/ziqwang/Documents/GitHub/AssemblyEnv/cmake-build-release/ext/libccd/src/CMakeFiles/Export/1ec81c47dcc60201f3455480ce4c19b2/ccd-targets.cmake")
    if(_cmake_export_file_changed)
      file(GLOB _cmake_old_config_files "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/ccd/ccd-targets-*.cmake")
      if(_cmake_old_config_files)
        string(REPLACE ";" ", " _cmake_old_config_files_text "${_cmake_old_config_files}")
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/ccd/ccd-targets.cmake\" will be replaced.  Removing files [${_cmake_old_config_files_text}].")
        unset(_cmake_old_config_files_text)
        file(REMOVE ${_cmake_old_config_files})
      endif()
      unset(_cmake_old_config_files)
    endif()
    unset(_cmake_export_file_changed)
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/ccd" TYPE FILE FILES "/Users/ziqwang/Documents/GitHub/AssemblyEnv/cmake-build-release/ext/libccd/src/CMakeFiles/Export/1ec81c47dcc60201f3455480ce4c19b2/ccd-targets.cmake")
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/ccd" TYPE FILE FILES "/Users/ziqwang/Documents/GitHub/AssemblyEnv/cmake-build-release/ext/libccd/src/CMakeFiles/Export/1ec81c47dcc60201f3455480ce4c19b2/ccd-targets-release.cmake")
  endif()
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/Users/ziqwang/Documents/GitHub/AssemblyEnv/cmake-build-release/ext/libccd/src/testsuites/cmake_install.cmake")

endif()

