# Install script for directory: /Users/ziqwang/Documents/GitHub/AssemblyEnv/cmake-build-debug/_deps/mqtt-src

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
    set(CMAKE_INSTALL_CONFIG_NAME "Debug")
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
  set(CMAKE_OBJDUMP "/Library/Developer/CommandLineTools/usr/bin/objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/mqtt_cpp_iface" TYPE FILE FILES "/Users/ziqwang/Documents/GitHub/AssemblyEnv/cmake-build-debug/ext/mqtt/mqtt_cpp_ifaceConfig.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  include("/Users/ziqwang/Documents/GitHub/AssemblyEnv/cmake-build-debug/ext/mqtt/include/CMakeFiles/mqtt_cpp_iface.dir/install-cxx-module-bmi-Debug.cmake" OPTIONAL)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/mqtt_cpp_iface/mqtt_cpp_ifaceTargets.cmake")
    file(DIFFERENT _cmake_export_file_changed FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/mqtt_cpp_iface/mqtt_cpp_ifaceTargets.cmake"
         "/Users/ziqwang/Documents/GitHub/AssemblyEnv/cmake-build-debug/ext/mqtt/CMakeFiles/Export/d47f6471858d2c1f24d0acd5ddda0901/mqtt_cpp_ifaceTargets.cmake")
    if(_cmake_export_file_changed)
      file(GLOB _cmake_old_config_files "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/mqtt_cpp_iface/mqtt_cpp_ifaceTargets-*.cmake")
      if(_cmake_old_config_files)
        string(REPLACE ";" ", " _cmake_old_config_files_text "${_cmake_old_config_files}")
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/mqtt_cpp_iface/mqtt_cpp_ifaceTargets.cmake\" will be replaced.  Removing files [${_cmake_old_config_files_text}].")
        unset(_cmake_old_config_files_text)
        file(REMOVE ${_cmake_old_config_files})
      endif()
      unset(_cmake_old_config_files)
    endif()
    unset(_cmake_export_file_changed)
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/mqtt_cpp_iface" TYPE FILE FILES "/Users/ziqwang/Documents/GitHub/AssemblyEnv/cmake-build-debug/ext/mqtt/CMakeFiles/Export/d47f6471858d2c1f24d0acd5ddda0901/mqtt_cpp_ifaceTargets.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/Users/ziqwang/Documents/GitHub/AssemblyEnv/cmake-build-debug/ext/mqtt/include/cmake_install.cmake")
  include("/Users/ziqwang/Documents/GitHub/AssemblyEnv/cmake-build-debug/ext/mqtt/test/cmake_install.cmake")
  include("/Users/ziqwang/Documents/GitHub/AssemblyEnv/cmake-build-debug/ext/mqtt/example/cmake_install.cmake")

endif()

