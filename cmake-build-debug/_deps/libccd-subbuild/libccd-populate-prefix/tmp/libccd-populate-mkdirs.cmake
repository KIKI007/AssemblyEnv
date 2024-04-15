# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/Users/ziqwang/Documents/GitHub/AssemblyEnv/cmake-build-debug/_deps/libccd-src"
  "/Users/ziqwang/Documents/GitHub/AssemblyEnv/cmake-build-debug/_deps/libccd-build"
  "/Users/ziqwang/Documents/GitHub/AssemblyEnv/cmake-build-debug/_deps/libccd-subbuild/libccd-populate-prefix"
  "/Users/ziqwang/Documents/GitHub/AssemblyEnv/cmake-build-debug/_deps/libccd-subbuild/libccd-populate-prefix/tmp"
  "/Users/ziqwang/Documents/GitHub/AssemblyEnv/cmake-build-debug/_deps/libccd-subbuild/libccd-populate-prefix/src/libccd-populate-stamp"
  "/Users/ziqwang/Documents/GitHub/AssemblyEnv/cmake-build-debug/_deps/libccd-subbuild/libccd-populate-prefix/src"
  "/Users/ziqwang/Documents/GitHub/AssemblyEnv/cmake-build-debug/_deps/libccd-subbuild/libccd-populate-prefix/src/libccd-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/Users/ziqwang/Documents/GitHub/AssemblyEnv/cmake-build-debug/_deps/libccd-subbuild/libccd-populate-prefix/src/libccd-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/Users/ziqwang/Documents/GitHub/AssemblyEnv/cmake-build-debug/_deps/libccd-subbuild/libccd-populate-prefix/src/libccd-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
