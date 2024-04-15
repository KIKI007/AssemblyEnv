SET (Boost_USE_MULTITHREADED ON)
include(CMakeFindDependencyMacro)
FIND_DEPENDENCY (Threads)
FIND_DEPENDENCY (Boost 1.74.0 COMPONENTS program_options)



IF (NOT TARGET mqtt_cpp_iface::mqtt_cpp_iface)
  get_filename_component(CURRENT_CMAKE_DIR "${CMAKE_CURRENT_LIST_FILE}" DIRECTORY)
  include("${CURRENT_CMAKE_DIR}/mqtt_cpp_ifaceTargets.cmake")
ENDIF ()

SET (mqtt_cpp_iface_LIBRARIES mqtt_cpp_iface::mqtt_cpp_iface)
