# Install script for directory: /Users/ziqwang/Documents/GitHub/AssemblyEnv/cmake-build-debug/_deps/mqtt-src/include

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
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES
    "/Users/ziqwang/Documents/GitHub/AssemblyEnv/cmake-build-debug/_deps/mqtt-src/include/mqtt_client_cpp.hpp"
    "/Users/ziqwang/Documents/GitHub/AssemblyEnv/cmake-build-debug/_deps/mqtt-src/include/mqtt_server_cpp.hpp"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/mqtt" TYPE FILE FILES
    "/Users/ziqwang/Documents/GitHub/AssemblyEnv/cmake-build-debug/_deps/mqtt-src/include/mqtt/any.hpp"
    "/Users/ziqwang/Documents/GitHub/AssemblyEnv/cmake-build-debug/_deps/mqtt-src/include/mqtt/async_client.hpp"
    "/Users/ziqwang/Documents/GitHub/AssemblyEnv/cmake-build-debug/_deps/mqtt-src/include/mqtt/attributes.hpp"
    "/Users/ziqwang/Documents/GitHub/AssemblyEnv/cmake-build-debug/_deps/mqtt-src/include/mqtt/buffer.hpp"
    "/Users/ziqwang/Documents/GitHub/AssemblyEnv/cmake-build-debug/_deps/mqtt-src/include/mqtt/callable_overlay.hpp"
    "/Users/ziqwang/Documents/GitHub/AssemblyEnv/cmake-build-debug/_deps/mqtt-src/include/mqtt/client.hpp"
    "/Users/ziqwang/Documents/GitHub/AssemblyEnv/cmake-build-debug/_deps/mqtt-src/include/mqtt/config.hpp"
    "/Users/ziqwang/Documents/GitHub/AssemblyEnv/cmake-build-debug/_deps/mqtt-src/include/mqtt/connect_flags.hpp"
    "/Users/ziqwang/Documents/GitHub/AssemblyEnv/cmake-build-debug/_deps/mqtt-src/include/mqtt/connect_return_code.hpp"
    "/Users/ziqwang/Documents/GitHub/AssemblyEnv/cmake-build-debug/_deps/mqtt-src/include/mqtt/const_buffer_util.hpp"
    "/Users/ziqwang/Documents/GitHub/AssemblyEnv/cmake-build-debug/_deps/mqtt-src/include/mqtt/constant.hpp"
    "/Users/ziqwang/Documents/GitHub/AssemblyEnv/cmake-build-debug/_deps/mqtt-src/include/mqtt/control_packet_type.hpp"
    "/Users/ziqwang/Documents/GitHub/AssemblyEnv/cmake-build-debug/_deps/mqtt-src/include/mqtt/deprecated.hpp"
    "/Users/ziqwang/Documents/GitHub/AssemblyEnv/cmake-build-debug/_deps/mqtt-src/include/mqtt/deprecated_msg.hpp"
    "/Users/ziqwang/Documents/GitHub/AssemblyEnv/cmake-build-debug/_deps/mqtt-src/include/mqtt/endpoint.hpp"
    "/Users/ziqwang/Documents/GitHub/AssemblyEnv/cmake-build-debug/_deps/mqtt-src/include/mqtt/error_code.hpp"
    "/Users/ziqwang/Documents/GitHub/AssemblyEnv/cmake-build-debug/_deps/mqtt-src/include/mqtt/exception.hpp"
    "/Users/ziqwang/Documents/GitHub/AssemblyEnv/cmake-build-debug/_deps/mqtt-src/include/mqtt/fixed_header.hpp"
    "/Users/ziqwang/Documents/GitHub/AssemblyEnv/cmake-build-debug/_deps/mqtt-src/include/mqtt/four_byte_util.hpp"
    "/Users/ziqwang/Documents/GitHub/AssemblyEnv/cmake-build-debug/_deps/mqtt-src/include/mqtt/hexdump.hpp"
    "/Users/ziqwang/Documents/GitHub/AssemblyEnv/cmake-build-debug/_deps/mqtt-src/include/mqtt/log.hpp"
    "/Users/ziqwang/Documents/GitHub/AssemblyEnv/cmake-build-debug/_deps/mqtt-src/include/mqtt/message.hpp"
    "/Users/ziqwang/Documents/GitHub/AssemblyEnv/cmake-build-debug/_deps/mqtt-src/include/mqtt/message_variant.hpp"
    "/Users/ziqwang/Documents/GitHub/AssemblyEnv/cmake-build-debug/_deps/mqtt-src/include/mqtt/move.hpp"
    "/Users/ziqwang/Documents/GitHub/AssemblyEnv/cmake-build-debug/_deps/mqtt-src/include/mqtt/namespace.hpp"
    "/Users/ziqwang/Documents/GitHub/AssemblyEnv/cmake-build-debug/_deps/mqtt-src/include/mqtt/null_strand.hpp"
    "/Users/ziqwang/Documents/GitHub/AssemblyEnv/cmake-build-debug/_deps/mqtt-src/include/mqtt/optional.hpp"
    "/Users/ziqwang/Documents/GitHub/AssemblyEnv/cmake-build-debug/_deps/mqtt-src/include/mqtt/packet_id_manager.hpp"
    "/Users/ziqwang/Documents/GitHub/AssemblyEnv/cmake-build-debug/_deps/mqtt-src/include/mqtt/packet_id_type.hpp"
    "/Users/ziqwang/Documents/GitHub/AssemblyEnv/cmake-build-debug/_deps/mqtt-src/include/mqtt/property.hpp"
    "/Users/ziqwang/Documents/GitHub/AssemblyEnv/cmake-build-debug/_deps/mqtt-src/include/mqtt/property_id.hpp"
    "/Users/ziqwang/Documents/GitHub/AssemblyEnv/cmake-build-debug/_deps/mqtt-src/include/mqtt/property_parse.hpp"
    "/Users/ziqwang/Documents/GitHub/AssemblyEnv/cmake-build-debug/_deps/mqtt-src/include/mqtt/property_variant.hpp"
    "/Users/ziqwang/Documents/GitHub/AssemblyEnv/cmake-build-debug/_deps/mqtt-src/include/mqtt/protocol_version.hpp"
    "/Users/ziqwang/Documents/GitHub/AssemblyEnv/cmake-build-debug/_deps/mqtt-src/include/mqtt/publish.hpp"
    "/Users/ziqwang/Documents/GitHub/AssemblyEnv/cmake-build-debug/_deps/mqtt-src/include/mqtt/reason_code.hpp"
    "/Users/ziqwang/Documents/GitHub/AssemblyEnv/cmake-build-debug/_deps/mqtt-src/include/mqtt/remaining_length.hpp"
    "/Users/ziqwang/Documents/GitHub/AssemblyEnv/cmake-build-debug/_deps/mqtt-src/include/mqtt/server.hpp"
    "/Users/ziqwang/Documents/GitHub/AssemblyEnv/cmake-build-debug/_deps/mqtt-src/include/mqtt/session_present.hpp"
    "/Users/ziqwang/Documents/GitHub/AssemblyEnv/cmake-build-debug/_deps/mqtt-src/include/mqtt/setup_log.hpp"
    "/Users/ziqwang/Documents/GitHub/AssemblyEnv/cmake-build-debug/_deps/mqtt-src/include/mqtt/shared_ptr_array.hpp"
    "/Users/ziqwang/Documents/GitHub/AssemblyEnv/cmake-build-debug/_deps/mqtt-src/include/mqtt/shared_scope_guard.hpp"
    "/Users/ziqwang/Documents/GitHub/AssemblyEnv/cmake-build-debug/_deps/mqtt-src/include/mqtt/shared_subscriptions.hpp"
    "/Users/ziqwang/Documents/GitHub/AssemblyEnv/cmake-build-debug/_deps/mqtt-src/include/mqtt/store.hpp"
    "/Users/ziqwang/Documents/GitHub/AssemblyEnv/cmake-build-debug/_deps/mqtt-src/include/mqtt/strand.hpp"
    "/Users/ziqwang/Documents/GitHub/AssemblyEnv/cmake-build-debug/_deps/mqtt-src/include/mqtt/string_check.hpp"
    "/Users/ziqwang/Documents/GitHub/AssemblyEnv/cmake-build-debug/_deps/mqtt-src/include/mqtt/string_view.hpp"
    "/Users/ziqwang/Documents/GitHub/AssemblyEnv/cmake-build-debug/_deps/mqtt-src/include/mqtt/subscribe_entry.hpp"
    "/Users/ziqwang/Documents/GitHub/AssemblyEnv/cmake-build-debug/_deps/mqtt-src/include/mqtt/subscribe_options.hpp"
    "/Users/ziqwang/Documents/GitHub/AssemblyEnv/cmake-build-debug/_deps/mqtt-src/include/mqtt/sync_client.hpp"
    "/Users/ziqwang/Documents/GitHub/AssemblyEnv/cmake-build-debug/_deps/mqtt-src/include/mqtt/tcp_endpoint.hpp"
    "/Users/ziqwang/Documents/GitHub/AssemblyEnv/cmake-build-debug/_deps/mqtt-src/include/mqtt/time_point_t.hpp"
    "/Users/ziqwang/Documents/GitHub/AssemblyEnv/cmake-build-debug/_deps/mqtt-src/include/mqtt/tls.hpp"
    "/Users/ziqwang/Documents/GitHub/AssemblyEnv/cmake-build-debug/_deps/mqtt-src/include/mqtt/topic_alias_recv.hpp"
    "/Users/ziqwang/Documents/GitHub/AssemblyEnv/cmake-build-debug/_deps/mqtt-src/include/mqtt/topic_alias_send.hpp"
    "/Users/ziqwang/Documents/GitHub/AssemblyEnv/cmake-build-debug/_deps/mqtt-src/include/mqtt/two_byte_util.hpp"
    "/Users/ziqwang/Documents/GitHub/AssemblyEnv/cmake-build-debug/_deps/mqtt-src/include/mqtt/two_or_four_byte_util.hpp"
    "/Users/ziqwang/Documents/GitHub/AssemblyEnv/cmake-build-debug/_deps/mqtt-src/include/mqtt/type.hpp"
    "/Users/ziqwang/Documents/GitHub/AssemblyEnv/cmake-build-debug/_deps/mqtt-src/include/mqtt/type_erased_socket.hpp"
    "/Users/ziqwang/Documents/GitHub/AssemblyEnv/cmake-build-debug/_deps/mqtt-src/include/mqtt/unique_scope_guard.hpp"
    "/Users/ziqwang/Documents/GitHub/AssemblyEnv/cmake-build-debug/_deps/mqtt-src/include/mqtt/utf8encoded_strings.hpp"
    "/Users/ziqwang/Documents/GitHub/AssemblyEnv/cmake-build-debug/_deps/mqtt-src/include/mqtt/v5_message.hpp"
    "/Users/ziqwang/Documents/GitHub/AssemblyEnv/cmake-build-debug/_deps/mqtt-src/include/mqtt/value_allocator.hpp"
    "/Users/ziqwang/Documents/GitHub/AssemblyEnv/cmake-build-debug/_deps/mqtt-src/include/mqtt/variable_length.hpp"
    "/Users/ziqwang/Documents/GitHub/AssemblyEnv/cmake-build-debug/_deps/mqtt-src/include/mqtt/variant.hpp"
    "/Users/ziqwang/Documents/GitHub/AssemblyEnv/cmake-build-debug/_deps/mqtt-src/include/mqtt/variant_visit.hpp"
    "/Users/ziqwang/Documents/GitHub/AssemblyEnv/cmake-build-debug/_deps/mqtt-src/include/mqtt/visitor_util.hpp"
    "/Users/ziqwang/Documents/GitHub/AssemblyEnv/cmake-build-debug/_deps/mqtt-src/include/mqtt/will.hpp"
    "/Users/ziqwang/Documents/GitHub/AssemblyEnv/cmake-build-debug/_deps/mqtt-src/include/mqtt/ws_endpoint.hpp"
    )
endif()

