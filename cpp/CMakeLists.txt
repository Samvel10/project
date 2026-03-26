cmake_minimum_required(VERSION 3.16)
project(binance_fast LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Release optimizations
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")

# Find required packages
find_package(OpenSSL REQUIRED)
find_package(CURL REQUIRED)
find_package(pybind11 REQUIRED)

# Build the pybind11 module
pybind11_add_module(binance_fast
    bindings.cpp
    binance_filters.cpp
    binance_sign.cpp
    binance_params.cpp
    binance_http.cpp
)

target_include_directories(binance_fast PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OPENSSL_INCLUDE_DIR}
    ${CURL_INCLUDE_DIRS}
)

target_link_libraries(binance_fast PRIVATE
    OpenSSL::SSL
    OpenSSL::Crypto
    CURL::libcurl
)

# Install the built .so into the project execution/ directory
# so Python can import it as: from execution.binance_fast import ...
install(TARGETS binance_fast
    LIBRARY DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/../execution/
)
