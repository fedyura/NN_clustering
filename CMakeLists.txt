CMAKE_MINIMUM_REQUIRED (VERSION 2.8)
PROJECT (Soinn)

SET (CMAKE_CXX_FLAGS "-std=c++0x")

ENABLE_TESTING()
FIND_PACKAGE (Boost 1.50.0 COMPONENTS
    filesystem
    iostreams
    regex
    program_options
    system
    thread
    unit_test_framework
REQUIRED)
INCLUDE_DIRECTORIES (${Boost_INCLUDE_DIRS})
INCLUDE_DIRECTORIES (${PROJECT_SOURCE_DIR}/include)
ADD_SUBDIRECTORY (src)
