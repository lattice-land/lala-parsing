cmake_minimum_required(VERSION 3.20)

include(FetchContent)
include(CMakePrintHelpers)

project(lala-parsing LANGUAGES CXX)

if(NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

# Cuda-battery dependency

FetchContent_Declare(
  cuda_battery
  GIT_REPOSITORY "https://github.com/lattice-land/cuda-battery.git"
  GIT_TAG        4c63ed2f3e529d95afe367ee07abd6ce68e0d045
)
FetchContent_MakeAvailable(cuda_battery)

FetchContent_Declare(
  lala_core
  GIT_REPOSITORY "https://github.com/lattice-land/lala-core.git"
  GIT_TAG        4f2f465df5a979f008cf8d541c486b2386a498f2
)
FetchContent_MakeAvailable(lala_core)

# XCSP3 C++ Parser dependency

find_package(LibXml2 REQUIRED)

FetchContent_Declare(
    xcsp3parser
    GIT_REPOSITORY    "https://github.com/xcsp3team/XCSP3-CPP-Parser"
    GIT_TAG           50ee7bfa59327c31a0a66ca4072a23cd08d0b182
)
FetchContent_MakeAvailable(xcsp3parser)

# Preparing the library

add_library(lala_parsing INTERFACE)
target_include_directories(lala_parsing INTERFACE include)

target_compile_options(lala_parsing INTERFACE
    $<$<CONFIG:Debug>:-g -DDEBUG>
    $<$<CONFIG:LDebug>:-g -DTRACE>
)

# Thrust dependency

FetchContent_Declare(
  thrust
  GIT_REPOSITORY "https://github.com/NVIDIA/thrust.git"
  GIT_TAG        1.12.0
)
FetchContent_GetProperties(thrust)
if(NOT thrust_POPULATED)
    FetchContent_Populate(thrust)
endif()

# Example binary

add_executable(example_xcsp3 example/xcsp3/xcsp3.cpp)
target_include_directories(example_xcsp3 PRIVATE ${LIBXML2_INCLUDE_DIR} ${lala_parsing_SOURCE_DIR}/include ${xcsp3parser_SOURCE_DIR}/include ${cuda_battery_SOURCE_DIR}/include ${lala_core_SOURCE_DIR}/include ${thrust_SOURCE_DIR})
target_link_libraries(example_xcsp3 PRIVATE xcsp3parser ${LIBXML2_LIBRARIES} cuda_battery lala_core lala_parsing)

# Documentation

find_package(Doxygen REQUIRED doxygen)

# Options based on those of Halide: https://github.com/halide/Halide/blob/feb81a2a0f7bc39cfa499e4e1a7dfedbc5250aaa/doc/CMakeLists.txt
set(DOXYGEN_PROJECT_NAME "Parsing of Combinatorial Model Formats")
set(DOXYGEN_BUILTIN_STL_SUPPORT YES)
set(DOXYGEN_CASE_SENSE_NAMES NO)
set(DOXYGEN_CLASS_DIAGRAMS NO)
set(DOXYGEN_DISTRIBUTE_GROUP_DOC YES)
set(DOXYGEN_EXTRACT_ALL YES)
set(DOXYGEN_FILE_PATTERNS *.hpp)
set(DOXYGEN_GENERATE_TREEVIEW YES)
set(DOXYGEN_HIDE_IN_BODY_DOCS YES)
set(DOXYGEN_QUIET YES)
set(DOXYGEN_RECURSIVE YES)
set(DOXYGEN_REFERENCED_BY_RELATION YES)
set(DOXYGEN_REFERENCES_RELATION YES)
set(DOXYGEN_SORT_BY_SCOPE_NAME YES)
set(DOXYGEN_SORT_MEMBER_DOCS NO)
set(DOXYGEN_SOURCE_BROWSER NO)
set(DOXYGEN_STRIP_CODE_COMMENTS NO)
set(DOXYGEN_USE_MATHJAX YES)
set(DOXYGEN_EXTRA_PACKAGES stmaryrd)

doxygen_add_docs(doc_lalaparsing
                 "include/"
                 ALL
                 COMMENT "Generate HTML documentation")
