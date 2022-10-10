cmake_minimum_required(VERSION 2.8.12)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

PROJECT(face_detection)

find_package( OpenCV REQUIRED )
find_package( dlib REQUIRED )

#find_library(TENSORFLOW_LIB tensorflow HINT /usr/local/lib)

message(STATUS "  OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

include_directories( ${OpenCV_INCLUDE_DIRS})

find_package(dlib QUIET)
if(dlib_FOUND)
  message(STATUS "  Found system instance of DLib")
else()
  message(STATUS "  Use own instance of DLib")
  # set(dlib_FOUND 1)
  # include(./dlib/dlib/cmake)
endif()

# ------------------------------------------------- #

set( NAME_SRC
    src/main.cpp
    src/image.cpp
    src/video.cpp
    src/functs.cpp
    src/matching.cpp
    src/loadFace.cpp
    src/helpers.cpp
)

set( NAME_SRC_API
    src/api.cpp
    src/image.cpp
    src/video.cpp
    src/functs.cpp
    src/matching.cpp
    src/loadFace.cpp
    src/helpers.cpp
)

set( NAME_HEADERS       
     include/image.h
     include/video.h
     include/functs.h
     include/helpers.h
     include/matching.h
     include/loadFace.h
)

INCLUDE_DIRECTORIES( ${CMAKE_CURRENT_SOURCE_DIR}/include )
link_directories( ${CMAKE_BINARY_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
add_executable( output ${NAME_SRC} ${NAME_HEADERS} )

target_link_libraries( output ${OpenCV_LIBS} )
target_link_libraries( output dlib::dlib )

INCLUDE_DIRECTORIES( ${CMAKE_CURRENT_SOURCE_DIR}/include )
link_directories( ${CMAKE_BINARY_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
add_executable( api ${NAME_SRC_API} ${NAME_HEADERS} )

target_link_libraries( api ${OpenCV_LIBS} )
target_link_libraries( api dlib::dlib )

#set(CMAKE_BUILD_TYPE RelWithDebInfo)