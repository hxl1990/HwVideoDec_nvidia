cmake_minimum_required(VERSION 3.7)

project(videodecoderDemo)


set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Release)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include   /usr/local/cuda/include)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

set(APP_SOURCES
 ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
)

find_package(OpenCV
        PATHS /usr/local
        NO_DEFAULT_PATH
        REQUIRED)
if (OpenCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS})
    message( ${OpenCV_LIBS})
else()
    message("OpenCV not found, so we won't build the project.")
endif()


add_executable(${PROJECT_NAME}   ${APP_SOURCES})
target_link_libraries(${PROJECT_NAME}   videodecoder ${OpenCV_LIBS}) 

