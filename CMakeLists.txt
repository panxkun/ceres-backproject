cmake_minimum_required(VERSION 3.0)

project(3dwarp)

set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )


find_package( Ceres REQUIRED )
include_directories( ${CERES_INCLUDE_DIRS} )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_DIRS} )
find_package( Pangolin REQUIRED )
include_directories( ${Pangolin_INCLUDE_DIRS} )

add_executable(ceresBackProject ceresBackProject.cpp)
target_link_libraries( ceresBackProject ${CERES_LIBRARIES} ${OpenCV_LIBS} )

add_executable(visualize visualize.cpp)
target_link_libraries( visualize  ${Pangolin_LIBRARIES} ${OpenCV_LIBS}  )
