cmake_minimum_required (VERSION 2.6)

project (retinaface_ncnn)

#set(CMAKE_C_FLAGS ${CMAKE_C_FLAGS} "-msse4")
#set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-msse4")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -qopenmp")
add_executable(retinaface_ncnn src/anchor_generator.cpp  src/config.cpp  src/detect.cpp  src/tools.cpp 
src/anchor_generator.h src/config.h src/tools.h)
find_package( OpenCV REQUIRED )
target_link_libraries( retinaface_ncnn ${OpenCV_LIBS} )

set_target_properties(retinaface_ncnn PROPERTIES CXX_STANDARD 11)
find_package(OpenMP)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

include_directories(./include/ncnn)
add_library(ncnn STATIC IMPORTED) 
set_property(TARGET ncnn PROPERTY IMPORTED_LOCATION ../src/libncnn.a)
target_link_libraries(retinaface_ncnn ncnn)

#find_package(Boost REQUIRED COMPONENTS filesystem)
#if(NOT Boost_FOUND)
#    message("Not found Boost")
#endif()
#include_directories(${Boost_INCLUDE_DIRS})
#target_link_libraries(retinaface_ncnn ${Boost_LIBRARIES})
