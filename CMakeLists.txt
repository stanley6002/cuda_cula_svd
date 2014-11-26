cmake_minimum_required(VERSION 2.8)
project(SVD_Image)

find_package(CUDA QUIET REQUIRED)
 
# Pass options to NVCC
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 -gencode arch=compute_22,code=sm_22)
 

# set up include-directories
include_directories(
   /usr/local/cula/include
   /usr/local/Cellar/opencv/2.4.7.1/include/
   /usr/local/Cellar/opencv/2.4.7.1/include/opencv
   /Developer/NVIDIA/CUDA-6.5/include/
   )  # to find foo/config.h
   
find_library(OPENCV_CORE opencv_core.2.4.7)
find_library(OPENCV_HIGHGUI opencv_highgui.2.4.7)

link_directories(
    /usr/local/cula/lib64/
    /usr/local/cula/include
    /usr/local/lib
    /Developer/NVIDIA/CUDA-6.5/lib
    )
    
# search dylib path    
MESSAGE( STATUS "CMAKE_CURRENT_OPENCV_CORE_DIR: " ${OPENCV_CORE} )
MESSAGE( STATUS "CMAKE_CURRENT_OPENCV_HIGHGUI_DIR: " ${OPENCV_HIGHGUI} )
MESSAGE( STATUS "CMAKE_SOURCE_DIRECTORY: " ${CMAKE_SOURCE_DIR} )
MESSAGE( STATUS "CMAKE_SOURCE_DIRECTORY: " ${CMAKE_SOURCE_DIR} )

cuda_add_executable(
    SVD_Image
    src/svd_main.c
    )

target_link_libraries(
    SVD_Image
    #-lopencv_highgui.2.4.7
    #-lopencv_core.2.4.7
    -lopencv_imgproc.2.4.7
    ${OPENCV_HIGHGUI}
    ${OPENCV_CORE}
    -lcula_core
    -lcula_lapack
    -lcublas
    )


