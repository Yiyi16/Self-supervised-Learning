#####################################################################################
### REQUIRED STUFF ################################################################## 
##################################################################################### 
cmake_minimum_required(VERSION 2.6)
if(COMMAND cmake_policy)
	cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)
#find_package(OpenMP)
#if(OPENMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
#endif()

#SET(CMAKE_BUILD_TYPE Release)
##################################################################################### 
### SET STUFF ####################################################################### 
#####################################################################################
INCLUDE_DIRECTORIES("/home/panziqi/software/opencv-2.4.9/include")
INCLUDE_DIRECTORIES("/home/panziqi/software/boost_1_55_0/include")
INCLUDE_DIRECTORIES("/home/panziqi/software/dlib-18.7")
INCLUDE_DIRECTORIES("/home/panziqi/software/vlfeat-0.9.16")
INCLUDE_DIRECTORIES("/home/panziqi/project/20190304_action_recognition/DejaVu/src/helpers")
INCLUDE_DIRECTORIES("/home/panziqi/project/20190304_action_recognition/DejaVu/src/structuredRF")
INCLUDE_DIRECTORIES("/home/panziqi/project/20190304_action_recognition/DejaVu/src/structuredRF/evaluation")
INCLUDE_DIRECTORIES("/home/panziqi/project/20190304_action_recognition/DejaVu/src/motionRF")
INCLUDE_DIRECTORIES("/home/panziqi/project/20190304_action_recognition/DejaVu/src/motionRF/evaluation")
INCLUDE_DIRECTORIES("/home/panziqi/project/20190304_action_recognition/DejaVu/src/third")
INCLUDE_DIRECTORIES("/home/panziqi/project/20190304_action_recognition/DejaVu/src/third/sintel")
INCLUDE_DIRECTORIES("/home/panziqi/project/20190304_action_recognition/DejaVu/src/third/forest")
INCLUDE_DIRECTORIES("/home/panziqi/project/20190304_action_recognition/DejaVu/src/third/vlfeat-0.9.16")

# Path to the sources
SET(CMAKE_INSTALL_PREFIX /home/panziqi/project/20190304_action_recognition/DejaVu CACHE PATH "Prefix path to where stuff is")

#PROJECT TO BE BUILT_______________________________________________________ 
PROJECT(dejavu)

# Paths to project stuff___________________________________________________
SET(SRC ${CMAKE_INSTALL_PREFIX}/src CACHE PATH "Location of the sources")
SET(EXE ${CMAKE_INSTALL_PREFIX}/bin CACHE PATH "Location of the binary files")

# Paths to opencv stuff____________________________________________________
SET(OPENCV_INC /home/panziqi/software/opencv-2.4.9/include CACHE PATH "Location of the OpenCV2.4.9 headers")
SET(OPENCV_LIB /home/panziqi/software/opencv-2.4.9/lib CACHE PATH "Location of the OpenCV2.4.9 libraries")

# Paths to boost stuff______________________________________________________ 
SET(BOOST_LIB /home/panziqi/software/boost_1_55_0/lib "Location of boost libraries")
SET(BOOST_INC /home/panziqi/software/boost_1_55_0/include "Boost include directory")

# Path for VlFeat___________________________________________________________
SET(VLFEAT_LIB /home/panziqi/software/vlfeat-0.9.16/bin/glnxa64 CACHE PATH "Location of the VLfit")

# Path to the Dlib stuff_____________________________________________________
SET(DLIB_INC /home/panziqi/software/dlib-18.7 CACHE PATH "Location of the dlib headers")
SET(DLIB_LIB /home/panziqi/software/dlib-18.7/examples/build/dlib_build CACHE PATH "Location of the dlib libraries")
##################################################################################### 
### INCLUDE & LINK STUFF ############################################################ 
#####################################################################################
LINK_DIRECTORIES("/home/panziqi/software/opencv-2.4.9/lib")
LINK_DIRECTORIES("/home/panziqi/software/dlib-18.7/examples/build/dlib_build")
LINK_DIRECTORIES("/home/panziqi/software/boost_1_55_0/lib")
LINK_DIRECTORIES("/home/panziqi/software/vlfeat-0.9.16/bin/glnxa64")

ADD_LIBRARY(helpers  
	${SRC}/helpers/Auxiliary.cpp
	${SRC}/helpers/Tree.cpp
	${SRC}/helpers/SIFTlike.cpp
)
ADD_LIBRARY(sintel
	${SRC}/third/sintel/flowIO.cpp
	${SRC}/third/sintel/imageLib/Image.cpp
	${SRC}/third/sintel/imageLib/Convert.cpp
	${SRC}/third/sintel/imageLib/Convolve.cpp
	${SRC}/third/sintel/imageLib/ImageIO.cpp
	${SRC}/third/sintel/imageLib/ImageIOpng.cpp
	${SRC}/third/sintel/imageLib/RefCntMem.cpp
)
ADD_LIBRARY(forest
	${SRC}/third/forest/CRPatch.cpp
	${SRC}/third/forest/HoG.cpp	
)
ADD_LIBRARY(structuredRF
	${SRC}/structuredRF/Puzzle.cpp
	${SRC}/structuredRF/RunRF.cpp
	${SRC}/structuredRF/StructuredRFdetector.cpp
	${SRC}/structuredRF/StructuredRF.cpp
	${SRC}/structuredRF/StructuredTree.cpp
	${SRC}/structuredRF/StructuredTreeNode.cpp
	${SRC}/structuredRF/StructuredPatch.cpp
)
ADD_LIBRARY(motionRF
	${SRC}/motionRF/RunMotionRF.cpp	
	${SRC}/motionRF/MotionRFdetector.cpp
	${SRC}/motionRF/MotionRF.cpp	
	${SRC}/motionRF/MotionPuzzle.cpp
	${SRC}/motionRF/MotionTree.cpp
	${SRC}/motionRF/MotionTreeNode.cpp
	${SRC}/motionRF/MotionPatch.cpp
	${SRC}/motionRF/MotionPatchFeature.cpp
)
ADD_LIBRARY(evaluation
	${SRC}/structuredRF/evaluation/LabelEval.cpp
	${SRC}/motionRF/evaluation/MotionEval.cpp
)

##################################################################################### 
### EXECUTABLE STUFF ################################################################ 
##################################################################################### 
ADD_EXECUTABLE(dejavu
	${SRC}/motionMain.cpp 
)
TARGET_LINK_LIBRARIES(dejavu
	dlib
	motionRF
	structuredRF
	forest
	sintel
	helpers
        vl 
	evaluation
	opencv_core 
	opencv_video
	opencv_calib3d 
	opencv_imgproc 
	opencv_highgui
        opencv_ml
	opencv_features2d
	opencv_flann
	opencv_nonfree
	opencv_legacy	
	boost_date_time-mt 
	boost_thread-mt
	boost_filesystem
	boost_system
)	
