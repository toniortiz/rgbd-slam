cmake_minimum_required(VERSION 2.8)
project(vslam)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Wall   -O3 -march=native -pthread")

#------------------------------------------
#                    G2O
#------------------------------------------
list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )
find_package( G2O REQUIRED )
include_directories(${G2O_INCLUDE_DIR})
find_package( Cholmod )
include_directories(${CHOLMOD_INCLUDE_DIR})
find_package(CSparse)
include_directories(${CSPARSE_INCLUDE_DIR})
list(APPEND G2O_LIBS
  g2o_cli g2o_ext_freeglut_minimal g2o_simulator
  g2o_solver_slam2d_linear g2o_types_icp g2o_types_slam2d
  g2o_core g2o_interface g2o_solver_csparse g2o_solver_structure_only
  g2o_types_sba g2o_types_slam3d g2o_csparse_extension
  g2o_opengl_helper g2o_solver_dense g2o_stuff
  g2o_types_sclam2d g2o_parser g2o_solver_pcg
  g2o_types_data g2o_types_sim3
)

#------------------------------------------
#                    OpenCV
#------------------------------------------
find_package( OpenCV REQUIRED )
include_directories(${OPENCV_INCLUDE_DIRS})

#------------------------------------------
#                    Eigen
#------------------------------------------
find_package(Eigen3 3.0 REQUIRED NO_MODULE)
include_directories(${EIGEN3_INCLUDE_DIR})

#------------------------------------------
#                    DBoW3
#------------------------------------------
find_package( DBoW3 REQUIRED )
include_directories( ${DBoW3_INCLUDE_DIRS} )

#------------------------------------------
#                    Pangolin
#------------------------------------------
find_package(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})

#------------------------------------------
#                    PCL
#------------------------------------------
find_package(PCL REQUIRED COMPONENTS common registration visualization)
add_definitions(${PCL_DEFINITIONS})
include_directories(${PCL_INCLUDE_DIRS})
link_libraries(${PCL_LIBRARY_DIRS})

#------------------------------------------
#                    OpenGL
#------------------------------------------
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
include_directories(${OPENGL_INCLUDE_DIRS} ${GLUT_INCLUDE_DIRS})

#------------------------------------------
#                    Octomap
#------------------------------------------
find_package(octomap)
include_directories(${OCTOMAP_INCLUDE_DIRS})

#------------------------------------------
#                    fast
#------------------------------------------
FIND_PACKAGE(fast REQUIRED)
include_directories(${fast_INCLUDE_DIRS})

#------------------------------------------
#                    opengv
#------------------------------------------
find_package(opengv REQUIRED)

#------------------------------------------
#                    Qt
#------------------------------------------
#set(CMAKE_AUTOMOC ON)
#set(CMAKE_AUTOUIC ON)
#find_package(Qt5Widgets)
#find_package(Qt5Core)

include_directories(${PROJECT_SOURCE_DIR})

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_library(${PROJECT_NAME} SHARED
    System/Tracking.cpp
    System/Converter.cpp
    System/Utility.cpp
    System/Random.cpp

    Features/Extractor.cpp
    Features/ORBextractor.cpp
    Features/Matcher.cpp
    Features/ExtractorNode.cpp
    Features/SVOextractor.cpp
    Features/DetectorAdjuster.cpp
    Features/StatefulFeatureDetector.cpp
    Features/VideoDynamicAdaptedFeatureDetector.cpp
    Features/VideoGridAdaptedFeatureDetector.cpp

    Core/RGBDcamera.cpp
    Core/IntrinsicMatrix.cpp
    Core/Frame.cpp
    Core/Map.cpp
    Core/Landmark.cpp
    Core/GraphNode.cpp

    Solver/SolverSE3.cpp
    Solver/Icp.cpp
    Solver/PoseGraph.cpp
    Solver/PnPRansac.cpp
    Solver/Solver.cpp
    Solver/PnPSolver.cpp
    Solver/Ransac.cpp
    Solver/Gicp.cpp
    Solver/PnP.cpp

    PlaceRecognition/LoopDetector.cpp

    Drawer/Viewer.cpp
    Drawer/MapDrawer.cpp
    Drawer/OctomapDrawer.cpp

    IO/Dataset.cpp
    IO/DatasetTUM.cpp
    IO/DatasetICL.cpp
    IO/DatasetCORBS.cpp
)

target_link_libraries(${PROJECT_NAME}
${OpenCV_LIBS}
${EIGEN3_LIBS}
${PCL_LIBRARIES}
${Pangolin_LIBRARIES}
${DBoW3_LIBS}
${BOOST_LIBRARIES}
${OPENGL_LIBRARIES}
${GLUT_LIBRARY}
${G2O_LIBS}
${CHOLMOD_LIBRARIES}
${CSPARSE_LIBRARY}
${OCTOMAP_LIBRARIES}
${fast_LIBRARIES}
opengv
#Qt5::Widgets
#Qt5::Core
)

add_executable(main main.cpp)
target_link_libraries(main
    ${PROJECT_NAME}
)
