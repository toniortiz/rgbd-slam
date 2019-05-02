#ifndef MAPDRAWER_H
#define MAPDRAWER_H

#include "System/macros.h"
#include <memory>
#include <mutex>
#include <octomap/ColorOcTree.h>
#include <opencv2/core.hpp>
#include <pangolin/pangolin.h>

class Frame;
class Map;
class PoseGraph;
class OctomapDrawer;

class MapDrawer {
public:
    SMART_POINTER_TYPEDEFS(MapDrawer);

public:
    MapDrawer(std::shared_ptr<Map> pMap, std::shared_ptr<PoseGraph> pGraph);

    std::shared_ptr<Map> mpMap;
    std::shared_ptr<PoseGraph> mpGraph;

    void drawLandmarks();
    void drawOctomap();
    void drawPoseGraph(const bool bDrawVertices, const bool bDrawEdges);
    void drawCurrentCamera(pangolin::OpenGlMatrix& Twc);
    void setCurrentCameraPose(const cv::Mat& Tcw);
    void getCurrentOpenGLCameraMatrix(pangolin::OpenGlMatrix& M);

private:
    float mKeyFrameSize;
    float mPointSize;
    float mKeyFrameLineWidth;
    float mGraphLineWidth;
    float mCameraSize;
    float mCameraLineWidth;

    cv::Mat mCameraPose;

    std::shared_ptr<OctomapDrawer> mpOctomapDrawer;

    int mLastBigChange;

    std::mutex mMutexCamera;
};

#endif // MAPDRAWER_H
