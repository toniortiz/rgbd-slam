#ifndef VIEWER_H
#define VIEWER_H

#include "System/macros.h"
#include <memory>
#include <mutex>
#include <thread>

class MapDrawer;
class Map;

class Viewer {
public:
    SMART_POINTER_TYPEDEFS(Viewer);

public:
    Viewer(std::shared_ptr<MapDrawer> pMapDrawer, std::shared_ptr<Map> pMap);

    void run();

    void requestFinish();

    bool isFinished();

    void setMeanTrackingTime(const double& time);
    void loopCandidates(const size_t& n);

    void shutdown();

private:
    std::shared_ptr<MapDrawer> mpMapDrawer;
    std::shared_ptr<Map> mpMap;

    // 1/fps in ms
    double mT;
    float mImageWidth, mImageHeight;

    float mViewpointX, mViewpointY, mViewpointZ, mViewpointF;

    bool checkFinish();
    void setFinish();
    bool mbFinishRequested;
    bool mbFinished;
    std::mutex mMutexFinish;

    std::mutex mMutexUpdate;
    double mMeanTrackTime;
    int mnLoopCandidates;

    std::thread mRunThread;
};

#endif // VIEWER_H
