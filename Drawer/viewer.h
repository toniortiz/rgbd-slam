#ifndef VIEWER_H
#define VIEWER_H

#include <memory>
#include <mutex>
#include "System/macros.h"

class MapDrawer;
class Map;

class Viewer {
public:
    SMART_POINTER_TYPEDEFS(Viewer);

public:
    Viewer(std::shared_ptr<MapDrawer> pMapDrawer, std::shared_ptr<Map> pMap);

    void run();

    void requestFinish();

    void requestStop();

    bool isFinished();

    bool isStopped();

    void release();

    void setMeanTrackingTime(const double& time);
    void loopCandidates(const size_t& n);

    void shutdown();

private:
    bool stop();

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

    bool mbStopped;
    bool mbStopRequested;
    std::mutex mMutexStop;

    std::mutex mMutexUpdate;
    double mMeanTrackTime;
    int mnLoopCandidates;
};

#endif // VIEWER_H
