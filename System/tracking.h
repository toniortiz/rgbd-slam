#ifndef GX_TRACKING_H
#define GX_TRACKING_H

#include "System/macros.h"
#include <DBoW3/Vocabulary.h>
#include <list>
#include <mutex>
#include <opencv2/core.hpp>
#include <thread>

class Frame;
class LoopDetector;
class Viewer;
class MapDrawer;
class PoseGraph;
class Map;

class Tracking {
public:
    SMART_POINTER_TYPEDEFS(Tracking);

    enum TrackerState {
        NOT_INITIALIZED = 0,
        OK,
        LOST
    };

public:
    Tracking(std::shared_ptr<DBoW3::Vocabulary> pVoc, std::shared_ptr<Map> pMap);

    cv::Mat track(std::shared_ptr<Frame> newFrame);

    TrackerState getState() const;

    bool correct();

    void setTime(double meanTime);
    void setCurrentPose(const cv::Mat& pose);

    void loopCandidates(const size_t& n);

    void shutdown();

    void saveKeyFrameTrajectory(const std::string& filename);

    void saveCameraTrajectory(const std::string& filename);

    int getMeanInliers();
    int getCurrentInliers();

    std::list<cv::Mat> mRelativeFramePoses;
    std::list<std::shared_ptr<Frame>> mReferences;
    std::list<double> mFrameTimes;

protected:
    void initialize();

    void trackReference();

    void recover();

    bool needKeyFrame();
    void createKeyFrame();

    void updateLastFrame();
    void updateRelativePose();

    std::shared_ptr<Frame> mpCurFrame;
    std::pair<std::shared_ptr<Frame>, std::shared_ptr<Frame>> mpRefFrame;

    TrackerState mState;

    std::shared_ptr<DBoW3::Vocabulary> mpVoc;
    std::shared_ptr<Map> mpMap;
    std::shared_ptr<LoopDetector> mpLoopCloser;

    std::shared_ptr<PoseGraph> mpPoseGraph;
    std::shared_ptr<std::thread> mpGraphThread;

    std::shared_ptr<MapDrawer> mpMapDrawer;
    std::shared_ptr<Viewer> mpViewer;
    std::shared_ptr<std::thread> mpViewerThread;

    std::shared_ptr<Frame> mpLastKeyFrame;

    int mnAcumInliers;
    int mnInliers;
    int mnMeanInliers;
    std::mutex mMutexStatistics;

    std::mutex mMutexTrack;
};

#endif
