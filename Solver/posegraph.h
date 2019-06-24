#ifndef GX_POSEGRAPH_H
#define GX_POSEGRAPH_H

#include "System/macros.h"
#include <condition_variable>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/types/slam3d/edge_se3.h>
#include <mutex>
#include <opencv2/features2d.hpp>
#include <thread>

class Tracking;
class LoopDetector;
class Map;
class Frame;

class PoseGraph {
public:
    SMART_POINTER_TYPEDEFS(PoseGraph);

    typedef std::map<int, int> EdgeID;
    typedef std::map<EdgeID, g2o::EdgeSE3*> Edges;

public:
    PoseGraph(Tracking* pTracker, std::shared_ptr<LoopDetector> pLooper, std::shared_ptr<Map> pMap);

    void run();

    void insertKeyFrame(std::shared_ptr<Frame> pKF);

    size_t totalEdges();
    std::vector<g2o::EdgeSE3*> getEdges();

    bool removeVertex(int id);
    bool removeEdge(int id);

    void optimize(const int& iterations = 10);

    void nearestNodes(std::shared_ptr<Frame> pKF, std::vector<std::shared_ptr<Frame>>& candidates);

    void requestFinish();

    bool isFinished();

    void shutdown();

protected:
    void updateGraph();
    void createNode();
    void createEdgeWithReference();
    double createEdge(std::shared_ptr<Frame> pKF, const Eigen::Isometry3d& T);

    bool detectLoop();

    void createLocalEdges();

    bool existEdge(const int v1, const int v2);

    void matchLandmarks(std::shared_ptr<Frame> pKF, std::vector<cv::DMatch>& inliers);

    bool checkNewKeyFrames();

    bool checkFinish();
    void setFinish();
    bool mbFinishRequested;
    bool mbFinished;
    std::mutex mMutexFinish;

    std::mutex mMutexOptimizer;
    g2o::SparseOptimizer mOptimizer;
    Edges mEdges;

    Tracking* mpTracker;
    std::shared_ptr<LoopDetector> mpLoopCloser;
    std::shared_ptr<Map> mpMap;

    std::list<std::shared_ptr<Frame>> mlpKeyFrameQueue;
    std::mutex mMutexQueue;

    std::shared_ptr<Frame> mpCurrentKF;
    std::shared_ptr<Frame> mpReferenceKF;

    int mnKFsFromLastLoop;
    std::mutex mMutexUpdate;

    std::thread mRunThread;
};

#endif // POSEGRAPH_H
