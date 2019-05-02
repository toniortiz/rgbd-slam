#ifndef LANDMARK_H
#define LANDMARK_H

#include "System/macros.h"
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <opencv2/core/core.hpp>

class Frame;
class Map;

// Represent a point in the world
class Landmark {
public:
    SMART_POINTER_TYPEDEFS(Landmark);

    typedef int KeyFrameID;

public:
    Landmark(const cv::Mat& Pos, std::shared_ptr<Map> pMap, std::shared_ptr<Frame> frame, const size_t& idxF);

    void setWorldPos(const cv::Mat& Pos);
    cv::Mat getWorldPos();

    cv::Mat getNormal();
    KeyFrameID getReferenceKeyFrameId();

    std::map<KeyFrameID, size_t> getObservations();
    std::list<std::pair<KeyFrameID, size_t>> getObservationsList();
    int observations();

    void addObservation(KeyFrameID id, size_t obsId);
    void eraseObservation(KeyFrameID id);

    int getIndexInKeyFrame(KeyFrameID id);
    bool isInKeyFrame(KeyFrameID id);

    cv::Vec3b getColor();
    void setColor(const cv::Vec3b& color);

    void replace(Landmark::Ptr pMP);
    Landmark::Ptr getReplaced();

    cv::Mat getDescriptor();

    void updateNormalAndDepth();

    bool operator==(const Landmark& pLM) const;

public:
    int mnId;
    static int nNextId;
    int nObs;

    static std::mutex mGlobalMutex;

protected:
    // Position in absolute coordinates
    cv::Mat mWorldPos;

    cv::Vec3b mColor;

    // Keyframes observing the point and associated index in keyframe
    std::map<KeyFrameID, size_t> mObservations;

    // Mean viewing direction
    cv::Mat mNormalVector;

    // Best descriptor to fast matching
    cv::Mat mDescriptor;

    // Reference KeyFrame
    KeyFrameID mnRefKFid;

    std::shared_ptr<Map> mpMap;

    std::mutex mMutexPos;
    std::mutex mMutexFeatures;
};

#endif // MAPPOINT_H
