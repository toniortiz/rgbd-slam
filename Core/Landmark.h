#ifndef LANDMARK_H
#define LANDMARK_H

#include "System/Macros.h"
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <opencv2/core/core.hpp>

class Frame;

// Represent a point in the world
class Landmark {
public:
    SMART_POINTER_TYPEDEFS(Landmark);

    typedef std::shared_ptr<Frame> KeyFramePtr;

public:
    Landmark();
    Landmark(const cv::Mat& Pos, KeyFramePtr frame, const size_t& idxF);

    static Landmark::Ptr create();

    void setWorldPos(const cv::Mat& Pos);
    cv::Mat getWorldPos();

    std::map<KeyFramePtr, size_t> getObservations();
    std::list<std::pair<KeyFramePtr, size_t>> getObservationsList();

    void addObservation(KeyFramePtr pKF, size_t obsId);
    void eraseObservation(KeyFramePtr pKF);

    int getIndexInKeyFrame(KeyFramePtr pKF);
    bool isInKeyFrame(KeyFramePtr pKF);

    cv::Vec3b getColor();
    void setColor(const cv::Vec3b& color);

    void replace(Landmark::Ptr pMP);
    Landmark::Ptr getReplaced();

    void setDescriptor(cv::Mat& desc);
    cv::Mat getDescriptor();

    int id();
    int obs();

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
    std::map<KeyFramePtr, size_t> mObservations;

    // Best descriptor to fast matching
    cv::Mat mDescriptor;

    std::mutex mMutexPos;
    std::mutex mMutexFeatures;
};

#endif // MAPPOINT_H
