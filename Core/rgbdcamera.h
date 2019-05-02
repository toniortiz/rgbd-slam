#ifndef RGBDCAMERA_H
#define RGBDCAMERA_H

#include "System/macros.h"
#include <memory>
#include <mutex>
#include <opencv2/core.hpp>
#include <opengv/types.hpp>

class IntrinsicMatrix;
class Frame;
class Extractor;

class RGBDcamera {
public:
    SMART_POINTER_TYPEDEFS(RGBDcamera);

    typedef opengv::bearingVector_t Bearing;
    typedef opengv::bearingVectors_t BearingsVector;
    typedef opengv::point_t Point3d;
    typedef opengv::points_t Points3dVector;

public:
    RGBDcamera(IntrinsicMatrix* pIntrinsics, const float bf, const float thDepth, const float depthMapFactor,
        const float fps, size_t imageWidth, size_t imageHeight);
    ~RGBDcamera();

    cv::Mat k();
    cv::Mat distCoef();
    float fx();
    float fy();
    float cx();
    float cy();
    float invfx();
    float invfy();
    float depthThreshold();
    float baseLineFx();

    std::shared_ptr<Frame> createFrame(const std::string& baseDir, const std::string& colorfile,
        const std::string& depthfile, double timestamp, std::shared_ptr<Extractor> pExtractor);

    // Compute the 3d bearing vector in euclidean coordinates given a keypoint in image coordinates
    void backproject(const cv::KeyPoint& keypoint, Bearing& bearing);
    void backprojectVectorized(const std::vector<cv::KeyPoint>& keypoints, BearingsVector& bearings);

    void unproject(const cv::KeyPoint& kp, const float& depth, Point3d& xyz);
    void unproject(const float& x, const float& y, const float& depth, Point3d& xyz);
    void unproject(const float& x, const float& y, const float& depth, float& ux, float& uy, float& uz);

    cv::KeyPoint createRandomKeypoint();
    Bearing createRandomVisiblePoint(double depth);

    cv::Point2f project3Dto2D(Eigen::Vector3f feature3D);

    friend std::ostream& operator<<(std::ostream& out, RGBDcamera& camera);

    // The following variables are accesed from only 1 thread or never change (no mutex needed).
public:
    // Stereo baseline in meters.
    float mb;

    float mfps;

    // For some datasets (e.g. TUM) the depthmap values are scaled.
    float mDepthMapFactor;

    // The following variables need to be accessed trough a mutex to be thread safe.
protected:
    IntrinsicMatrix* mpIntrinsic;

    // Stereo baseline multiplied by fx.
    float mbf;

    // Threshold close/far points. Close points are inserted from 1 view.
    // Far points are inserted as in the monocular case from 2 views.
    float mThDepth;

    size_t mnImageWidth;
    size_t mnImageHeight;

    std::mutex mMutexIntrinsic;
    std::mutex mMutexCamera;
};

#endif // RGBDCAMERA_H
