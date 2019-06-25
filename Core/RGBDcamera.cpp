#include "RGBDcamera.h"
#include "Features/Extractor.h"
#include "Frame.h"
#include "IntrinsicMatrix.h"
#include "System/Random.h"
#include <Eigen/Dense>
#include <opencv2/imgcodecs.hpp>

using namespace std;

RGBDcamera::RGBDcamera(IntrinsicMatrix* pIntrinsics, const float bf, const float thDepth, const float depthMapFactor,
    const float fps, size_t imageWidth, size_t imageHeight)
    : mfps(fps)
    , mpIntrinsic(pIntrinsics)
    , mbf(bf)
    , mnImageWidth(imageWidth)
    , mnImageHeight(imageHeight)
{
    mThDepth = mbf * thDepth / mpIntrinsic->mFx;
    mb = mbf / mpIntrinsic->mFx;
    mDepthMapFactor = 1.0f / depthMapFactor;
}

RGBDcamera::~RGBDcamera()
{
    delete mpIntrinsic;
}

cv::Mat RGBDcamera::k()
{
    unique_lock<mutex> lock(mMutexIntrinsic);
    return mpIntrinsic->mK.clone();
}

cv::Mat RGBDcamera::distCoef()
{
    unique_lock<mutex> lock(mMutexIntrinsic);
    return mpIntrinsic->mDistCoef.clone();
}

float RGBDcamera::fx()
{
    unique_lock<mutex> lock(mMutexIntrinsic);
    return mpIntrinsic->mFx;
}

float RGBDcamera::fy()
{
    unique_lock<mutex> lock(mMutexIntrinsic);
    return mpIntrinsic->mFy;
}

float RGBDcamera::cx()
{
    unique_lock<mutex> lock(mMutexIntrinsic);
    return mpIntrinsic->mCx;
}

float RGBDcamera::cy()
{
    unique_lock<mutex> lock(mMutexIntrinsic);
    return mpIntrinsic->mCy;
}

float RGBDcamera::invfx()
{
    unique_lock<mutex> lock(mMutexIntrinsic);
    return mpIntrinsic->mInvfx;
}

float RGBDcamera::invfy()
{
    unique_lock<mutex> lock(mMutexIntrinsic);
    return mpIntrinsic->mInvfy;
}

float RGBDcamera::depthThreshold()
{
    unique_lock<mutex> lock(mMutexCamera);
    return mThDepth;
}

float RGBDcamera::baseLineFx()
{
    unique_lock<mutex> lock(mMutexCamera);
    return mbf;
}

Frame::Ptr RGBDcamera::createFrame(const string& baseDir, const string& colorfile, const string& depthfile,
    double timestamp, Extractor::Ptr pExtractor)
{
    cv::Mat imColor = cv::imread(baseDir + colorfile, cv::IMREAD_COLOR);
    cv::Mat imD = cv::imread(baseDir + depthfile, cv::IMREAD_UNCHANGED);

    Frame::Ptr pFrame(new Frame(imColor, imD, timestamp, pExtractor, this));
    return pFrame;
}

void RGBDcamera::backproject(const cv::KeyPoint& keypoint, Bearing& bearing)
{
    float fx, fy, cx, cy;
    {
        unique_lock<mutex> lock(mMutexIntrinsic);
        fx = mpIntrinsic->mFx;
        fy = mpIntrinsic->mFy;
        cx = mpIntrinsic->mCx;
        cy = mpIntrinsic->mCy;
    }

    const cv::Point2f& p2d = keypoint.pt;
    bearing[0] = static_cast<double>((p2d.x - cx) / fx);
    bearing[1] = static_cast<double>((p2d.y - cy) / fy);
    bearing[2] = 1.0;

    bearing = bearing.normalized();
}

void RGBDcamera::backprojectVectorized(const vector<cv::KeyPoint>& keypoints, BearingsVector& bearings)
{
    bearings = BearingsVector(keypoints.size(), Bearing::Zero());

    for (size_t i = 0; i < keypoints.size(); ++i)
        backproject(keypoints[i], bearings[i]);
}

void RGBDcamera::unproject(const cv::KeyPoint& kp, const float& depth, Point3d& xyz)
{
    unproject(kp.pt.x, kp.pt.y, depth, xyz);
}

void RGBDcamera::unproject(const float& x, const float& y, const float& depth, Point3d& xyz)
{
    float invfx, invfy, cx, cy;
    {
        unique_lock<mutex> lock(mMutexIntrinsic);
        invfx = mpIntrinsic->mInvfx;
        invfy = mpIntrinsic->mInvfy;
        cx = mpIntrinsic->mCx;
        cy = mpIntrinsic->mCy;
    }

    xyz[0] = (x - cx) * depth * invfx;
    xyz[1] = (y - cy) * depth * invfy;
    xyz[2] = depth;
}

void RGBDcamera::unproject(const float& x, const float& y, const float& depth, float& ux, float& uy, float& uz)
{
    float invfx, invfy, cx, cy;
    {
        unique_lock<mutex> lock(mMutexIntrinsic);
        invfx = mpIntrinsic->mInvfx;
        invfy = mpIntrinsic->mInvfy;
        cx = mpIntrinsic->mCx;
        cy = mpIntrinsic->mCy;
    }

    ux = (x - cx) * depth * invfx;
    uy = (y - cy) * depth * invfy;
    uz = depth;
}

cv::KeyPoint RGBDcamera::createRandomKeypoint()
{
    Eigen::Vector2d out;
    out.setRandom();

    double border = min(mnImageWidth, mnImageHeight) * 0.1;

    out(0) = border + abs(out(0)) * (mnImageWidth - border * 2.0);
    out(1) = border + abs(out(1)) * (mnImageHeight - border * 2.0);

    cv::KeyPoint keypoint;
    keypoint.pt = cv::Point2f(out(0), out(1));
    keypoint.octave = Random::randomInt(0, 7);

    return keypoint;
}

RGBDcamera::Bearing RGBDcamera::createRandomVisiblePoint(double depth)
{
    if (depth > 0.0) {
        Bearing bearing;

        cv::KeyPoint y = createRandomKeypoint();
        backproject(y, bearing);
        bearing /= bearing.norm();

        return bearing * depth;
    } else
        return Bearing::Zero();
}

cv::Point2f RGBDcamera::project3Dto2D(Eigen::Vector3f feature3D)
{
    float fx, fy, cx, cy;
    {
        unique_lock<mutex> lock(mMutexIntrinsic);
        fx = mpIntrinsic->mFx;
        fy = mpIntrinsic->mFy;
        cx = mpIntrinsic->mCx;
        cy = mpIntrinsic->mCy;
    }

    // Project in image and check it is not outside
    const float invz = 1.0f / feature3D.z();
    const float u = fx * feature3D.x() * invz + cx;
    const float v = fy * feature3D.y() * invz + cy;
    return cv::Point2f(u, v);
}

void RGBDcamera::project3Dto2D(const float& x, const float& y, const float& z, float& ox, float& oy)
{
    float fx, fy, cx, cy;
    {
        unique_lock<mutex> lock(mMutexIntrinsic);
        fx = mpIntrinsic->mFx;
        fy = mpIntrinsic->mFy;
        cx = mpIntrinsic->mCx;
        cy = mpIntrinsic->mCy;
    }

    const float invz = 1.0f / z;
    ox = fx * x * invz + cx;
    oy = fy * y * invz + cy;
}

ostream& operator<<(ostream& out, RGBDcamera& camera)
{
    out << "Calibration: " << camera.k() << endl;
    out << "Distortion: " << camera.distCoef() << endl;
    return out;
}
