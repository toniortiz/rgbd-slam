#include "IntrinsicMatrix.h"

using namespace std;

IntrinsicMatrix::IntrinsicMatrix() {}

IntrinsicMatrix::IntrinsicMatrix(float fx, float fy, float cx, float cy)
{
    mK = cv::Mat::eye(3, 3, CV_32F);
    mK.at<float>(0, 0) = fx;
    mK.at<float>(1, 1) = fy;
    mK.at<float>(0, 2) = cx;
    mK.at<float>(1, 2) = cy;

    mFx = fx;
    mFy = fy;
    mCx = cx;
    mCy = cy;

    mInvfx = 1.0f / fx;
    mInvfy = 1.0f / fy;
}

IntrinsicMatrix::IntrinsicMatrix(const IntrinsicMatrix& obj)
    : mK(obj.mK)
    , mFx(obj.mFx)
    , mFy(obj.mFy)
    , mCx(obj.mCx)
    , mCy(obj.mCy)
    , mInvfx(obj.mInvfx)
    , mInvfy(obj.mInvfy)
{
    setDistortion(obj.mK1, obj.mK2, obj.mK3, obj.mP1, obj.mP2);
}

void IntrinsicMatrix::setDistortion(float k1, float k2, float k3, float p1, float p2)
{
    cv::Mat DistCoef(4, 1, CV_32F);
    DistCoef.at<float>(0) = k1;
    DistCoef.at<float>(1) = k2;
    DistCoef.at<float>(2) = p1;
    DistCoef.at<float>(3) = p2;
    if (k3 != 0.0f) {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    mK1 = k1;
    mK2 = k2;
    mK3 = k3;
    mP1 = p1;
    mP2 = p2;
}

void IntrinsicMatrix::invertOffset()
{
    mK.at<float>(0, 2) *= -1;
    mK.at<float>(1, 2) *= -1;
}

void IntrinsicMatrix::scale(float factor)
{
    mK *= factor;
}
