#include "ransacpnp.h"
#include "Core/frame.h"
#include "Core/rgbdcamera.h"
#include "System/converter.h"
#include <opencv2/calib3d.hpp>

using namespace std;

PnPRansac::PnPRansac(const Frame::Ptr F1, Frame::Ptr F2, const vector<cv::DMatch>& matches)
    : Solver(F1, F2, matches)
{
}

bool PnPRansac::compute(vector<cv::DMatch>& inliers)
{
    if (mMatches.size() < 10)
        return false;

    vector<cv::Point2f> v2D;
    v2D.reserve(mMatches.size());

    vector<cv::Point3f> v3D;
    v3D.reserve(mMatches.size());

    for (size_t i = 0; i < mMatches.size(); i++) {
        const cv::DMatch& m = mMatches[i];

        v2D.push_back(mF2->mvKeysUn[m.trainIdx].pt);
        cv::Mat pw = mF2->unprojectWorld(m.trainIdx);
        v3D.push_back(cv::Point3f(pw.at<float>(0), pw.at<float>(1), pw.at<float>(2)));

        mF2->setOutlier(m.trainIdx);
    }

    if (v2D.size() < 10)
        return false;

    cv::Mat r, t, inliersindex;
    bool status = cv::solvePnPRansac(v3D, v2D, mF2->mpCamera->k(), cv::Mat(), r, t, false, 500, 3.0f, 0.85, inliersindex);

    if (status) {
        cv::Mat Tcw = Converter::toHomogeneous(r, t);
        mF2->setPose(Tcw);

        inliers.clear();
        inliers.reserve(inliersindex.rows);
        for (int i = 0; i < inliersindex.rows; ++i) {
            const cv::DMatch& m = mMatches[inliersindex.at<int>(i)];

            inliers.push_back(m);
            mF2->setInlier(m.trainIdx);
        }
    }

    return status;
}
