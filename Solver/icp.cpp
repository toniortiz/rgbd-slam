#include "icp.h"
#include "Core/frame.h"
#include "System/converter.h"
#include "System/macros.h"
#include <memory>
#include <opengv/point_cloud/PointCloudAdapter.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/point_cloud/PointCloudSacProblem.hpp>

using namespace std;

typedef opengv::points_t Points3dVector;
typedef opengv::point_t Point3d;

typedef opengv::point_cloud::PointCloudAdapter Adapter;
typedef opengv::sac_problems::point_cloud::PointCloudSacProblem ICP;
typedef opengv::sac::Ransac<ICP> RANSAC;

Icp::Icp(const Frame::Ptr F1, Frame::Ptr F2, const vector<cv::DMatch>& matches)
    : Solver(F1, F2, matches)
{
}

bool Icp::compute(vector<cv::DMatch>& inliers)
{
    if (mMatches.empty())
        return false;

    Points3dVector points1, points2;

    points1.reserve(mMatches.size());
    points2.reserve(mMatches.size());

    for (const auto& m : mMatches) {
        cv::Point3f p1 = mF1->mvKeys3Dc[m.queryIdx];
        points1.push_back(Point3d(p1.x, p1.y, p1.z));

        cv::Point3f p2 = mF2->mvKeys3Dc[m.trainIdx];
        points2.push_back(Point3d(p2.x, p2.y, p2.z));

        mF2->setOutlier(m.trainIdx);
    }

    Adapter adapter(points2, points1);
    shared_ptr<ICP> icpproblem_ptr(new ICP(adapter));

    RANSAC ransac;
    ransac.sac_model_ = icpproblem_ptr;
    ransac.threshold_ = 0.07;
    ransac.max_iterations_ = 500;
    bool status = ransac.computeModel();

    if (status) {
        inliers.reserve(ransac.inliers_.size());
        for (size_t i = 0; i < ransac.inliers_.size(); ++i) {
            const auto& m = mMatches[ransac.inliers_.at(i)];
            inliers.push_back(m);
            mF2->setInlier(m.trainIdx);
        }

        cv::Mat R = Converter::toMat<float, 3, 3>(ransac.model_coefficients_.leftCols(3).cast<float>());
        cv::Mat t = Converter::toMat<float, 3, 1>(ransac.model_coefficients_.rightCols(1).cast<float>());
        mF2->setPose(Converter::toCvSE3(R, t) * mF1->getPose());

        return true;
    } else {
        return false;
    }
}
