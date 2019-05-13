#include "gicp.h"
#include "Core/frame.h"
#include "System/converter.h"
using namespace std;

Gicp::Gicp(const Frame::Ptr F1, Frame::Ptr F2, const vector<cv::DMatch>& matches, Eigen::Matrix4f& guess)
    : Solver(F1, F2, matches)
    , mGuess(guess)
{
    mGicp.setMaximumIterations(15);
    mGicp.setMaxCorrespondenceDistance(0.08);
    mGicp.setEuclideanFitnessEpsilon(1);
    mGicp.setTransformationEpsilon(1e-9);

    mpSrcCloud = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    mpTgtCloud = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
}

bool Gicp::compute(vector<cv::DMatch>& inliers)
{
    if (mMatches.size() < 20)
        return false;

    createCloudsFromMatches();
    Eigen::Matrix4f T = align();

    if (!T.isIdentity()) {
        mF2->setPose(Converter::toMat<float, 4, 4>(T) * mF1->getPose());
        return true;
    } else
        return false;
}

void Gicp::createCloudsFromMatches()
{
    mpSrcCloud->points.clear();
    mpTgtCloud->points.clear();

    mpSrcCloud->points.reserve(mMatches.size());
    mpTgtCloud->points.reserve(mMatches.size());

    for (const auto& m : mMatches) {
        const cv::Point3f& source = mF1->mvKeys3Dc[m.queryIdx];
        const cv::Point3f& target = mF2->mvKeys3Dc[m.trainIdx];

        mpSrcCloud->points.push_back(pcl::PointXYZ(source.x, source.y, source.z));
        mpTgtCloud->points.push_back(pcl::PointXYZ(target.x, target.y, target.z));
    }
}

Eigen::Matrix4f Gicp::align()
{
    mGicp.setInputSource(mpSrcCloud);
    mGicp.setInputTarget(mpTgtCloud);

    pcl::PointCloud<pcl::PointXYZ> regCloud;
    mGicp.align(regCloud, mGuess);

    if (mGicp.hasConverged())
        return mGicp.getFinalTransformation().matrix();
    else
        return Eigen::Matrix4f::Identity();
}

void Gicp::setMaximumIterations(int iters) { mGicp.setMaximumIterations(iters); }

void Gicp::setMaxCorrespondenceDistance(double dist) { mGicp.setMaxCorrespondenceDistance(dist); }

void Gicp::setEuclideanFitnessEpsilon(double epsilon) { mGicp.setEuclideanFitnessEpsilon(epsilon); }

void Gicp::setMaximumOptimizerIterations(int iters) { mGicp.setMaximumOptimizerIterations(iters); }

void Gicp::setRANSACIterations(int iters) { mGicp.setRANSACIterations(iters); }

void Gicp::setRANSACOutlierRejectionThreshold(double thresh) { mGicp.setRANSACOutlierRejectionThreshold(thresh); }

void Gicp::setTransformationEpsilon(double epsilon) { mGicp.setTransformationEpsilon(epsilon); }

void Gicp::setUseReciprocalCorrespondences(bool flag) { mGicp.setUseReciprocalCorrespondences(flag); }

//bool GeneralizedICP::ComputeSubset(const Frame* pF1, const Frame* pF2, const vector<cv::DMatch>& vMatches12)
//{
//    vector<cv::DMatch> vMatchesSubset = GetSubset(vMatches12);
//    Eigen::Matrix4f guess = Eigen::Matrix4f::Identity();

//    CreateCloudsFromMatches(pF1, pF2, vMatchesSubset, guess, false);

//    return Align(guess);
//}

//vector<cv::DMatch> GeneralizedICP::GetSubset(const vector<cv::DMatch>& vMatches12)
//{
//    set<vector<cv::DMatch>::size_type> sSampledIds;
//    int safetyNet = 0;

//    while (sSampledIds.size() < 0.75 * vMatches12.size()) {
//        int id1 = rand() % vMatches12.size();
//        int id2 = rand() % vMatches12.size();

//        if (id1 > id2)
//            id1 = id2;

//        sSampledIds.insert(id1);

//        if (++safetyNet > 10000)
//            break;
//    }

//    vector<cv::DMatch> vSampledMatches;
//    vSampledMatches.reserve(sSampledIds.size());
//    for (const auto& id : sSampledIds)
//        vSampledMatches.push_back(vMatches12[id]);

//    return vSampledMatches;
//}
