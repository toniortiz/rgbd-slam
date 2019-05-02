#include "gicp.h"
#include "Core/frame.h"

using namespace std;

//GeneralizedICP::GeneralizedICP()
//    : GeneralizedICP(15, 0.05)
//{
//}

//GeneralizedICP::GeneralizedICP(int iters, double maxCorrespondenceDist)
//{
//    mGicp.setMaximumIterations(iters);
//    mGicp.setMaxCorrespondenceDistance(maxCorrespondenceDist);
//    mGicp.setEuclideanFitnessEpsilon(1);
//    mGicp.setTransformationEpsilon(1e-9);

//    mpSourceCloud = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
//    mpTargetCloud = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();

//    mT12 = Eigen::Matrix4f::Identity();
//}

//bool GeneralizedICP::Compute(const Frame* pF1, const Frame* pF2, const vector<cv::DMatch>& vMatches12, Eigen::Matrix4f& guess)
//{
//    CreateCloudsFromMatches(pF1, pF2, vMatches12, guess, false);
//    return Align(guess);
//}

//bool GeneralizedICP::Compute(const pcl::PointCloud<pcl::PointXYZ>::Ptr pSourceCloud, const pcl::PointCloud<pcl::PointXYZ>::Ptr pTargetCloud,
//    const Eigen::Matrix4f& guess, const bool isSourceTransformed)
//{
//    if (pSourceCloud->points.size() < 20 || pTargetCloud->points.size() < 20)
//        return false;

//    mpSourceCloud = pSourceCloud;
//    mpTargetCloud = pTargetCloud;
//    return Align(guess, isSourceTransformed);
//}

//bool GeneralizedICP::ComputeSubset(const Frame* pF1, const Frame* pF2, const vector<cv::DMatch>& vMatches12)
//{
//    vector<cv::DMatch> vMatchesSubset = GetSubset(vMatches12);
//    Eigen::Matrix4f guess = Eigen::Matrix4f::Identity();

//    CreateCloudsFromMatches(pF1, pF2, vMatchesSubset, guess, false);

//    return Align(guess);
//}

//bool GeneralizedICP::Align(const Eigen::Matrix4f& guess, const bool isSourceTransformed)
//{
//    mGicp.setInputSource(mpSourceCloud);
//    mGicp.setInputTarget(mpTargetCloud);

//    pcl::PointCloud<pcl::PointXYZ> regCloud;
//    if (isSourceTransformed)
//        mGicp.align(regCloud);
//    else
//        mGicp.align(regCloud, guess);

//    if (mGicp.hasConverged()) {
//        if (isSourceTransformed)
//            mT12 = mGicp.getFinalTransformation().matrix() * guess;
//        else
//            mT12 = mGicp.getFinalTransformation().matrix();

//        mScore = mGicp.getFitnessScore();
//        return true;
//    } else {
//        mT12 = Eigen::Matrix4f::Identity();
//        mScore = 1e6;
//        return false;
//    }
//}

//void GeneralizedICP::CreateCloudsFromMatches(const Frame* pF1, const Frame* pF2, const vector<cv::DMatch>& vMatches12, const Eigen::Matrix4f& T12, const bool& calcDist)
//{
//    mpSourceCloud->points.clear();
//    mpTargetCloud->points.clear();

//    double dist = 0.0;
//    cv::Mat R(3, 3, CV_32F);
//    cv::Mat t(3, 1, CV_32F);

//    if (calcDist) {
//        for (int i = 0; i < 3; ++i)
//            for (int j = 0; j < 3; ++j)
//                R.at<float>(i, j) = T12(i, j);

//        for (int i = 0; i < 3; ++i)
//            t.at<float>(i, 0) = T12(i, 3);
//    }

//    for (const auto& m : vMatches12) {
//        const cv::Point3f& source = pF1->mvKeys3Dc[m.queryIdx];
//        const cv::Point3f& target = pF2->mvKeys3Dc[m.trainIdx];

//        if (isnan(source.z) || isnan(target.z))
//            continue;
//        if (source.z <= 0 || target.z <= 0)
//            continue;

//        if (calcDist) {
//            cv::Mat src = (cv::Mat_<float>(3, 1) << source.x, source.y, source.z);
//            cv::Mat tgt = (cv::Mat_<float>(3, 1) << target.x, target.y, target.z);

//            cv::Mat trans = R * src + t;
//            dist += cv::norm(trans - tgt);
//        }

//        mpSourceCloud->points.push_back(pcl::PointXYZ(source.x, source.y, source.z));
//        mpTargetCloud->points.push_back(pcl::PointXYZ(target.x, target.y, target.z));
//    }

//    if (calcDist) {
//        double meanDist = dist / mpTargetCloud->size();
//        SetMaxCorrespondenceDistance(std::min(std::max(0.01, meanDist), 0.08));
//    }
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

//void GeneralizedICP::SetMaximumIterations(int iters) { mGicp.setMaximumIterations(iters); }

//void GeneralizedICP::SetMaxCorrespondenceDistance(double dist) { mGicp.setMaxCorrespondenceDistance(dist); }

//void GeneralizedICP::SetEuclideanFitnessEpsilon(double epsilon) { mGicp.setEuclideanFitnessEpsilon(epsilon); }

//void GeneralizedICP::SetMaximumOptimizerIterations(int iters) { mGicp.setMaximumOptimizerIterations(iters); }

//void GeneralizedICP::SetRANSACIterations(int iters) { mGicp.setRANSACIterations(iters); }

//void GeneralizedICP::SetRANSACOutlierRejectionThreshold(double thresh) { mGicp.setRANSACOutlierRejectionThreshold(thresh); }

//void GeneralizedICP::SetTransformationEpsilon(double epsilon) { mGicp.setTransformationEpsilon(epsilon); }

//void GeneralizedICP::SetUseReciprocalCorrespondences(bool flag) { mGicp.setUseReciprocalCorrespondences(flag); }
