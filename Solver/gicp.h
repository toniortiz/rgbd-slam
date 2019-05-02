#ifndef GENERALIZEDICP_H
#define GENERALIZEDICP_H

#include "System/macros.h"
#include "solver.h"
#include <opencv2/opencv.hpp>
#include <pcl/registration/gicp.h>
#include <vector>

class Gicp : public Solver {
public:
    Gicp(const std::shared_ptr<Frame> F1, std::shared_ptr<Frame> F2, const std::vector<cv::DMatch>& matches);

    virtual ~Gicp() {}

    // Compute the geometric relations T12 which allow us to estimate the motion
    // between the state pF1 and pF2 (pF1 -->[T12]--> pF2)
    //    bool Compute(const Frame* pF1, const Frame* pF2, const std::vector<cv::DMatch>& vMatches12, Eigen::Matrix4f& guess);
    //    bool ComputeSubset(const Frame* pF1, const Frame* pF2, const std::vector<cv::DMatch>& vMatches12);

    bool compute(std::vector<cv::DMatch>& inliers) override;

    bool Align(const Eigen::Matrix4f& guess, const bool isSourceTransformed = false);

    void SetMaximumIterations(int iters);
    void SetMaxCorrespondenceDistance(double dist);
    void SetEuclideanFitnessEpsilon(double epsilon);
    void SetMaximumOptimizerIterations(int iters);
    void SetRANSACIterations(int iters);
    void SetRANSACOutlierRejectionThreshold(double thresh);
    void SetTransformationEpsilon(double epsilon);
    void SetUseReciprocalCorrespondences(bool flag);

private:
    void CreateCloudsFromMatches(const Frame* pF1, const Frame* pF2, const std::vector<cv::DMatch>& vMatches12, const Eigen::Matrix4f& T12, const bool& calcDist = false);

    std::vector<cv::DMatch> GetSubset(const std::vector<cv::DMatch>& vMatches12);

    pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> mGicp;

public:
    pcl::PointCloud<pcl::PointXYZ>::Ptr mpSourceCloud;
    pcl::PointCloud<pcl::PointXYZ>::Ptr mpTargetCloud;
};

#endif // GENERALIZEDICP_H
