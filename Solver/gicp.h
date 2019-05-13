#ifndef GENERALIZEDICP_H
#define GENERALIZEDICP_H

#include "System/macros.h"
#include "solver.h"
#include <opencv2/opencv.hpp>
#include <pcl/registration/gicp.h>
#include <vector>

class Gicp : public Solver {
public:
    Gicp(const std::shared_ptr<Frame> F1, std::shared_ptr<Frame> F2, const std::vector<cv::DMatch>& matches, Eigen::Matrix4f& guess);

    virtual ~Gicp() {}

    // Compute the geometric relations T12 which allow us to estimate the motion
    // between the state pF1 and pF2 (pF1 -->[T12]--> pF2)
    //    bool Compute(const Frame* pF1, const Frame* pF2, const std::vector<cv::DMatch>& vMatches12, Eigen::Matrix4f& guess);
    //    bool ComputeSubset(const Frame* pF1, const Frame* pF2, const std::vector<cv::DMatch>& vMatches12);

    bool compute(std::vector<cv::DMatch>& inliers) override;

    Eigen::Matrix4f align();

    void setMaximumIterations(int iters);
    void setMaxCorrespondenceDistance(double dist);
    void setEuclideanFitnessEpsilon(double epsilon);
    void setMaximumOptimizerIterations(int iters);
    void setRANSACIterations(int iters);
    void setRANSACOutlierRejectionThreshold(double thresh);
    void setTransformationEpsilon(double epsilon);
    void setUseReciprocalCorrespondences(bool flag);

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
    void createCloudsFromMatches();

    std::vector<cv::DMatch> getSubset(const std::vector<cv::DMatch>& vMatches12);

    pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> mGicp;
    Eigen::Matrix4f mGuess;

public:
    pcl::PointCloud<pcl::PointXYZ>::Ptr mpSrcCloud;
    pcl::PointCloud<pcl::PointXYZ>::Ptr mpTgtCloud;
};

#endif // GENERALIZEDICP_H
