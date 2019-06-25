#ifndef RICP_H
#define RICP_H

#include <Eigen/Core>
#include <memory>
#include <opencv2/opencv.hpp>
#include <vector>

class Frame;

class RansacSE3 {
public:
    RansacSE3();

    RansacSE3(int iters, uint minInlierTh, float maxMahalanobisDist, uint sampleSize);

    ~RansacSE3() {}

    // Compute the geometric relations T12 which allow us to estimate the motion
    // between the state pF1 and pF2 (pF1 -->[T12]--> pF2)
    bool compute(std::shared_ptr<Frame> pF1, std::shared_ptr<Frame> pF2,
        const std::vector<cv::DMatch>& m12, const bool& updateF2 = true);

    void setIterations(int iters);
    void setMaxMahalanobisDistance(float dist);
    void setSampleSize(uint sampleSize);
    void setInlierThreshold(uint th);

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
    std::vector<cv::DMatch> sampleMatches(const std::vector<cv::DMatch>& vMatches);

    Eigen::Matrix4f getTransformFromMatches(const std::vector<cv::DMatch>& vMatches, bool& valid);

    double computeInliersAndError(const std::vector<cv::DMatch>& m12, const Eigen::Matrix4f& transformation4f, std::vector<cv::DMatch>& vInlierMatches);

    double errorFunction2(const Eigen::Vector4f& x1, const Eigen::Vector4f& x2, const Eigen::Matrix4d& transformation);

    double depthCovariance(double depth);

    double depthStdDev(double depth);

    // Ransac parameters
    int mIterations;
    uint mMinInlierTh;
    float mMaxMahalanobisDistance;
    uint mSampleSize;

    std::shared_ptr<Frame> mpSourceFrame;
    std::shared_ptr<Frame> mpTargetFrame;

public:
    float rmse;
    std::vector<cv::DMatch> mvInliers;
    Eigen::Matrix4f mT21; // transform points from 2 into 1 (2 -> 1)
};

#endif // RANSAC_H
