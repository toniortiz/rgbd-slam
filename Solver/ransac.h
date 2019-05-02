#ifndef RANSAC_H
#define RANSAC_H

#include "solver.h"
#include <Eigen/Dense>
#include <memory>
#include <opencv2/core.hpp>
#include <set>
#include <vector>

class RGBDcamera;

class Ransac : public Solver {
public:
    enum ERROR_VERSION { EUCLIDEAN_ERROR,
        REPROJECTION_ERROR,
        EUCLIDEAN_AND_REPROJECTION_ERROR,
        MAHALANOBIS_ERROR,
        ADAPTIVE_ERROR };

    struct parameters {
        int verbose;
        int errorVersion, errorVersionVO, errorVersionMap;
        double inlierThresholdEuclidean, inlierThresholdReprojection, inlierThresholdMahalanobis;
        double minimalInlierRatioThreshold;
        int minimalNumberOfMatches;
        int usedPairs;
        int iterationCount;
    };

    Ransac(const std::shared_ptr<Frame> F1, std::shared_ptr<Frame> F2, const std::vector<cv::DMatch>& matches, Ransac::parameters params);

    virtual ~Ransac(){}

    bool compute(std::vector<cv::DMatch>& inliers) override;

protected:
    // Robustly estimate transformation from given two sets of 3D features and
    // potentially correct matches between those sets
    Eigen::Matrix4f estimateTransformation(std::vector<cv::Point3f> prevFeatures,
        std::vector<cv::Point3f> features, std::vector<cv::DMatch> matches,
        std::vector<cv::DMatch>& inlierMatches);

    // Compute inlier ratio w.r.t. points in the second image
    static double pointInlierRatio(std::vector<cv::DMatch>& inlierMatches,
        std::vector<cv::DMatch>& allMatches)
    {
        std::set<int> inlier, all;
        for (auto& m : allMatches)
            all.insert(m.trainIdx);

        for (auto& in : inlierMatches)
            inlier.insert(in.trainIdx);

        return double(inlier.size()) / double(all.size());
    }

    std::shared_ptr<RGBDcamera> mpCamera;
    parameters mRANSACParams;

    enum TransfEstimationType {
        UMEYAMA,
        G2O
    };

    std::vector<cv::DMatch> getRandomMatches(const std::vector<cv::DMatch> matches);

    bool computeTransformationModel(const std::vector<cv::Point3f> prevFeatures,
        const std::vector<cv::Point3f> features, const std::vector<cv::DMatch> matches,
        Eigen::Matrix4f& transformationModel, TransfEstimationType usedType = UMEYAMA);

    float computeMatchInlierRatioEuclidean(const std::vector<cv::Point3f> prevFeatures,
        const std::vector<cv::Point3f> features, const std::vector<cv::DMatch> matches,
        const Eigen::Matrix4f transformationModel, std::vector<cv::DMatch>& modelConsistentMatches);

    float computeInlierRatioMahalanobis(const std::vector<cv::Point3f> prevFeatures,
        const std::vector<cv::Point3f> features, const std::vector<cv::DMatch> matches,
        const Eigen::Matrix4f transformationModel, std::vector<cv::DMatch>& modelConsistentMatches);

    float computeInlierRatioReprojection(const std::vector<cv::Point3f> prevFeatures,
        const std::vector<cv::Point3f> features, const std::vector<cv::DMatch> matches,
        const Eigen::Matrix4f transformationModel, std::vector<cv::DMatch>& modelConsistentMatches);

    float computeInlierRatioEuclideanAndReprojection(const std::vector<cv::Point3f> prevFeatures,
        const std::vector<cv::Point3f> features, const std::vector<cv::DMatch> matches,
        const Eigen::Matrix4f transformationModel, std::vector<cv::DMatch>& modelConsistentMatches);

    inline void saveBetterModel(const double inlierRatio, const Eigen::Matrix4f transformationModel,
        std::vector<cv::DMatch> modelConsistentMatches, double& bestInlierRatio, Eigen::Matrix4f& bestTransformationModel,
        std::vector<cv::DMatch>& bestInlierMatches);

    inline int computeRANSACIteration(double inlierRatio, double successProbability = 0.98, int numberOfPairs = 3);
};

#endif // RANSAC_H
