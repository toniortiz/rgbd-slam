#ifndef MATCHER_H
#define MATCHER_H

#include <memory>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <tuple>
#include <vector>

class Frame;

class Matcher {
public:
    Matcher(float nnratio = 0.6);

    // Draw
    static void drawMatches(std::shared_ptr<Frame> ref, std::shared_ptr<Frame> cur, const std::vector<cv::DMatch>& m12, const int delay = 1);

    int match(std::shared_ptr<Frame> ref, std::shared_ptr<Frame> cur, std::vector<cv::DMatch>& vMatches12,
        const bool discardOutliers = true);

    int projectionMatch(std::shared_ptr<Frame> F1, std::shared_ptr<Frame> F2, std::vector<cv::DMatch>& vMatches12, const float th = 5.0f);

    // Computes the Hamming distance between two descriptors
    static double descriptorDistance(const cv::Mat& a, const cv::Mat& b);

    static const int TH_LOW;
    static const int TH_HIGH;
    static const int HISTO_LENGTH;

protected:
    float mfNNratio;

    cv::Ptr<cv::DescriptorMatcher> mpMatcher;
};

#endif // ORBMATCHER_H
