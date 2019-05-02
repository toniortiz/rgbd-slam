#ifndef STATEFULFEATUREDETECTOR_H
#define STATEFULFEATUREDETECTOR_H

#include <opencv2/features2d.hpp>

class StatefulFeatureDetector : public cv::Feature2D {
public:
    virtual cv::Ptr<StatefulFeatureDetector> clone() const = 0;

    CV_WRAP virtual void detect(cv::InputArray image, CV_OUT std::vector<cv::KeyPoint>& keypoints, cv::InputArray mask = cv::noArray());
};

#endif // STATEFULFEATUREDETECTOR_H
