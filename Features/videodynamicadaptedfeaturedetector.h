#ifndef VIDEODYNAMICADAPTEDFEATUREDETECTOR_H
#define VIDEODYNAMICADAPTEDFEATUREDETECTOR_H

#include "detectoradjuster.h"
#include "statefulfeaturedetector.h"

class VideoDynamicAdaptedFeatureDetector : public StatefulFeatureDetector {
public:
    VideoDynamicAdaptedFeatureDetector(cv::Ptr<DetectorAdjuster> pAdjuster, int minFeatures = 400, int maxFeatures = 500, int maxIters = 5);

    virtual cv::Ptr<StatefulFeatureDetector> clone() const;

    CV_WRAP virtual void detect(cv::InputArray image, CV_OUT std::vector<cv::KeyPoint>& keypoints, cv::InputArray mask = cv::noArray());

private:
    VideoDynamicAdaptedFeatureDetector& operator=(const VideoDynamicAdaptedFeatureDetector&);
    VideoDynamicAdaptedFeatureDetector(const VideoDynamicAdaptedFeatureDetector&);

    int mEscapeIters;
    int mMinFeatures, mMaxFeatures;
    mutable cv::Ptr<DetectorAdjuster> mpAdjuster;
};
#endif // VIDEODYNAMICADAPTEDFEATUREDETECTOR_H
