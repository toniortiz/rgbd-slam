#ifndef VIDEOGRIDADAPTEDFEATUREDETECTOR_H
#define VIDEOGRIDADAPTEDFEATUREDETECTOR_H

#include "statefulfeaturedetector.h"

class VideoGridAdaptedFeatureDetector : public StatefulFeatureDetector {
public:
    /**
     * \param detector            Detector that will be adapted.
     * \param maxTotalKeypoints   Maximum count of keypoints detected on the image. Only the strongest keypoints will be keeped.
     * \param gridRows            Grid rows count.
     * \param gridCols            Grid column count.
     * \param edgeThreshold       How much overlap is needed, to not lose keypoints at the inner borders of the grid (should be the same value, e.g., as edgeThreshold for ORB)
     */
    VideoGridAdaptedFeatureDetector(const cv::Ptr<StatefulFeatureDetector>& pDetector,
        int mMaxTotalKeypoints = 1000, int mGridRows = 4, int mGridCols = 4, int mEdgeThreshold = 31);

    virtual cv::Ptr<StatefulFeatureDetector> clone() const;

    CV_WRAP virtual void detect(cv::InputArray image, CV_OUT std::vector<cv::KeyPoint>& keypoints, cv::InputArray mask = cv::noArray());

protected:
    VideoGridAdaptedFeatureDetector& operator=(const VideoGridAdaptedFeatureDetector&);
    VideoGridAdaptedFeatureDetector(const VideoGridAdaptedFeatureDetector&);

    std::vector<cv::Ptr<StatefulFeatureDetector>> vpDetectors;
    int mMaxTotalKeypoints;
    int mGridRows;
    int mGridCols;
    int mEdgeThreshold;
};

#endif // VIDEOGRIDADAPTEDFEATUREDETECTOR_H
