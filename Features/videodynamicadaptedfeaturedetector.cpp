#include "videodynamicadaptedfeaturedetector.h"

using namespace std;

VideoDynamicAdaptedFeatureDetector::VideoDynamicAdaptedFeatureDetector(cv::Ptr<DetectorAdjuster> pAdjuster,
    int minFeatures, int maxFeatures, int maxIters)
    : mEscapeIters(maxIters)
    , mMinFeatures(minFeatures)
    , mMaxFeatures(maxFeatures)
    , mpAdjuster(pAdjuster)
{
}

cv::Ptr<StatefulFeatureDetector> VideoDynamicAdaptedFeatureDetector::clone() const
{
    StatefulFeatureDetector* pNew = new VideoDynamicAdaptedFeatureDetector(mpAdjuster->clone(),
        mMinFeatures,
        mMaxFeatures,
        mEscapeIters);
    cv::Ptr<StatefulFeatureDetector> pCloned(pNew);
    return pCloned;
}

void VideoDynamicAdaptedFeatureDetector::detect(cv::InputArray image, vector<cv::KeyPoint>& keypoints, cv::InputArray mask)
{
    int iterCount = mEscapeIters;

    do {
        keypoints.clear();

        mpAdjuster->detect(image, keypoints, mask);
        int keypointsFound = static_cast<int>(keypoints.size());

        if (keypointsFound < mMinFeatures) {
            mpAdjuster->tooFew(mMinFeatures, keypointsFound);
        } else if (int(keypoints.size()) > mMaxFeatures) {
            mpAdjuster->tooMany(mMaxFeatures, (int)keypoints.size());
            break;
        } else
            break;

        iterCount--;
    } while (iterCount > 0 && mpAdjuster->good());
}
