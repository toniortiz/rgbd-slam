#include "DetectorAdjuster.h"
#include "SVOextractor.h"
#include <iostream>
#include <opencv2/xfeatures2d.hpp>

using namespace std;

DetectorAdjuster::DetectorAdjuster(const Extractor::eType& detector, double initialThresh, double minThresh,
    double maxThresh, double increaseFactor, double decreaseFactor)
    : mThresh(initialThresh)
    , mMinThresh(minThresh)
    , mMaxThresh(maxThresh)
    , mIncreaseFactor(increaseFactor)
    , mDecreaseFactor(decreaseFactor)
    , mDetectorAlgorithm(detector)
{
    if (!(detector == Extractor::eType::SURF || detector == Extractor::eType::SIFT || detector == Extractor::eType::FAST
            || detector == Extractor::eType::GFTT || detector == Extractor::eType::ORB)) {
        cerr << "Not supported detector" << endl;
    }
}

void DetectorAdjuster::detect(cv::InputArray image, vector<cv::KeyPoint>& keypoints, cv::InputArray mask)
{
    cv::Ptr<cv::Feature2D> detector;

    if (mDetectorAlgorithm == Extractor::eType::FAST)
        detector = cv::FastFeatureDetector::create(mThresh);
    else if (mDetectorAlgorithm == Extractor::eType::GFTT)
        detector = cv::GFTTDetector::create(1000, 0.01, mThresh, 3, false, 0.04);
    else if (mDetectorAlgorithm == Extractor::eType::ORB)
        detector = cv::ORB::create(10000, 1.2f, 8, 15, 0, 2, 0, 31, static_cast<int>(mThresh));
    else if (mDetectorAlgorithm == Extractor::eType::SURF)
        detector = cv::xfeatures2d::SurfFeatureDetector::create(mThresh);
    else if (mDetectorAlgorithm == Extractor::eType::SIFT)
        detector = cv::xfeatures2d::SiftFeatureDetector::create(0, 3, mThresh);

    detector->detect(image, keypoints);
}

void DetectorAdjuster::setDecreaseFactor(double new_factor) { mDecreaseFactor = new_factor; }

void DetectorAdjuster::setIncreaseFactor(double new_factor) { mIncreaseFactor = new_factor; }

void DetectorAdjuster::tooFew(int, int)
{
    mThresh *= mDecreaseFactor;
    if (mThresh < mMinThresh)
        mThresh = mMinThresh;
}

void DetectorAdjuster::tooMany(int, int)
{
    mThresh *= mIncreaseFactor;
    if (mThresh > mMaxThresh)
        mThresh = mMaxThresh;
}

bool DetectorAdjuster::good() const
{
    return (mThresh > mMinThresh) && (mThresh < mMaxThresh);
}

cv::Ptr<DetectorAdjuster> DetectorAdjuster::clone() const
{
    cv::Ptr<DetectorAdjuster> pNewObject(new DetectorAdjuster(mDetectorAlgorithm, mThresh, mMinThresh, mMaxThresh, mIncreaseFactor, mDecreaseFactor));
    return pNewObject;
}
