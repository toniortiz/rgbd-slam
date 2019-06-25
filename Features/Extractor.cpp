#include "Extractor.h"
#include "DetectorAdjuster.h"
#include "ORBextractor.h"
#include "SVOextractor.h"
#include "StatefulFeatureDetector.h"
#include "VideoDynamicAdaptedFeatureDetector.h"
#include "VideoGridAdaptedFeatureDetector.h"
#include <iostream>
#include <opencv2/xfeatures2d.hpp>

using namespace std;

int Extractor::mNorm;

Extractor::Extractor(eType detector, eType descriptor, eMode mode)
{
    mDetectorType = detector;
    mDescriptorType = descriptor;
    mMode = mode;

    setParameters(1000, 1.2f, 8, 20, 7);
}

void Extractor::setParameters(int _nfeatures, float _scaleFactor, int _nlevels, int _iniThFAST, int _minThFAST)
{
    nfeatures = _nfeatures;
    scaleFactor = _scaleFactor;
    nlevels = _nlevels;
    iniThFAST = _iniThFAST;
    minThFAST = _minThFAST;

    if (mMode == ADAPTIVE) {
        if (mDetectorType == FAST || mDetectorType == GFTT || mDetectorType == SURF || mDetectorType == SIFT || mDetectorType == ORB) {
            createAdaptiveDetector();
            init();
            cout << "- Adaptive Feature Extraction" << endl;
        } else {
            cerr << "Not supported adaptive detector" << endl;
            terminate();
        }
    } else if (mMode == NORMAL) {
        createDetector();
        cout << "- Normal Feature Extraction" << endl;
    }

    createDescriptor();
    mNorm = mpDescriptor->defaultNorm();
}

void Extractor::detectAndCompute(cv::InputArray image, cv::InputArray mask, std::vector<cv::KeyPoint>& keypoints, cv::OutputArray descriptors)
{
    if (mDetectorType == ORB2 && mDescriptorType == ORB2) {
        mpDetector->detectAndCompute(image, mask, keypoints, descriptors);
    } else {
        mpDetector->detect(image, keypoints);
        if (keypoints.size() > nfeatures)
            cv::KeyPointsFilter::retainBest(keypoints, nfeatures);

        mpDescriptor->compute(image, keypoints, descriptors);
    }
}

void Extractor::init()
{
    mvScaleFactor.resize(nlevels);
    mvLevelSigma2.resize(nlevels);
    mvScaleFactor[0] = 1.0f;
    mvLevelSigma2[0] = 1.0f;
    for (int i = 1; i < nlevels; i++) {
        mvScaleFactor[i] = mvScaleFactor[i - 1] * scaleFactor;
        mvLevelSigma2[i] = mvScaleFactor[i] * mvScaleFactor[i];
    }

    mvInvScaleFactor.resize(nlevels);
    mvInvLevelSigma2.resize(nlevels);
    for (int i = 0; i < nlevels; i++) {
        mvInvScaleFactor[i] = 1.0f / mvScaleFactor[i];
        mvInvLevelSigma2[i] = 1.0f / mvLevelSigma2[i];
    }
}

void Extractor::createAdaptiveDetector()
{
    DetectorAdjuster* pAdjuster = nullptr;

    if (mDetectorType == FAST)
        pAdjuster = new DetectorAdjuster(mDetectorType, 20, 2, 10000, 1.3, 0.7);
    else if (mDetectorType == GFTT)
        pAdjuster = new DetectorAdjuster(mDetectorType, 3, 1, 10, 1.3, 0.7);
    else if (mDetectorType == SURF)
        pAdjuster = new DetectorAdjuster(mDetectorType, 200, 2, 10000, 1.3, 0.7);
    else if (mDetectorType == SIFT)
        pAdjuster = new DetectorAdjuster(mDetectorType, 0.04, 0.0001, 10000, 1.3, 0.7);
    else if (mDetectorType == ORB)
        pAdjuster = new DetectorAdjuster(mDetectorType, 20, 2, 10000, 1.3, 0.7);

    const int minFeatures = 600;
    const int maxFeatures = minFeatures * 1.7;
    const int iterations = 5;
    const int edgeTh = 31;

    const int gridResolution = 3;
    int gridCells = gridResolution * gridResolution;
    int gridMin = round(minFeatures / static_cast<float>(gridCells));
    int gridMax = round(maxFeatures / static_cast<float>(gridCells));

    StatefulFeatureDetector* pDetector = new VideoDynamicAdaptedFeatureDetector(pAdjuster, gridMin, gridMax, iterations);
    mpDetector.reset(new VideoGridAdaptedFeatureDetector(pDetector, maxFeatures, gridResolution, gridResolution, edgeTh));
}

int Extractor::getLevels()
{
    if (mDetectorType == ORB2 && mDescriptorType == ORB2)
        return dynamic_cast<ORBextractor*>(mpDetector.get())->GetLevels();
    else
        return nlevels;
}

float Extractor::getScaleFactor()
{
    if (mDetectorType == ORB2 && mDescriptorType == ORB2)
        return dynamic_cast<ORBextractor*>(mpDetector.get())->GetScaleFactor();
    else
        return scaleFactor;
}

vector<float> Extractor::getScaleFactors()
{
    if (mDetectorType == ORB2 && mDescriptorType == ORB2)
        return dynamic_cast<ORBextractor*>(mpDetector.get())->GetScaleFactors();
    else
        return mvScaleFactor;
}

vector<float> Extractor::getInverseScaleFactors()
{
    if (mDetectorType == ORB2 && mDescriptorType == ORB2)
        return dynamic_cast<ORBextractor*>(mpDetector.get())->GetInverseScaleFactors();
    else {
        return mvInvScaleFactor;
    }
}

vector<float> Extractor::getScaleSigmaSquares()
{
    if (mDetectorType == ORB2 && mDescriptorType == ORB2)
        return dynamic_cast<ORBextractor*>(mpDetector.get())->GetScaleSigmaSquares();
    else
        return mvLevelSigma2;
}

vector<float> Extractor::getInverseScaleSigmaSquares()
{
    if (mDetectorType == ORB2 && mDescriptorType == ORB2)
        return dynamic_cast<ORBextractor*>(mpDetector.get())->GetInverseScaleSigmaSquares();
    else
        return mvInvLevelSigma2;
}

void Extractor::createDetector()
{
    switch (mDetectorType) {
    case ORB:
        init();
        mpDetector = cv::ORB::create(nfeatures, scaleFactor, nlevels);
        break;

    case FAST:
        init();
        mpDetector = cv::FastFeatureDetector::create();
        break;

    case SVO:
        init();
        mpDetector.reset(new SVOextractor(nlevels, 5, 20));
        break;

    case GFTT:
        init();
        mpDetector = cv::GFTTDetector::create(nfeatures, 0.01, 3, 3, false, 0.04);
        break;

    case STAR:
        init();
        mpDetector = cv::xfeatures2d::StarDetector::create();
        break;

    case BRISK:
        init();
        mpDetector = cv::BRISK::create(30, nlevels, 1.0f);
        break;

    case SURF:
        init();
        mpDetector = cv::xfeatures2d::SURF::create(100, 4, 3, false, false);
        break;

    case SIFT:
        init();
        mpDetector = cv::xfeatures2d::SIFT::create(nfeatures);
        break;

    case ORB2:
        mpDetector.reset(new ORBextractor(nfeatures, scaleFactor, nlevels, iniThFAST, minThFAST));
        break;

    default:
        cout << "Invalid detector!" << endl;
        std::terminate();
    }
}

void Extractor::createDescriptor()
{
    switch (mDescriptorType) {
    case ORB:
        mpDescriptor = cv::ORB::create();
        break;

    case BRISK:
        mpDescriptor = cv::BRISK::create();
        break;

    case BRIEF:
        mpDescriptor = cv::xfeatures2d::BriefDescriptorExtractor::create();
        break;

    case FREAK:
        mpDescriptor = cv::xfeatures2d::FREAK::create();
        break;

    case SURF:
        mpDescriptor = cv::xfeatures2d::SURF::create();
        break;

    case SIFT:
        mpDescriptor = cv::xfeatures2d::SIFT::create();
        break;

    case LATCH:
        mpDescriptor = cv::xfeatures2d::LATCH::create();
        break;

    case ORB2:
        // Only for update mNorm
        mpDescriptor = cv::ORB::create();
        break;

    default:
        cout << "Invalid descriptor!" << endl;
        std::terminate();
    }
}
