#ifndef EXTRACTOR_H
#define EXTRACTOR_H

#include "System/macros.h"
#include <mutex>
#include <opencv2/features2d.hpp>
#include <string>

class Extractor {
public:
    SMART_POINTER_TYPEDEFS(Extractor);

    enum eType {
        ORB = 0,
        ORB2,
        SVO,
        FAST,
        GFTT,
        STAR,
        BRISK,
        FREAK,
        BRIEF,
        LATCH,
        SURF,
        SIFT
    };

    enum eMode {
        NORMAL = 0,
        ADAPTIVE
    };

    eType mDetectorType, mDescriptorType;
    eMode mMode;

    Extractor(eType detector, eType descriptor, eMode mode);

    void setParameters(int nfeatures, float scaleFactor, int nlevels, int iniThFAST, int minThFAST);

    void detectAndCompute(cv::InputArray image, cv::InputArray mask, std::vector<cv::KeyPoint>& keypoints, cv::OutputArray descriptors);

    void init();
    void createAdaptiveDetector();
    void createDetector();
    void createDescriptor();

    int getLevels();
    float getScaleFactor();
    std::vector<float> getScaleFactors();
    std::vector<float> getInverseScaleFactors();
    std::vector<float> getScaleSigmaSquares();
    std::vector<float> getInverseScaleSigmaSquares();

    // L2 or HAMMING
    static int mNorm;

private:
    std::mutex mMutexExtraction;
    cv::Ptr<cv::DescriptorExtractor> mpDescriptor;
    cv::Ptr<cv::FeatureDetector> mpDetector;

    int nfeatures;
    double scaleFactor;
    int nlevels;
    int iniThFAST;
    int minThFAST;

    std::vector<float> mvScaleFactor;
    std::vector<float> mvInvScaleFactor;
    std::vector<float> mvLevelSigma2;
    std::vector<float> mvInvLevelSigma2;
};

#endif // EXTRACTOR_H
