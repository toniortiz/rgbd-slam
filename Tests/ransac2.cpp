#include "Core/frame.h"
#include "Core/map.h"
#include "Core/rgbdcamera.h"
#include "Features/extractor.h"
#include "Features/matcher.h"
#include "Solver/ransac.h"
#include "System/converter.h"
#include "System/random.h"
#include "System/tracking.h"
#include "System/utility.h"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <thread>
#include <unistd.h>

using namespace std;
using namespace chrono;

const string baseDir = "/home/antonio/Documents/M.C.C/Tesis/Dataset/rgbd_dataset_freiburg1_desk/";
const string vocDir = "./Vocabulary/voc_fr1_GFTT_BRIEF.yml.gz";
const string settingsDir = "./Settings/TUM1.yaml";

int main(int argc, char* argv[])
{
    Random::initSeed();

    vector<string> vImageFilenamesRGB;
    vector<string> vImageFilenamesD;
    vector<double> vTimestamps;
    string associationFilename = string(baseDir + "associations.txt");
    Utility::loadImages(associationFilename, vImageFilenamesRGB, vImageFilenamesD, vTimestamps);

    size_t nImages = vImageFilenamesRGB.size();
    if (nImages == 0) {
        cerr << "\nNo images found in provided path." << endl;
        return 1;
    } else if (vImageFilenamesD.size() != vImageFilenamesRGB.size()) {
        cerr << "\nDifferent number of images for rgb and depth." << endl;
        return 1;
    }

    RGBDcamera::Ptr camera(Utility::loadCamera(settingsDir));
    Extractor::Ptr extractor(new Extractor(Extractor::GFTT, Extractor::BRIEF, Extractor::NORMAL));

    shared_ptr<DBoW3::Vocabulary> voc(new DBoW3::Vocabulary(vocDir));
    Map::Ptr pMap(new Map());
    //    Tracking tracker(voc, pMap);

    cout << "\n-------\n"
         << "Start processing sequence: " << baseDir << endl
         << "Images in the sequence: " << nImages << endl
         << endl;

    cv::TickMeter tm;
    cv::Mat imRGB, imD;
    Frame::Ptr refFrame = nullptr;
    ofstream f;
    f.open("CameraTrajectory.txt");
    f << fixed;

    Ransac::parameters params;
    params.verbose = 0;
    params.errorVersionVO = 0;
    params.errorVersionMap = 0;
    params.inlierThresholdEuclidean = 0.04;
    params.inlierThresholdReprojection = 2.0;
    params.inlierThresholdMahalanobis = 0.0002;
    params.minimalInlierRatioThreshold = 0.2;
    params.minimalNumberOfMatches = 15;
    params.usedPairs = 3;
    params.errorVersion = Ransac::EUCLIDEAN_ERROR;

    for (size_t ni = 0; ni < nImages; ni++) {
        tm.start();
        Frame::Ptr frame = camera->createFrame(baseDir, vImageFilenamesRGB[ni], vImageFilenamesD[ni], vTimestamps[ni], extractor);
        frame->drawObservations();

        if (ni == 0) {
            frame->setPose(cv::Mat::eye(4, 4, CV_32F));
        } else {
            Matcher matcher(0.9f);
            vector<cv::DMatch> matches;
            matcher.match(refFrame, frame, matches, false);

            vector<Eigen::Vector3f> prevFeatures(refFrame->N);
            for (size_t i = 0; i < refFrame->N; ++i) {
                const cv::Point3f& pr = refFrame->mvKeys3Dc[i];
                prevFeatures[i] = Eigen::Vector3f(pr.x, pr.y, pr.z);
            }
            vector<Eigen::Vector3f> features(frame->N);
            for (size_t i = 0; i < frame->N; ++i) {
                const cv::Point3f& fr = frame->mvKeys3Dc[i];
                features[i] = Eigen::Vector3f(fr.x, fr.y, fr.z);
            }

            Ransac ransac(params, camera);
            vector<cv::DMatch> inliers;
            Eigen::Matrix4f T = ransac.estimateTransformation(prevFeatures, features, matches, inliers);

            frame->setPose(Converter::toMat<float, 4, 4>(T).inv() * refFrame->getPose());
        }

        //        tracker.track(frame);
        //        tracker.setCurrentPose(frame->getPose());

        tm.stop();
        f << *frame;
        refFrame = frame;

        //        tracker.setTime(tm.getTimeSec() / tm.getCounter());
    }

    f.close();

    //    tracker.shutdown();
    //    tracker.saveKeyFrameTrajectory("KeyFrameTrajectory.txt");
    //    tracker.saveCameraTrajectory("CameraTrajectory.txt");

    return 0;
}

