#include "utility.h"
#include "Core/frame.h"
#include "Core/intrinsicmatrix.h"
#include "Core/rgbdcamera.h"
#include "System/converter.h"
#include <Eigen/Core>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <sstream>

using namespace std;

void Utility::loadImages(const string& associationFilename, vector<string>& vImageFilenamesRGB,
    vector<string>& vImageFilenamesD, vector<double>& vTimestamps)
{
    ifstream fAssociation;
    fAssociation.open(associationFilename.c_str());
    while (!fAssociation.eof()) {
        string s;
        getline(fAssociation, s);
        if (!s.empty()) {
            stringstream ss;
            ss << s;
            double t;
            string sRGB, sD;
            ss >> t;
            vTimestamps.push_back(t);
            ss >> sRGB;
            vImageFilenamesRGB.push_back(sRGB);
            ss >> t;
            ss >> sD;
            vImageFilenamesD.push_back(sD);
        }
    }
}

RGBDcamera* Utility::loadCamera(const string& settigsFilename)
{
    cv::FileStorage fSettings(settigsFilename, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    float k1 = fSettings["Camera.k1"];
    float k2 = fSettings["Camera.k2"];
    float k3 = fSettings["Camera.k3"];
    float p1 = fSettings["Camera.p1"];
    float p2 = fSettings["Camera.p2"];

    float bf = fSettings["Camera.bf"];
    float fps = fSettings["Camera.fps"];
    float thDepth = (float)fSettings["ThDepth"];
    float depthMapFactor = fSettings["DepthMapFactor"];

    int imageWidth = fSettings["Camera.width"];
    int imageHeight = fSettings["Camera.height"];

    IntrinsicMatrix* pIntrinsic = new IntrinsicMatrix(fx, fy, cx, cy);
    pIntrinsic->setDistortion(k1, k2, k3, p1, p2);
    RGBDcamera* pCamera = new RGBDcamera(pIntrinsic, bf, thDepth, depthMapFactor, fps, imageWidth, imageHeight);

    return pCamera;
}

int Utility::PnPRansac(Frame& cur, vector<cv::DMatch>& vInliers)
{
    vector<cv::Point2f> v2D;
    vector<cv::Point3f> v3D;
    vector<size_t> vnIndex;

    for (const auto& m : vInliers) {
        size_t idx = m.trainIdx;
        cv::Mat Xw = cur.unprojectWorld(idx);
        const cv::KeyPoint& kpU = cur.mvKeysUn[idx];

        v2D.push_back(kpU.pt);
        v3D.push_back(cv::Point3f(Xw.at<float>(0), Xw.at<float>(1), Xw.at<float>(2)));
        vnIndex.push_back(idx);
        cur.setOutlier(idx);
    }

    if (v2D.size() < 10)
        return 0;

    cv::Mat r, t, inliers, T;
    cv::Rodrigues(cur.getRotation(), r);
    t = cur.getTranslation();
    bool bOK = cv::solvePnPRansac(v3D, v2D, cur.mpCamera->k(), cv::Mat(), r, t, true, 500, 3.0f, 0.85, inliers);

    if (bOK) {
        T = Converter::toHomogeneous(r, t);
        cur.setPose(T);

        for (int i = 0; i < inliers.rows; ++i) {
            int n = inliers.at<int>(i);
            const size_t idx = vnIndex[n];
            cur.setInlier(idx);
        }
    } else {
        cerr << "PnPRansac fail" << endl;
        terminate();
    }

    return inliers.rows;
}
