#ifndef UTILITY_H
#define UTILITY_H

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <string>
#include <vector>

class RGBDcamera;
class Frame;

class Utility {
public:
    static void loadImages(const std::string& associationFilename, std::vector<std::string>& vImageFilenamesRGB,
        std::vector<std::string>& vImageFilenamesD, std::vector<double>& vTimestamps);

    static RGBDcamera* loadCamera(const std::string& settigsFilename);

    static int PnPRansac(Frame& cur, std::vector<cv::DMatch>& vInliers);
};
#endif // UTILITY_H
