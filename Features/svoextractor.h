#ifndef SVOEXTRACTOR_H
#define SVOEXTRACTOR_H

#include <opencv2/features2d.hpp>
#include <vector>

class SVOextractor : public cv::Feature2D {
public:
    SVOextractor(int nLevels, int cellSize, double tresh);

    ~SVOextractor() {}

    CV_WRAP virtual void detect(cv::InputArray image, CV_OUT std::vector<cv::KeyPoint>& keypoints,
        cv::InputArray mask = cv::noArray());

    std::vector<cv::Mat> mvImagePyramid;

protected:
    void createImagePyramid(cv::Mat& image);
    void resetGrid();
    void setExistingKeyPoints(std::vector<cv::KeyPoint>& keypoints);
    void setGridOccupancy(cv::Point2f& px);

    double mThresh;
    int mnLevels;
    int mnCellSize;
    int mnGridCols;
    int mnGridRows;
    std::vector<bool> mvbGridOccupancy;
};

#endif // SVOEXTRACTOR_H
