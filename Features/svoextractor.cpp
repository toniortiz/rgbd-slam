#include "svoextractor.h"
#include <fast/fast.h>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;

SVOextractor::SVOextractor(int nLevels, int cellSize, double tresh)
    : mnLevels(nLevels)
    , mnCellSize(cellSize)
    , mThresh(tresh)
{
}

void halfSample(const cv::Mat& in, cv::Mat& out)
{
    assert(in.rows / 2 == out.rows && in.cols / 2 == out.cols);
    assert(in.type() == CV_8UC1 && out.type() == CV_8UC1);

    const int stride = in.step.p[0];
    uint8_t* top = (uint8_t*)in.data;
    uint8_t* bottom = top + stride;
    uint8_t* end = top + stride * in.rows;
    const int out_width = out.cols;
    uint8_t* p = (uint8_t*)out.data;
    while (bottom < end) {
        for (int j = 0; j < out_width; j++) {
            *p = static_cast<uint8_t>((uint16_t(top[0]) + top[1] + bottom[0] + bottom[1]) / 4);
            p++;
            top += 2;
            bottom += 2;
        }
        top += stride;
        bottom += stride;
    }
}

float ShiTomasiScore(const cv::Mat& img, int u, int v)
{
    assert(img.type() == CV_8UC1);

    float dXX = 0.0;
    float dYY = 0.0;
    float dXY = 0.0;
    const int halfbox_size = 4;
    const int box_size = 2 * halfbox_size;
    const int box_area = box_size * box_size;
    const int x_min = u - halfbox_size;
    const int x_max = u + halfbox_size;
    const int y_min = v - halfbox_size;
    const int y_max = v + halfbox_size;

    if (x_min < 1 || x_max >= img.cols - 1 || y_min < 1 || y_max >= img.rows - 1)
        return 0.0; // patch is too close to the boundary

    const int stride = img.step.p[0];
    for (int y = y_min; y < y_max; ++y) {
        const uint8_t* ptr_left = img.data + stride * y + x_min - 1;
        const uint8_t* ptr_right = img.data + stride * y + x_min + 1;
        const uint8_t* ptr_top = img.data + stride * (y - 1) + x_min;
        const uint8_t* ptr_bottom = img.data + stride * (y + 1) + x_min;
        for (int x = 0; x < box_size; ++x, ++ptr_left, ++ptr_right, ++ptr_top, ++ptr_bottom) {
            float dx = *ptr_right - *ptr_left;
            float dy = *ptr_bottom - *ptr_top;
            dXX += dx * dx;
            dYY += dy * dy;
            dXY += dx * dy;
        }
    }

    // Find and return smaller eigenvalue:
    dXX = dXX / (2.0 * box_area);
    dYY = dYY / (2.0 * box_area);
    dXY = dXY / (2.0 * box_area);
    return 0.5 * (dXX + dYY - sqrt((dXX + dYY) * (dXX + dYY) - 4 * (dXX * dYY - dXY * dXY)));
}

void SVOextractor::detect(cv::InputArray _image, vector<cv::KeyPoint>& keypoints, cv::InputArray mask)
{
    if (_image.empty())
        return;

    cv::Mat image = _image.getMat();
    assert(image.type() == CV_8UC1);

    mnGridCols = ceil(static_cast<double>(image.cols) / mnCellSize);
    mnGridRows = ceil(static_cast<double>(image.rows) / mnCellSize);
    mvbGridOccupancy = vector<bool>(mnGridCols * mnGridRows, false);

    vector<cv::KeyPoint> _keypoints(mnGridCols * mnGridRows, cv::KeyPoint(0, 0, 0, -1, 0, 0, -1));
    createImagePyramid(image);

    for (int L = 0; L < mnLevels; L++) {
        const int scale = (1 << L);

        cv::Mat& workingMat = mvImagePyramid[L];
        // cv::GaussianBlur(workingMat, workingMat, cv::Size(3, 3), 2, 2, cv::BORDER_REFLECT_101);

        vector<fast::fast_xy> vFastCorners;
        fast::fast_corner_detect_10((fast::fast_byte*)workingMat.data, workingMat.cols,
            workingMat.rows, workingMat.cols, mThresh, vFastCorners);

        vector<int> vScores;
        vector<int> vnmCorners;
        fast::fast_corner_score_10((fast::fast_byte*)workingMat.data, workingMat.cols,
            vFastCorners, 20, vScores);
        fast::fast_nonmax_3x3(vFastCorners, vScores, vnmCorners);

        for (auto it = vnmCorners.begin(), ite = vnmCorners.end(); it != ite; ++it) {
            fast::fast_xy& pt = vFastCorners.at(*it);
            const int k = static_cast<int>((pt.y * scale) / mnCellSize) * mnGridCols + static_cast<int>((pt.x * scale) / mnCellSize);

            if (mvbGridOccupancy[k])
                continue;
            const float score = ShiTomasiScore(workingMat, pt.x, pt.y);
            if (score > _keypoints.at(k).response) {
                cv::KeyPoint kp;
                kp.pt = cv::Point2f(pt.x * scale, pt.y * scale);
                kp.response = score;
                kp.octave = L;
                _keypoints.at(k) = kp;
            }
        }
    }

    for (auto& kp : _keypoints) {
        if (kp.response > 20.0)
            keypoints.push_back(kp);
    }

    resetGrid();
}

void SVOextractor::createImagePyramid(cv::Mat& image)
{
    mvImagePyramid.clear();
    mvImagePyramid.resize(mnLevels);
    mvImagePyramid[0] = image;
    for (int i = 1; i < mnLevels; ++i) {
        mvImagePyramid[i] = cv::Mat(mvImagePyramid[i - 1].rows / 2, mvImagePyramid[i - 1].cols / 2, CV_8UC1);
        halfSample(mvImagePyramid[i - 1], mvImagePyramid[i]);
    }
}

void SVOextractor::resetGrid()
{
    fill(mvbGridOccupancy.begin(), mvbGridOccupancy.end(), false);
}

void SVOextractor::setExistingKeyPoints(std::vector<cv::KeyPoint>& keypoints)
{
    for_each(keypoints.begin(), keypoints.end(), [&](cv::KeyPoint& kp) {
        mvbGridOccupancy.at(
            static_cast<int>(kp.pt.y / mnCellSize) * mnGridCols
            + static_cast<int>(kp.pt.x / mnCellSize))
            = true;
    });
}

void SVOextractor::setGridOccupancy(cv::Point2f& px)
{
    mvbGridOccupancy.at(
        static_cast<int>(px.y / mnCellSize) * mnGridCols
        + static_cast<int>(px.x / mnCellSize))
        = true;
}
