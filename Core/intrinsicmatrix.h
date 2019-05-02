#ifndef INTRINSICMATRIX_H
#define INTRINSICMATRIX_H

#include <opencv2/core.hpp>

class IntrinsicMatrix {
public:
    IntrinsicMatrix();

    IntrinsicMatrix(float fx, float fy, float cx, float cy);

    // Copy constructor
    IntrinsicMatrix(const IntrinsicMatrix& obj);

    void setDistortion(float k1, float k2, float k3, float p1, float p2);

    void invertOffset();

    void scale(float factor);

    // Calibration matrix
    cv::Mat mK;
    cv::Mat mDistCoef;

    // Individual calibration params
    float mFx, mFy, mCx, mCy, mInvfx, mInvfy;

    // Distortion coefficients
    float mK1, mK2, mK3, mP1, mP2;
};

#endif // INTRINSICMATRIX_H
