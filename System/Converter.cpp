#include "Converter.h"
#include <opencv2/calib3d.hpp>

std::vector<cv::Mat> Converter::toDescriptorVector(const cv::Mat& Descriptors)
{
    std::vector<cv::Mat> vDesc;
    vDesc.reserve(Descriptors.rows);
    for (int j = 0; j < Descriptors.rows; j++)
        vDesc.push_back(Descriptors.row(j));

    return vDesc;
}

g2o::SE3Quat Converter::toSE3Quat(const cv::Mat& cvT)
{
    Eigen::Matrix<double, 3, 3> R;
    R << cvT.at<float>(0, 0), cvT.at<float>(0, 1), cvT.at<float>(0, 2),
        cvT.at<float>(1, 0), cvT.at<float>(1, 1), cvT.at<float>(1, 2),
        cvT.at<float>(2, 0), cvT.at<float>(2, 1), cvT.at<float>(2, 2);

    Eigen::Matrix<double, 3, 1> t(cvT.at<float>(0, 3), cvT.at<float>(1, 3),
        cvT.at<float>(2, 3));

    return g2o::SE3Quat(R, t);
}

cv::Mat Converter::toHomogeneous(const cv::Mat& r, const cv::Mat& t)
{
    cv::Mat R(3, 3, CV_32F);
    cv::Rodrigues(r, R);

    cv::Mat Tcw = cv::Mat::eye(4, 4, CV_32F);
    R.copyTo(Tcw.rowRange(0, 3).colRange(0, 3));
    t.copyTo(Tcw.rowRange(0, 3).col(3));

    return Tcw.clone();
}

Eigen::Isometry3d Converter::toIsometry3d(const Eigen::Matrix4d& m)
{
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
    Eigen::AngleAxisd angle(m.block<3, 3>(0, 0));
    T = angle;
    T.pretranslate(m.col(3).head<3>());
    return T;
}

cv::Mat Converter::toCvMat(const Eigen::Matrix<float, 4, 4>& m)
{
    cv::Mat cvMat(4, 4, CV_32F);
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            cvMat.at<float>(i, j) = m(i, j);

    return cvMat.clone();
}

cv::Mat Converter::toCvMat(const g2o::SE3Quat& SE3)
{
    Eigen::Matrix<double, 4, 4> eigMat = SE3.to_homogeneous_matrix();
    return toCvMat(eigMat);
}

cv::Mat Converter::toCvMat(const g2o::Sim3& Sim3)
{
    Eigen::Matrix3d eigR = Sim3.rotation().toRotationMatrix();
    Eigen::Vector3d eigt = Sim3.translation();
    double s = Sim3.scale();
    return toCvSE3(s * eigR, eigt);
}

cv::Mat Converter::toCvMat(const Eigen::Matrix<double, 4, 4>& m)
{
    cv::Mat cvMat(4, 4, CV_32F);
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            cvMat.at<float>(i, j) = m(i, j);

    return cvMat.clone();
}

cv::Mat Converter::toCvMat(const Eigen::Matrix3d& m)
{
    cv::Mat cvMat(3, 3, CV_32F);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            cvMat.at<float>(i, j) = m(i, j);

    return cvMat.clone();
}

cv::Mat Converter::toCvMat(const Eigen::Matrix<double, 3, 1>& m)
{
    cv::Mat cvMat(3, 1, CV_32F);
    for (int i = 0; i < 3; i++)
        cvMat.at<float>(i) = m(i);

    return cvMat.clone();
}

cv::Mat Converter::toCvSE3(const Eigen::Matrix<double, 3, 3>& R, const Eigen::Matrix<double, 3, 1>& t)
{
    cv::Mat cvMat = cv::Mat::eye(4, 4, CV_32F);

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++)
            cvMat.at<float>(i, j) = R(i, j);
    }

    for (int i = 0; i < 3; i++)
        cvMat.at<float>(i, 3) = t(i);

    return cvMat.clone();
}

cv::Mat Converter::toCvSE3(const cv::Mat& R, const cv::Mat t)
{
    cv::Mat cvMat = cv::Mat::eye(4, 4, CV_32F);

    R.copyTo(cvMat.rowRange(0, 3).colRange(0, 3));
    t.copyTo(cvMat.rowRange(0, 3).col(3));

    return cvMat.clone();
}

Eigen::Matrix<double, 3, 1> Converter::toVector3d(const cv::Mat& cvVector)
{
    Eigen::Matrix<double, 3, 1> v;
    v << cvVector.at<float>(0), cvVector.at<float>(1), cvVector.at<float>(2);

    return v;
}

Eigen::Matrix<double, 3, 1> Converter::toVector3d(const cv::Point3f& cvPoint)
{
    Eigen::Matrix<double, 3, 1> v;
    v << cvPoint.x, cvPoint.y, cvPoint.z;

    return v;
}

Eigen::Matrix<double, 3, 3> Converter::toMatrix3d(const cv::Mat& cvMat3)
{
    Eigen::Matrix<double, 3, 3> M;

    M << cvMat3.at<float>(0, 0), cvMat3.at<float>(0, 1), cvMat3.at<float>(0, 2),
        cvMat3.at<float>(1, 0), cvMat3.at<float>(1, 1), cvMat3.at<float>(1, 2),
        cvMat3.at<float>(2, 0), cvMat3.at<float>(2, 1), cvMat3.at<float>(2, 2);

    return M;
}

Eigen::Matrix<float, 4, 4> Converter::toMatrix4f(const cv::Mat& cvMat4)
{
    Eigen::Matrix<float, 4, 4> m;

    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            m(i, j) = cvMat4.at<float>(i, j);

    return m;
}

std::vector<float> Converter::toQuaternion(const cv::Mat& M)
{
    Eigen::Matrix<double, 3, 3> eigMat = toMatrix3d(M);
    Eigen::Quaterniond q(eigMat);

    std::vector<float> v(4);
    v[0] = q.x();
    v[1] = q.y();
    v[2] = q.z();
    v[3] = q.w();

    return v;
}
