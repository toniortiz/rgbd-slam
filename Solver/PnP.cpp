#include "PnP.h"
#include "Core/Frame.h"
#include "Core/Landmark.h"
#include "Core/RGBDcamera.h"
#include "System/Converter.h"
#include <opengv/absolute_pose/CentralAbsoluteAdapter.hpp>
#include <opengv/sac/Ransac.hpp>

using namespace std;

typedef opengv::absolute_pose::CentralAbsoluteAdapter Adapter;
typedef opengv::sac::Ransac<PnP::Problem> RANSAC;

PnP::PnP(std::shared_ptr<Frame> pFrame)
    : mpFrame(pFrame)
{
    for (size_t i = 0; i < pFrame->N; ++i) {
        Landmark::Ptr pMP = pFrame->getLandmark(i);
        if (!pMP)
            continue;

        cv::Mat Xw = pMP->getWorldPos();
        Bearing bv;
        pFrame->mpCamera->backproject(pFrame->mvKeys[i], bv);

        mvBearings.push_back(bv);
        mvLandmarks.push_back(Point3d(Xw.at<float>(0), Xw.at<float>(1), Xw.at<float>(2)));
        mvIndex.push_back(i);

        pFrame->setOutlier(i);
    }

    mFocalLength = static_cast<double>(pFrame->mpCamera->fx());
    twc = Converter::toVector3d(mpFrame->getCameraCenter());
    Rwc = Converter::toMatrix3d(mpFrame->getRotationInverse());

    setRansacParameters(1000, 1.0 - cos(atan(sqrt(2.0) * 0.5 / mFocalLength)), 0.99);
    setAlgorithm(KNEIP);
    refine(true);
}

void PnP::setRansacParameters(int iterations, double threshold, double probability)
{
    mnIterations = iterations;
    mThreshold = threshold;
    mProbability = probability;
}

PnP& PnP::setReprojectionTh(double pixels)
{
    mThreshold = (1.0 - cos(atan(pixels / mFocalLength)));
    return *this;
}

PnP& PnP::setAlgorithm(const PnP::eAlgorithm& algorithm)
{
    switch (algorithm) {
    case KNEIP:
        mAlgorithm = Problem::KNEIP;
        break;
    case GAO:
        mAlgorithm = Problem::GAO;
        break;
    case EPNP:
        mAlgorithm = Problem::EPNP;
        break;
    }

    return *this;
}

PnP& PnP::setIterations(int its)
{
    mnIterations = its;
    return *this;
}

PnP& PnP::refine(bool b)
{
    mbRefinement = b;
    return *this;
}

bool PnP::compute()
{
    Adapter adapter(mvBearings, mvLandmarks, twc, Rwc);
    shared_ptr<Problem> absposeproblem_ptr(new Problem(adapter, mAlgorithm));

    RANSAC ransac;
    ransac.sac_model_ = absposeproblem_ptr;
    ransac.probability_ = mProbability;
    ransac.threshold_ = mThreshold;
    ransac.max_iterations_ = mnIterations;
    bool bOK = ransac.computeModel();

    opengv::transformation_t T = ransac.model_coefficients_;

    if (bOK && mbRefinement) {
        opengv::transformation_t nlT;
        absposeproblem_ptr->optimizeModelCoefficients(ransac.inliers_, T, nlT);
        T = nlT;
    }

    if (bOK) {
        Eigen::Matrix4d Rt;
        Rt << T(0, 0), T(0, 1), T(0, 2), T(0, 3),
            T(1, 0), T(1, 1), T(1, 2), T(1, 3),
            T(2, 0), T(2, 1), T(2, 2), T(2, 3),
            0, 0, 0, 1;

        // opengv give us Twc, we need Tcw
        mpFrame->setPose(Converter::toMat<double, 4, 4>(Rt.inverse().matrix()));

        for (size_t i = 0; i < ransac.inliers_.size(); ++i)
            mpFrame->setInlier(mvIndex[ransac.inliers_.at(i)]);

    } else {
        for (size_t i = 0; i < mpFrame->N; ++i) {
            Landmark::Ptr pMP = mpFrame->getLandmark(i);
            if (!pMP)
                continue;
            mpFrame->setInlier(i);
        }
    }

    return bOK;
}
