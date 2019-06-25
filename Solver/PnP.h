#ifndef PNP_H
#define PNP_H

#include "System/Macros.h"
#include <opencv2/features2d.hpp>
#include <opengv/sac_problems/absolute_pose/AbsolutePoseSacProblem.hpp>

class Frame;

class PnP {
public:
    SMART_POINTER_TYPEDEFS(PnP);

    enum eAlgorithm {
        KNEIP = 1, // P3P
        GAO = 2, // P3P
        EPNP = 3, // P6P
    };

    typedef opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem Problem;
    typedef opengv::bearingVector_t Bearing;
    typedef opengv::bearingVectors_t BearingsVector;
    typedef opengv::point_t Point3d;
    typedef opengv::points_t Points3dVector;

public:
    PnP(std::shared_ptr<Frame> pFrame);

    void setRansacParameters(int iterations, double threshold, double probability);

    // Reprojection threshold expresed as pixels tolerance
    PnP& setReprojectionTh(double pixels);
    PnP& setAlgorithm(const eAlgorithm& algorithm);
    PnP& setIterations(int its);
    PnP& refine(bool b);

    bool compute();

protected:
    std::shared_ptr<Frame> mpFrame;

    BearingsVector mvBearings;
    Points3dVector mvLandmarks;

    std::vector<size_t> mvIndex;

    // Calibration
    double mFocalLength;

    // Ransac parameters
    int mnIterations;
    double mThreshold;
    double mProbability;

    bool mbRefinement;

    // Guess
    opengv::translation_t twc;
    opengv::rotation_t Rwc;

    Problem::Algorithm mAlgorithm;
};

#endif // PNP_H
