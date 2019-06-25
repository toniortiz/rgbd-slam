#ifndef GX_SOLVER_H
#define GX_SOLVER_H

#include "System/Macros.h"
#include <opencv2/features2d.hpp>
#include <vector>

class Frame;

class Solver {
public:
    SMART_POINTER_TYPEDEFS(Solver);

public:
    Solver(const std::shared_ptr<Frame> F1, std::shared_ptr<Frame> F2, const std::vector<cv::DMatch>& matches);

    virtual ~Solver() {}

    virtual bool compute(std::vector<cv::DMatch>& inliers) = 0;

protected:
    const std::shared_ptr<Frame> mF1;
    std::shared_ptr<Frame> mF2;

    const std::vector<cv::DMatch>& mMatches;
};

#endif // SOLVER_H
