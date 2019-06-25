#ifndef PNPSOLVER_H
#define PNPSOLVER_H

#include "Solver.h"

class PnPSolver : public Solver {
public:
    PnPSolver(const std::shared_ptr<Frame> F1, std::shared_ptr<Frame> F2, const std::vector<cv::DMatch>& matches);

    virtual ~PnPSolver() {}

    bool compute(std::vector<cv::DMatch>& inliers) override;
};

#endif // PNPSOLVER_H
