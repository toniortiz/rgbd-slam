#ifndef GX_RANSACPNP_H
#define GX_RANSACPNP_H

#include "Solver.h"

class PnPRansac : public Solver {
public:
    PnPRansac(const std::shared_ptr<Frame> F1, std::shared_ptr<Frame> F2, const std::vector<cv::DMatch>& matches);

    ~PnPRansac() {}

    bool compute(std::vector<cv::DMatch>& inliers) override;
};

#endif // RANSACPNP_H
