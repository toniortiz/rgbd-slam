#ifndef ICP_H
#define ICP_H

#include "Solver.h"
#include <opencv2/features2d.hpp>
#include <vector>

class Frame;

class Icp : public Solver {
public:
    Icp(const std::shared_ptr<Frame> F1, std::shared_ptr<Frame> F2, const std::vector<cv::DMatch>& matches);

    virtual ~Icp() {}

    bool compute(std::vector<cv::DMatch>& inliers) override;
};

#endif // ICP_H
