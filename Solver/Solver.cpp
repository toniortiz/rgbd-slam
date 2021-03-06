#include "Solver.h"
#include "Core/Frame.h"

using namespace std;

Solver::Solver(const Frame::Ptr F1, Frame::Ptr F2, const vector<cv::DMatch>& matches)
    : mF1(F1)
    , mF2(F2)
    , mMatches(matches)
{
}
