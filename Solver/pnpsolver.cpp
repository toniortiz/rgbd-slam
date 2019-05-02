#include "pnpsolver.h"
#include "Core/frame.h"
#include "Core/rgbdcamera.h"
#include "System/converter.h"
#include <Eigen/StdVector>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/solvers/cholmod/linear_solver_cholmod.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/solvers/pcg/linear_solver_pcg.h>
#include <g2o/types/icp/types_icp.h>
#include <g2o/types/sba/types_six_dof_expmap.h>

using namespace std;

typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> BlockSolver;
typedef g2o::LinearSolverCSparse<BlockSolver::PoseMatrixType> CSparseSolver;
typedef g2o::LinearSolverCholmod<BlockSolver::PoseMatrixType> CholmodSolver;
typedef g2o::LinearSolverPCG<BlockSolver::PoseMatrixType> PCGsolver;
typedef g2o::LinearSolverDense<BlockSolver::PoseMatrixType> DenseSolver;
typedef g2o::LinearSolverEigen<BlockSolver::PoseMatrixType> EigenSolver;

PnPSolver::PnPSolver(const Frame::Ptr F1, Frame::Ptr F2, const vector<cv::DMatch>& matches)
    : Solver(F1, F2, matches)
{
}

bool PnPSolver::compute(vector<cv::DMatch>& vInliers)
{
    g2o::SparseOptimizer optimizer;
    BlockSolver::LinearSolverType* linearSolver = new DenseSolver();

    BlockSolver* solver_ptr = new BlockSolver(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    double fx = static_cast<double>(mF2->mpCamera->fx());
    double fy = static_cast<double>(mF2->mpCamera->fy());
    double cx = static_cast<double>(mF2->mpCamera->cx());
    double cy = static_cast<double>(mF2->mpCamera->cy());

    g2o::VertexSE3Expmap* vSE3 = new g2o::VertexSE3Expmap();
    vSE3->setEstimate(Converter::toSE3Quat(mF2->getPose()));
    vSE3->setId(0);
    vSE3->setFixed(false);
    optimizer.addVertex(vSE3);

    vector<g2o::EdgeSE3ProjectXYZOnlyPose*> vpEdges;
    vector<size_t> vnIndexEdge;
    vpEdges.reserve(mMatches.size());
    vnIndexEdge.reserve(mMatches.size());

    const double delta = sqrt(5.991);

    // true for inlier
    map<size_t, pair<cv::DMatch, bool>> matchesFlag;
    typedef map<size_t, pair<cv::DMatch, bool>>::iterator MatchesIt;

    for (const auto& m : mMatches) {
        size_t i = m.trainIdx;
        cv::Mat Xw = mF2->unprojectWorld(i);
        const cv::KeyPoint& kpUn = mF2->mvKeysUn[i];

        mF2->setInlier(i);
        matchesFlag[i] = { m, true };

        Eigen::Matrix<double, 2, 1> obs;
        obs << kpUn.pt.x, kpUn.pt.y;

        g2o::EdgeSE3ProjectXYZOnlyPose* e = new g2o::EdgeSE3ProjectXYZOnlyPose();

        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
        e->setMeasurement(obs);
        e->setInformation(Eigen::Matrix2d::Identity() * 1.0);

        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
        e->setRobustKernel(rk);
        rk->setDelta(delta);

        e->fx = fx;
        e->fy = fy;
        e->cx = cx;
        e->cy = cy;

        e->Xw[0] = static_cast<double>(Xw.at<float>(0));
        e->Xw[1] = static_cast<double>(Xw.at<float>(1));
        e->Xw[2] = static_cast<double>(Xw.at<float>(2));

        optimizer.addEdge(e);

        vpEdges.push_back(e);
        vnIndexEdge.push_back(i);
    }

    if (vpEdges.size() < 3)
        return false;

    for (size_t it = 0; it < 4; it++) {
        vSE3->setEstimate(Converter::toSE3Quat(mF2->getPose()));
        optimizer.initializeOptimization(0);
        optimizer.optimize(10);

        for (size_t i = 0; i < vpEdges.size(); i++) {
            g2o::EdgeSE3ProjectXYZOnlyPose* e = vpEdges[i];

            const size_t idx = vnIndexEdge[i];

            if (mF2->isOutlier(idx))
                e->computeError();

            const double chi2 = e->chi2();

            if (chi2 > 5.991) {
                mF2->setOutlier(idx);
                matchesFlag[idx].second = false;
                e->setLevel(1);
            } else {
                mF2->setInlier(idx);
                matchesFlag[idx].second = true;
                e->setLevel(0);
            }

            if (it == 2)
                e->setRobustKernel(nullptr);
        }

        if (optimizer.edges().size() < 10)
            break;
    }

    g2o::VertexSE3Expmap* vSE3_recov = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0));
    g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
    cv::Mat pose = Converter::toCvMat(SE3quat_recov);
    mF2->setPose(pose);

    vInliers.clear();
    for (MatchesIt it = matchesFlag.begin(); it != matchesFlag.end(); it++) {
        const bool status = it->second.second;
        if (status) {
            const cv::DMatch& m = it->second.first;
            vInliers.push_back(m);
        }
    }

    return true;
}
