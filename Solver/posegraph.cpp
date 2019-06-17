#include "posegraph.h"
#include "Core/frame.h"
#include "Core/map.h"
#include "Features/matcher.h"
#include "LoopClosing/loopclosing.h"
#include "Solver/ricp.h"
#include "System/converter.h"
#include "System/random.h"
#include "System/tracking.h"
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
#include <mutex>
#include <unistd.h>

using namespace std;

typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> BlockSolver;
typedef g2o::LinearSolverCSparse<BlockSolver::PoseMatrixType> CSparseSolver;
typedef g2o::LinearSolverCholmod<BlockSolver::PoseMatrixType> CholmodSolver;
typedef g2o::LinearSolverPCG<BlockSolver::PoseMatrixType> PCGsolver;
typedef g2o::LinearSolverDense<BlockSolver::PoseMatrixType> DenseSolver;
typedef g2o::LinearSolverEigen<BlockSolver::PoseMatrixType> EigenSolver;

using namespace std;

PoseGraph::PoseGraph(Tracking* pTracker, LoopDetector::Ptr pLooper, Map::Ptr pMap)
    : mbFinishRequested(false)
    , mbFinished(true)
    , mpTracker(pTracker)
    , mpLoopCloser(pLooper)
    , mpMap(pMap)
    , mnKFsFromLastLoop(0)
{
    CSparseSolver* linearSolver = new CSparseSolver();
    linearSolver->setBlockOrdering(false);
    BlockSolver* solver_ptr = new BlockSolver(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* algorithm = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

    mOptimizer.setAlgorithm(algorithm);
    mOptimizer.setVerbose(false);

    mRunThread = thread(&PoseGraph::run, this);
}

void PoseGraph::run()
{
    mbFinished = false;

    while (true) {
        // Check if there are keyframes in the queue
        if (checkNewKeyFrames()) {
            updateGraph();

            // Pose-Graph global optimization
            if (detectLoop()) {
                cout << "Loop detected" << endl;
                optimize(20);
                mpTracker->correct();
                cout << endl;

                mpMap->informNewBigChange();
            }
            // Local optimization
            else {
                if (mpMap->keyFramesInMap() > 2) {
                    set<Frame::Ptr> vpConKFs = mpCurrentKF->getConnectedKFs();
                    vector<Frame::Ptr> vpKFs = mpMap->getAllKeyFrames();
                    for (Frame::Ptr pKF : vpKFs)
                        pKF->fixVertex(true);
                    for (Frame::Ptr pKF : vpConKFs)
                        pKF->fixVertex(false);

                    mOptimizer.initializeOptimization();
                    mOptimizer.optimize(10);

                    for (Frame::Ptr pKF : vpConKFs)
                        pKF->correctPose();
                }
            }

            // Add reliable landmarks to map
            for (size_t i = 0; i < mpCurrentKF->N; ++i) {
                Landmark::Ptr pLM = mpCurrentKF->getLandmark(i);
                if (!pLM)
                    continue;

                if (pLM->observations() > 5)
                    mpMap->addLandmark(pLM);
            }
        }

        if (checkFinish())
            break;

        usleep(3000);
    }

    setFinish();
}

void PoseGraph::updateGraph()
{
    {
        unique_lock<mutex> lock(mMutexQueue);
        mpCurrentKF = mlpKeyFrameQueue.front();
        mlpKeyFrameQueue.pop_front();
    }

    mpMap->addKeyFrame(mpCurrentKF);

    if (mpCurrentKF->getId() == 0) {
        createNode();
        mpReferenceKF = mpCurrentKF;
    } else {
        createNode();
        createEdgeWithReference();
        mpReferenceKF = mpCurrentKF;
    }

    createLocalEdges();
    createRandomEdges(15);
}

void PoseGraph::createLocalEdges()
{
    int nKFs = mpMap->keyFramesInMap();

    for (int i = nKFs - 1; i > nKFs - 10; i--) {
        if (i < 0)
            break;
        Frame::Ptr pKF = mpMap->getKeyFrameAt(i);
        if (pKF == mpCurrentKF)
            continue;

        if (existEdge(mpCurrentKF->getId(), pKF->getId()))
            continue;

        Matcher matcher(0.9f);
        vector<cv::DMatch> vMatches;
        matcher.match(pKF, mpCurrentKF, vMatches);
        if (vMatches.size() < 20)
            continue;

        RIcp sac(200, 20, 3.0f, 4);
        if (!sac.compute(pKF, mpCurrentKF, vMatches, false))
            continue;

        matchLandmarks(pKF, sac.mvInliers);
        createEdge(pKF, Converter::toIsometry3d(sac.mT21.cast<double>()));
    }
}

void PoseGraph::createRandomEdges(int n)
{
    int nKFs = mpMap->keyFramesInMap();

    // Try to add an edge with random KFs
    for (int i = 0; i < n; i++) {
        int index = Random::randomInt(0, nKFs - 1);

        Frame::Ptr pKF = mpMap->getKeyFrameAt(index);
        if (pKF == mpCurrentKF)
            continue;

        if (existEdge(mpCurrentKF->getId(), pKF->getId()))
            continue;

        Matcher matcher(0.9f);
        vector<cv::DMatch> vMatches;
        matcher.match(pKF, mpCurrentKF, vMatches);
        if (vMatches.size() < 20)
            continue;

        RIcp sac(200, 20, 3.0f, 4);
        if (!sac.compute(pKF, mpCurrentKF, vMatches, false))
            continue;

        matchLandmarks(pKF, sac.mvInliers);
        createEdge(pKF, Converter::toIsometry3d(sac.mT21.cast<double>()));
    }
}

void PoseGraph::createNode()
{
    g2o::VertexSE3* v = new g2o::VertexSE3();
    v->setId(mpCurrentKF->getId());
    v->setEstimate(Converter::toSE3Quat(mpCurrentKF->getPoseInverse()));
    v->setFixed(mpCurrentKF->getId() == 0);

    mpCurrentKF->setVertex(v);

    unique_lock<mutex> lock(mMutexOptimizer);
    mOptimizer.addVertex(v);
}

void PoseGraph::createEdgeWithReference()
{
    g2o::EdgeSE3* edge = new g2o::EdgeSE3();
    edge->setVertex(0, mpCurrentKF->getVertex());
    edge->setVertex(1, mpReferenceKF->getVertex());
    edge->setMeasurementFromState();
    edge->setInformation(Eigen::Matrix<double, 6, 6>::Identity() * 100.0);
    edge->setRobustKernel(new g2o::RobustKernelHuber());

    mpCurrentKF->addConnection(mpReferenceKF);
    mpReferenceKF->addConnection(mpCurrentKF);

    EdgeID id;
    id[mpReferenceKF->getId()] = mpCurrentKF->getId();

    unique_lock<mutex> lock(mMutexOptimizer);
    mEdges[id] = edge;
    mOptimizer.addEdge(edge);
}

double PoseGraph::createEdge(Frame::Ptr pKF, const Eigen::Isometry3d& T)
{
    g2o::EdgeSE3* edge = new g2o::EdgeSE3();
    edge->setVertex(0, mpCurrentKF->getVertex());
    edge->setVertex(1, pKF->getVertex());

    edge->setMeasurement(T);
    edge->setInformation(Eigen::Matrix<double, 6, 6>::Identity() * 100);
    edge->setRobustKernel(new g2o::RobustKernelHuber());
    edge->computeError();

    mpCurrentKF->addConnection(pKF);
    pKF->addConnection(mpCurrentKF);

    {
        EdgeID id;
        id[pKF->getId()] = mpCurrentKF->getId();

        unique_lock<mutex> lock(mMutexOptimizer);
        mEdges[id] = edge;
        mOptimizer.addEdge(edge);
    }

    return edge->chi2();
}

bool PoseGraph::detectLoop()
{
    // If the map contains less than 15 KF or less than 15 KF have passed from last loop detection
    {
        unique_lock<mutex> lock2(mMutexUpdate);
        if (mnKFsFromLastLoop < 15)
            return false;
    }

    vector<Frame::Ptr> vpCandidates = mpLoopCloser->obtainCandidates(mpCurrentKF);
    if (vpCandidates.empty())
        return false;

    mpTracker->loopCandidates(vpCandidates.size());

    for (Frame::Ptr pKF : vpCandidates) {
        if (existEdge(mpCurrentKF->getId(), pKF->getId()))
            continue;

        Matcher matcher(0.9f);
        vector<cv::DMatch> vMatches;
        matcher.match(pKF, mpCurrentKF, vMatches);

        uint th = mpTracker->getMeanInliers() * 0.20;

        if (vMatches.size() < th)
            continue;

        RIcp sac(200, th, 3.0f, 4);
        if (!sac.compute(pKF, mpCurrentKF, vMatches, false))
            continue;

        matchLandmarks(pKF, sac.mvInliers);
        createEdge(pKF, Converter::toIsometry3d(sac.mT21.cast<double>()));

        {
            unique_lock<mutex> lock2(mMutexUpdate);
            mnKFsFromLastLoop = 0;
        }

        return true;
    }

    return false;
}

void PoseGraph::insertKeyFrame(Frame::Ptr pKF)
{
    unique_lock<mutex> lock(mMutexQueue);
    mlpKeyFrameQueue.push_back(pKF);

    unique_lock<mutex> lock2(mMutexUpdate);
    mnKFsFromLastLoop++;
}

bool PoseGraph::checkNewKeyFrames()
{
    unique_lock<mutex> lock(mMutexQueue);
    return (!mlpKeyFrameQueue.empty());
}

size_t PoseGraph::totalEdges()
{
    unique_lock<mutex> lock(mMutexOptimizer);
    return mEdges.size();
}

vector<g2o::EdgeSE3*> PoseGraph::getEdges()
{
    unique_lock<mutex> lock(mMutexOptimizer);
    vector<g2o::EdgeSE3*> vEdges;
    vEdges.reserve(mEdges.size());

    for (auto& pair : mEdges)
        vEdges.push_back(pair.second);

    return vEdges;
}

bool PoseGraph::removeVertex(int id)
{
    unique_lock<mutex> lock(mMutexOptimizer);
    g2o::OptimizableGraph::VertexContainer vertices = mOptimizer.activeVertices();
    for (g2o::OptimizableGraph::VertexContainer::iterator it = vertices.begin(); it != vertices.end(); it++) {
        if ((*it)->id() == id) {
            mOptimizer.removeVertex(*it);
            return true;
        }
    }

    return false;
}

bool PoseGraph::removeEdge(int id)
{
    unique_lock<mutex> lock(mMutexOptimizer);
    g2o::OptimizableGraph::EdgeContainer edges = mOptimizer.activeEdges();
    for (g2o::OptimizableGraph::EdgeContainer::iterator it = edges.begin(); it != edges.end(); it++) {
        if ((*it)->id() == id) {
            mOptimizer.removeEdge(*it);
            return true;
        }
    }

    return false;
}

void PoseGraph::optimize(const int& iterations)
{
    unique_lock<mutex> lock(mMutexOptimizer);

    if (mOptimizer.vertices().size() > 5) {

        vector<Frame::Ptr> vpKFs = mpMap->getAllKeyFrames();
        for (Frame::Ptr pKF : vpKFs)
            pKF->fixVertex(pKF->getId() == 0);

        mOptimizer.initializeOptimization();
        mOptimizer.optimize(iterations);

        for (Frame::Ptr pKF : vpKFs)
            pKF->correctPose();

        cout << "Graph optimized" << endl;
    }
}

bool PoseGraph::existEdge(const int v1, const int v2)
{
    if (v1 == v2)
        return true;

    EdgeID e1, e2;
    e1[v1] = v2;
    e2[v2] = v1;

    unique_lock<mutex> lock(mMutexOptimizer);
    return mEdges.find(e1) != mEdges.end() || mEdges.find(e2) != mEdges.end();
}

void PoseGraph::matchLandmarks(Frame::Ptr pKF, vector<cv::DMatch>& inliers)
{
    for (auto& inlier : inliers) {
        Landmark::Ptr pLM = pKF->getLandmark(inlier.queryIdx);
        if (!pLM) {
            pLM = mpCurrentKF->getLandmark(inlier.trainIdx);
            if (!pLM) {
                cv::Mat Xw = mpCurrentKF->unprojectWorld(inlier.trainIdx);
                pLM = make_shared<Landmark>(Xw, mpMap, mpCurrentKF, inlier.trainIdx);
                pLM->addObservation(mpCurrentKF->getId(), inlier.trainIdx);
                pLM->addObservation(pKF->getId(), inlier.queryIdx);
                pLM->setColor(mpCurrentKF->mvKeysColor[inlier.trainIdx]);
                mpCurrentKF->addLandmark(pLM, inlier.trainIdx);
                pKF->addLandmark(pLM, inlier.queryIdx);
            } else {
                pKF->addLandmark(pLM, inlier.queryIdx);
                pLM->addObservation(pKF->getId(), inlier.queryIdx);
            }
        } else {
            mpCurrentKF->addLandmark(pLM, inlier.trainIdx);
            pLM->addObservation(mpCurrentKF->getId(), inlier.trainIdx);
        }
    }
}

void PoseGraph::requestFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

bool PoseGraph::isFinished()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}

bool PoseGraph::checkFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

void PoseGraph::setFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;
}

void PoseGraph::shutdown()
{
    requestFinish();
    while (!isFinished())
        usleep(5000);

    optimize();
    mpMap->informNewBigChange();

    if (mRunThread.joinable())
        mRunThread.join();
}
