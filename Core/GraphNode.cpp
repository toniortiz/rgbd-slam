#include "GraphNode.h"
#include "frame.h"
#include "landmark.h"
#include "map.h"

using namespace std;

GraphNode::GraphNode(Frame::Ptr pKF, MapPtr pMap, const bool spanningParentIsNotSet)
    : mpOwnerKF(pKF)
    , mpMap(pMap)
    , mbSpanningTreeIsNotSet(spanningParentIsNotSet)
{
}

void GraphNode::addConnection(KeyFramePtr pKF, const unsigned int weight)
{
    bool needUpdate = false;
    {
        lock_guard<mutex> lock(mMutexNode);
        if (!mConnectedKFs.count(pKF)) {
            mConnectedKFs[pKF] = weight;
            needUpdate = true;
        } else if (mConnectedKFs.at(pKF) != weight) {
            mConnectedKFs.at(pKF) = weight;
            needUpdate = true;
        }
    }

    if (needUpdate)
        updateCovisibilityOrders();
}

void GraphNode::eraseConnection(KeyFramePtr pKF)
{
    bool needUpdate = false;
    {
        lock_guard<mutex> lock(mMutexNode);
        if (mConnectedKFs.count(pKF)) {
            mConnectedKFs.erase(pKF);
            needUpdate = true;
        }
    }

    if (needUpdate)
        updateCovisibilityOrders();
}

void GraphNode::eraseAllConnections()
{
    for (auto& [pKF, w] : mConnectedKFs)
        pKF->mpNode->eraseConnection(mpOwnerKF);

    mConnectedKFs.clear();
    mvpOrderedKFs.clear();
    mvOrderedWeights.clear();
}

void GraphNode::updateConnections()
{
    vector<LandmarkPtr> vpLMs = mpOwnerKF->getLandmarks();
    map<KeyFramePtr, unsigned int> KFcounter;
    for (LandmarkPtr pLM : vpLMs) {
        if (!pLM)
            continue;
        // if (pLM->isBad())
        //  continue;

        const map<Landmark::KeyFrameID, size_t> obs = pLM->getObservations();
        for (auto& [KFid, idx] : obs) {
            KeyFramePtr pKF = mpMap->getKeyFrame(KFid);
            if (!pKF)
                continue;
            if (!pKF->isKF())
                continue;
            if (pKF == mpOwnerKF)
                continue;

            KFcounter[pKF]++;
        }
    }

    if (KFcounter.empty())
        return;

    unsigned int maxWeight = 0;
    KeyFramePtr pKFmax = nullptr;

    vector<pair<unsigned int, KeyFramePtr>> vPairs;
    vPairs.reserve(KFcounter.size());
    for (auto& [pKF, w] : KFcounter) {
        if (maxWeight <= w) {
            maxWeight = w;
            pKFmax = pKF;
        }

        if (mWeightTh < w) {
            vPairs.emplace_back(make_pair(w, pKF));
            pKF->mpNode->addConnection(mpOwnerKF, w);
        }
    }
    if (vPairs.empty()) {
        vPairs.emplace_back(make_pair(maxWeight, pKFmax));
        pKFmax->mpNode->addConnection(mpOwnerKF, maxWeight);
    }

    sort(vPairs.rbegin(), vPairs.rend());

    decltype(mvpOrderedKFs) vOrderedKFs;
    vOrderedKFs.reserve(vPairs.size());
    decltype(mvOrderedWeights) vOrderedWeights;
    vOrderedWeights.reserve(vPairs.size());

    for (auto& [w, pKF] : vPairs) {
        vOrderedKFs.push_back(pKF);
        vOrderedWeights.push_back(w);
    }

    {
        lock_guard<mutex> lock(mMutexNode);
        mConnectedKFs = KFcounter;
        mvpOrderedKFs = vOrderedKFs;
        mvOrderedWeights = vOrderedWeights;

        if (mbSpanningTreeIsNotSet && mpOwnerKF->getId() != 0) {
            mpParent = pKFmax;
            mpParent->mpNode->addChild(mpOwnerKF);
            mbSpanningTreeIsNotSet = false;
        }
    }
}

void GraphNode::updateCovisibilityOrders()
{
    lock_guard<mutex> lock(mMutexNode);

    vector<pair<unsigned int, KeyFramePtr>> vPairs;
    vPairs.reserve(mConnectedKFs.size());

    for (auto& [pKF, w] : mConnectedKFs)
        vPairs.emplace_back(make_pair(w, pKF));

    sort(vPairs.rbegin(), vPairs.rend());

    mvpOrderedKFs.clear();
    mvpOrderedKFs.reserve(vPairs.size());
    mvOrderedWeights.clear();
    mvOrderedWeights.reserve(vPairs.size());

    for (auto& [w, pKF] : vPairs) {
        mvpOrderedKFs.push_back(pKF);
        mvOrderedWeights.push_back(w);
    }
}

set<GraphNode::KeyFramePtr> GraphNode::getConnectedKFs() const
{
    lock_guard<mutex> lock(mMutexNode);

    set<KeyFramePtr> spKFs;
    for (auto& [pKF, w] : mConnectedKFs)
        spKFs.insert(pKF);
    return spKFs;
}

vector<GraphNode::KeyFramePtr> GraphNode::getCovisibles() const
{
    lock_guard<mutex> lock(mMutexNode);
    return mvpOrderedKFs;
}

vector<GraphNode::KeyFramePtr> GraphNode::getBestNCovisibles(const unsigned int n) const
{
    lock_guard<mutex> lock(mMutexNode);

    if (mvpOrderedKFs.size() < n)
        return mvpOrderedKFs;
    else
        return vector<KeyFramePtr>(mvpOrderedKFs.begin(), mvpOrderedKFs.begin() + n);
}

vector<GraphNode::KeyFramePtr> GraphNode::getBestCovisiblesByWeight(const unsigned int w) const
{
    lock_guard<mutex> lock(mMutexNode);

    if (mvpOrderedKFs.empty())
        return vector<KeyFramePtr>();

    auto itr = upper_bound(mvOrderedWeights.begin(), mvOrderedWeights.end(), w, greater<unsigned int>());
    if (itr == mvOrderedWeights.end())
        return vector<KeyFramePtr>();
    else {
        const auto num = static_cast<unsigned int>(itr - mvOrderedWeights.begin());
        return vector<KeyFramePtr>(mvpOrderedKFs.begin(), mvpOrderedKFs.begin() + num);
    }
}

unsigned int GraphNode::getWeight(GraphNode::KeyFramePtr pKF) const
{
    lock_guard<mutex> lock(mMutexNode);

    if (mConnectedKFs.count(pKF))
        return mConnectedKFs.at(pKF);
    else
        return 0;
}

void GraphNode::setParent(GraphNode::KeyFramePtr pParent)
{
    lock_guard<mutex> lock(mMutexNode);
    mpParent = pParent;
}

GraphNode::KeyFramePtr GraphNode::getParent() const
{
    lock_guard<mutex> lock(mMutexNode);
    return mpParent;
}

void GraphNode::changeParent(GraphNode::KeyFramePtr pNewParent)
{
    lock_guard<mutex> lock(mMutexNode);
    mpParent = pNewParent;
    pNewParent->mpNode->addChild(mpOwnerKF);
}

void GraphNode::addChild(GraphNode::KeyFramePtr pChild)
{
    lock_guard<mutex> lock(mMutexNode);
    mspChildren.insert(pChild);
}

void GraphNode::eraseChild(GraphNode::KeyFramePtr pChild)
{
    lock_guard<mutex> lock(mMutexNode);
    mspChildren.erase(pChild);
}

void GraphNode::recoverSpanningConnections()
{
}

set<GraphNode::KeyFramePtr> GraphNode::getChildren() const
{
    lock_guard<mutex> lock(mMutexNode);
    return mspChildren;
}

bool GraphNode::hasChild(GraphNode::KeyFramePtr pKF) const
{
    lock_guard<mutex> lock(mMutexNode);
    return static_cast<bool>(mspChildren.count(pKF));
}

void GraphNode::addLoopEdge(GraphNode::KeyFramePtr pLoopKF)
{
    lock_guard<mutex> lock(mMutexNode);
    //mpOwnerKF->setNotErease();
    mspLoopEdges.insert(pLoopKF);
}

set<GraphNode::KeyFramePtr> GraphNode::getLoopEdges() const
{
    lock_guard<mutex> lock(mMutexNode);
    return mspLoopEdges;
}

bool GraphNode::hasLoopEdge() const
{
    lock_guard<mutex> lock(mMutexNode);
    return !mspLoopEdges.empty();
}

template <typename T, typename U>
vector<GraphNode::KeyFramePtr> GraphNode::extractIntersection(const T& KFs1, const U& KFs2)
{
    vector<KeyFramePtr> intersection;
    intersection.reserve(min(KFs1.size(), KFs2.size()));

    for (KeyFramePtr pKF1 : KFs1) {
        for (KeyFramePtr pKF2 : KFs2) {
            if (pKF1 == pKF2)
                intersection.push_back(pKF1);
        }
    }

    return intersection;
}
