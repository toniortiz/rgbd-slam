#ifndef GRAPHNODE_H
#define GRAPHNODE_H

#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <vector>

class Frame;
class Landmark;
class Map;

class GraphNode {
public:
    typedef std::shared_ptr<Frame> KeyFramePtr;
    typedef std::shared_ptr<Landmark> LandmarkPtr;
    typedef std::shared_ptr<Map> MapPtr;

public:
    GraphNode(KeyFramePtr pKF, MapPtr pMap, const bool spanningParentIsNotSet = true);

    ~GraphNode() = default;

    // Covisibility graph functions
    void addConnection(KeyFramePtr pKF, const unsigned int weight);
    void eraseConnection(KeyFramePtr pKF);
    void eraseAllConnections();
    void updateConnections();
    void updateCovisibilityOrders();
    std::set<KeyFramePtr> getConnectedKFs() const;
    std::vector<KeyFramePtr> getCovisibles() const;
    std::vector<KeyFramePtr> getBestNCovisibles(const unsigned int n) const;
    std::vector<KeyFramePtr> getBestCovisiblesByWeight(const unsigned int w) const;
    unsigned int getWeight(KeyFramePtr pKF) const;

    // Spanning tree functions
    void setParent(KeyFramePtr pParent);
    KeyFramePtr getParent() const;
    void changeParent(KeyFramePtr pNewParent);
    void addChild(KeyFramePtr pChild);
    void eraseChild(KeyFramePtr pChild);
    void recoverSpanningConnections();
    std::set<KeyFramePtr> getChildren() const;
    bool hasChild(KeyFramePtr pKF) const;

    // Loop edges
    void addLoopEdge(KeyFramePtr pLoopKF);
    std::set<KeyFramePtr> getLoopEdges() const;
    bool hasLoopEdge() const;

protected:
    template <typename T, typename U>
    static std::vector<KeyFramePtr> extractIntersection(const T& KFs1, const U& KFs2);

    // KeyFrame of this node
    KeyFramePtr const mpOwnerKF;

    MapPtr mpMap;

    // All connected KFs and ther weights
    std::map<KeyFramePtr, unsigned int> mConnectedKFs;

    // Minimum threshold for covisibility graph connection
    static constexpr unsigned int mWeightTh = 15;

    // Covisibility KFs in descending order on weights
    std::vector<KeyFramePtr> mvpOrderedKFs;
    std::vector<unsigned int> mvOrderedWeights;

    // Spanning tree
    KeyFramePtr mpParent = nullptr;
    std::set<KeyFramePtr> mspChildren;
    bool mbSpanningTreeIsNotSet;

    std::set<KeyFramePtr> mspLoopEdges;

    mutable std::mutex mMutexNode;
};

#endif // GRAPHNODE_H
