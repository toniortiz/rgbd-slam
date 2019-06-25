#ifndef OCTOMAPDRAWER_H
#define OCTOMAPDRAWER_H

#include "System/Macros.h"
#include <memory>
#include <mutex>
#include <octomap/ColorOcTree.h>
#include <octomap/octomap.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <set>

class Frame;

class OctomapDrawer {
public:
    SMART_POINTER_TYPEDEFS(OctomapDrawer);

    typedef pcl::PointXYZRGB PointT;
    typedef pcl::PointCloud<PointT> PointCloudT;

public:
    OctomapDrawer();

    void reset();

    bool save(const std::string& filename) const;

    void insertCloud(std::shared_ptr<Frame> pKF, const double maxRange = -1.0);

    void insertOctomapCloud(std::shared_ptr<octomap::Pointcloud> pCloud, PointCloudT::ConstPtr pColors,
        const octomap::point3d& origin, double maxrange = -1.0);

    // Filter by occupancy of voxels, e.g. remove points in free space
    void filter(PointCloudT::ConstPtr in, PointCloudT::Ptr out, double th);

    void render();

protected:
    octomap::ColorOcTree mOctomap;
    std::set<int> msIdxs;
};
#endif // OCTOMAPDRAWER_H
