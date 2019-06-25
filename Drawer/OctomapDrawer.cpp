#include "OctomapDrawer.h"
#include "Core/Frame.h"
#include <fstream>
#include <pangolin/pangolin.h>
#include <pcl/common/transforms.h>

using namespace std;

OctomapDrawer::OctomapDrawer()
    : mOctomap(0.05)
{
    reset();
}

void OctomapDrawer::reset()
{
    msIdxs.clear();
    mOctomap.clear();
    mOctomap.setClampingThresMin(0.001);
    mOctomap.setClampingThresMax(0.999);
    mOctomap.setResolution(0.08);
    mOctomap.setOccupancyThres(0.5);
    mOctomap.setProbHit(0.9);
    mOctomap.setProbMiss(0.4);
}

bool OctomapDrawer::save(const string& filename) const
{
    ofstream ofile(filename, std::ios_base::out | std::ios_base::binary);
    if (!ofile.is_open())
        return false;

    mOctomap.write(ofile);
    ofile.close();
    return true;
}

void OctomapDrawer::insertCloud(Frame::Ptr pKF, const double maxRange)
{
    if (msIdxs.count(pKF->id()) > 0)
        return;

    pKF->updateSensor();
    msIdxs.insert(pKF->id());

    Eigen::Vector4f t = pKF->getOrigin();
    octomap::point3d origin(t[0], t[1], t[2]);

    PointCloudT::Ptr pWorldCloud = pKF->unprojectWorldCloud();

    shared_ptr<octomap::Pointcloud> octomapCloud(new octomap::Pointcloud());
    octomapCloud->reserve(pWorldCloud->size());
    for (PointCloudT::const_iterator it = pWorldCloud->begin(); it != pWorldCloud->end(); it++) {
        if (!isnan(it->z) || it->z <= 0)
            octomapCloud->push_back(it->x, it->y, it->z);
    }

    insertOctomapCloud(octomapCloud, pWorldCloud, origin, maxRange);
}

void OctomapDrawer::insertOctomapCloud(shared_ptr<octomap::Pointcloud> pCloud, PointCloudT::ConstPtr pColors,
    const octomap::point3d& origin, double maxrange)
{
    mOctomap.insertPointCloud(*pCloud, origin, maxrange, true);

    unsigned char* colors = new unsigned char[3];
    PointCloudT::const_iterator it;
    for (it = pColors->begin(); it != pColors->end(); it++) {
        if (!isnan(it->x) && !isnan(it->y) && !isnan(it->z)) {
            const int rgb = *reinterpret_cast<const int*>(&(it->rgb));
            colors[0] = ((rgb >> 16) & 0xff);
            colors[1] = ((rgb >> 8) & 0xff);
            colors[2] = (rgb & 0xff);
            mOctomap.averageNodeColor(it->x, it->y, it->z, colors[0], colors[1], colors[2]);
        }
    }

    mOctomap.updateInnerOccupancy();
}

void OctomapDrawer::filter(PointCloudT::ConstPtr in, PointCloudT::Ptr out, double th)
{
    if (out->points.capacity() < in->size())
        out->reserve(in->size());

    Eigen::Quaternionf q = in->sensor_orientation_;
    Eigen::Vector4f t = in->sensor_origin_;

    size_t size = in->size();
    size_t outidx = 0;
    for (size_t inidx = 0; inidx < size; ++inidx) {
        const PointT& inpoint = (*in)[inidx];
        Eigen::Vector3f invec = q * inpoint.getVector3fMap() + t.head<3>();
        if (isnan(invec.z()))
            continue;

        const int radius = 1;
        int x_a = mOctomap.coordToKey(invec.x()) - radius;
        int x_b = mOctomap.coordToKey(invec.x()) + radius;
        int y_a = mOctomap.coordToKey(invec.y()) - radius;
        int y_b = mOctomap.coordToKey(invec.y()) + radius;
        int z_a = mOctomap.coordToKey(invec.z()) - radius;
        int z_b = mOctomap.coordToKey(invec.z()) + radius;

        double sumOccupancy = 0, sumWeights = 0;
        for (; x_a <= x_b; ++x_a) {
            for (; y_a <= y_b; ++y_a) {
                for (; z_a <= z_b; ++z_a) {
                    octomap::OcTreeNode* node = mOctomap.search(octomap::OcTreeKey(x_a, y_a, z_a));
                    if (node != NULL) {
                        double dx = mOctomap.keyToCoord(x_a) - invec.x();
                        double dy = mOctomap.keyToCoord(y_a) - invec.y();
                        double dz = mOctomap.keyToCoord(z_a) - invec.z();
                        double weight = dx * dx + dy * dy + dz * dz;
                        double weighted_occ = node->getOccupancy() / weight;
                        sumWeights += weight;
                        sumOccupancy += weighted_occ;
                    }
                }
            }
        }

        if (sumOccupancy < th * sumWeights) {
            PointT& outpoint = (*out)[outidx];
            outpoint = inpoint;
            ++outidx;
        }
    }

    out->resize(outidx);
}

void OctomapDrawer::render()
{
    octomap::ColorOcTree::tree_iterator it = mOctomap.begin_tree();
    octomap::ColorOcTree::tree_iterator end = mOctomap.end_tree();
    int counter = 0;
    double occThresh = 0.9;
    int level = 16;

    if (occThresh > 0) {
        glDisable(GL_LIGHTING);
        glEnable(GL_BLEND);
        glBegin(GL_TRIANGLES);
        double stretch_factor = 128 / (1 - occThresh);

        for (; it != end; ++counter, ++it) {
            if (level != it.getDepth()) {
                continue;
            }
            double occ = it->getOccupancy();
            if (occ < occThresh)
                continue;

            glColor4ub(it->getColor().r, it->getColor().g, it->getColor().b, 128 /*basic visibility*/ + (occ - occThresh) * stretch_factor);
            float halfsize = it.getSize() / 2.0;
            float x = it.getX();
            float y = it.getY();
            float z = it.getZ();
            //Front
            glVertex3f(x - halfsize, y - halfsize, z - halfsize);
            glVertex3f(x - halfsize, y + halfsize, z - halfsize);
            glVertex3f(x + halfsize, y + halfsize, z - halfsize);

            glVertex3f(x - halfsize, y - halfsize, z - halfsize);
            glVertex3f(x + halfsize, y + halfsize, z - halfsize);
            glVertex3f(x + halfsize, y - halfsize, z - halfsize);

            //Back
            glVertex3f(x - halfsize, y - halfsize, z + halfsize);
            glVertex3f(x + halfsize, y - halfsize, z + halfsize);
            glVertex3f(x + halfsize, y + halfsize, z + halfsize);

            glVertex3f(x - halfsize, y - halfsize, z + halfsize);
            glVertex3f(x + halfsize, y + halfsize, z + halfsize);
            glVertex3f(x - halfsize, y + halfsize, z + halfsize);

            //Left
            glVertex3f(x - halfsize, y - halfsize, z - halfsize);
            glVertex3f(x - halfsize, y - halfsize, z + halfsize);
            glVertex3f(x - halfsize, y + halfsize, z + halfsize);

            glVertex3f(x - halfsize, y - halfsize, z - halfsize);
            glVertex3f(x - halfsize, y + halfsize, z + halfsize);
            glVertex3f(x - halfsize, y + halfsize, z - halfsize);

            //Right
            glVertex3f(x + halfsize, y - halfsize, z - halfsize);
            glVertex3f(x + halfsize, y + halfsize, z - halfsize);
            glVertex3f(x + halfsize, y + halfsize, z + halfsize);

            glVertex3f(x + halfsize, y - halfsize, z - halfsize);
            glVertex3f(x + halfsize, y + halfsize, z + halfsize);
            glVertex3f(x + halfsize, y - halfsize, z + halfsize);

            //?
            glVertex3f(x - halfsize, y - halfsize, z - halfsize);
            glVertex3f(x + halfsize, y - halfsize, z - halfsize);
            glVertex3f(x + halfsize, y - halfsize, z + halfsize);

            glVertex3f(x - halfsize, y - halfsize, z - halfsize);
            glVertex3f(x + halfsize, y - halfsize, z + halfsize);
            glVertex3f(x - halfsize, y - halfsize, z + halfsize);

            //?
            glVertex3f(x - halfsize, y + halfsize, z - halfsize);
            glVertex3f(x - halfsize, y + halfsize, z + halfsize);
            glVertex3f(x + halfsize, y + halfsize, z + halfsize);

            glVertex3f(x - halfsize, y + halfsize, z - halfsize);
            glVertex3f(x + halfsize, y + halfsize, z + halfsize);
            glVertex3f(x + halfsize, y + halfsize, z - halfsize);
        }
        glEnd();
    }
}
