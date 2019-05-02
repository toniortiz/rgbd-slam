#ifndef EXTRACTORNODE_H
#define EXTRACTORNODE_H

#include <list>
#include <opencv2/features2d.hpp>

class ExtractorNode {
public:
    ExtractorNode()
        : bNoMore(false)
    {
    }

    void divideNode(ExtractorNode& n1, ExtractorNode& n2, ExtractorNode& n3, ExtractorNode& n4);

    std::vector<cv::KeyPoint> vKeys;
    cv::Point2i UL, UR, BL, BR;
    std::list<ExtractorNode>::iterator lit;
    bool bNoMore;
};

#endif // EXTRACTORNODE_H
