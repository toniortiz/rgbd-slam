#ifndef DATASETTUMRGBD_H
#define DATASETTUMRGBD_H

#include "Dataset.h"
#include <fstream>
#include <vector>

class RGBDcamera;
class Extractor;

class DatasetTUM : public Dataset {
public:
    DatasetTUM();
    virtual ~DatasetTUM();

    bool isOpened() const override;
    std::shared_ptr<Frame> grabFrame(std::shared_ptr<Extractor> pExtractor, size_t i) override;
    bool open(const std::string& dataset) override;
    size_t size() const override;

    bool detectCamera();

    void print(std::ostream& out, const std::string& text) const override;
    friend std::ostream& operator<<(std::ostream& out, const DatasetTUM& dataset);

protected:
    std::string mBaseDir;
    std::shared_ptr<RGBDcamera> mpCamera;
    std::ifstream mAssociationFile;

    std::vector<std::string> mvImageFilenamesRGB;
    std::vector<std::string> mvImageFilenamesD;
    std::vector<double> mvTimestamps;
};

#endif // DATASETTUMRGBD_H
