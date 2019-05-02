#ifndef DATASETCORBSRGBD_H
#define DATASETCORBSRGBD_H

#include "dataset.h"
#include <fstream>
#include <vector>

class RGBDcamera;
class Extractor;

class DatasetCORBS : public Dataset {
public:
    DatasetCORBS();
    virtual ~DatasetCORBS();

    bool isOpened() const override;
    std::shared_ptr<Frame> grabFrame(std::shared_ptr<Extractor> pExtractor, size_t i) override;
    bool open(const std::string& dataset) override;
    size_t size() const override;

    void print(std::ostream& out, const std::string& text) const override;
    friend std::ostream& operator<<(std::ostream& out, const DatasetCORBS& dataset);

protected:
    std::string mBaseDir;
    std::shared_ptr<RGBDcamera> mpCamera;
    std::ifstream mAssociationFile;

    std::vector<std::string> mvImageFilenamesRGB;
    std::vector<std::string> mvImageFilenamesD;
    std::vector<double> mvTimestamps;
};

#endif // DATASETCORBSRGBD_H
