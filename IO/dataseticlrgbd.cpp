#include "dataseticlrgbd.h"
#include "Core/frame.h"
#include "Core/intrinsicmatrix.h"
#include "Core/rgbdcamera.h"
#include "Features/extractor.h"

using namespace std;

DatasetICL::DatasetICL()
    : Dataset("ICL")
    , mpCamera(nullptr)
{
}

DatasetICL::~DatasetICL() {}

bool DatasetICL::isOpened() const { return (mAssociationFile.is_open() && mpCamera); }

Frame::Ptr DatasetICL::grabFrame(Extractor::Ptr pExtractor, size_t i)
{
    if (isOpened()) {
        Frame::Ptr frame = mpCamera->createFrame(mBaseDir, mvImageFilenamesRGB[i], mvImageFilenamesD[i], mvTimestamps[i], pExtractor);
        return frame;
    } else
        return nullptr;
}

bool DatasetICL::open(const string& dataset)
{
    mBaseDir = dataset;
    mAssociationFile.open(mBaseDir + "associations.txt");
    if (!mAssociationFile.is_open()) {
        cerr << "Can't open association file" << endl;
        return false;
    }

    IntrinsicMatrix* pIntrinsic = new IntrinsicMatrix(481.2f, -480.0f, 319.5f, 239.5f);
    pIntrinsic->setDistortion(0, 0, 0, 0, 0);
    mpCamera = make_shared<RGBDcamera>(pIntrinsic, 40.0, 40.0, 5000.0, 30.0, 640, 480);

    while (!mAssociationFile.eof()) {
        string s;
        getline(mAssociationFile, s);
        if (!s.empty()) {
            stringstream ss;
            ss << s;
            double t;
            string sRGB, sD;
            ss >> t;
            mvTimestamps.push_back(t);
            ss >> sRGB;
            mvImageFilenamesRGB.push_back(sRGB);
            ss >> t;
            ss >> sD;
            mvImageFilenamesD.push_back(sD);
        }
    }
    return true;
}

size_t DatasetICL::size() const { return mvImageFilenamesRGB.size(); }

void DatasetICL::print(ostream& out, const string& text) const
{
    Dataset::print(out, text);
    out << "Base Dir: " << mBaseDir << endl;
    out << "Size: " << size() << endl;
    out << *mpCamera << endl;
}

ostream& operator<<(ostream& out, const DatasetICL& dataset)
{
    dataset.print(out, string(""));
    return out;
}
