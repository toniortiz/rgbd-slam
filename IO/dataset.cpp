#include "dataset.h"
#include "Core/frame.h"
#include "dataseticlrgbd.h"
#include "datasettumrgbd.h"

using namespace std;

Dataset::Dataset()
    : mName("Untitled")
{
}

Dataset::Dataset(const string& name)
    : mName(name)
{
}

Dataset::~Dataset() {}

string Dataset::name() const { return mName; }

bool Dataset::isOpened() const { return false; }

size_t Dataset::size() const { return 0; }

bool Dataset::open(const string& dataset) { return false; }

Dataset::Ptr Dataset::create(const DS& dataset)
{
    switch (dataset) {
    case TUM:
        return make_shared<DatasetTUM>();
    case ICL:
        return make_shared<DatasetICL>();
    case D:
        return nullptr;
    }

    return nullptr;
}

void Dataset::print(ostream& out, const string& text) const
{
    if (text.size() > 0)
        out << text << endl;

    out << "Name: " << mName << endl;
}

ostream& Dataset::operator<<(ostream& out)
{
    print(out, string(""));
    return out;
}
