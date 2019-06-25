#include "Random.h"

using namespace std;

bool Random::SET_RAND = false;

void Random::initSeed()
{
    if (!SET_RAND) {
        srand((unsigned int)time(NULL));
        SET_RAND = true;
    }
}

int Random::randomInt(int min, int max)
{
    initSeed();
    int d = max - min + 1;
    return int(((double)rand() / ((double)RAND_MAX + 1.0)) * d) + min;
}
