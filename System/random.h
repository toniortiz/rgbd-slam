#ifndef RANDOM_H
#define RANDOM_H

#include <random>

class Random {
public:
    static void initSeed();

    // Returns a random int in the range [min..max]
    static int randomInt(int min, int max);

    // Returns a random number in the range [0..1]
    template <class T>
    static T randomValue()
    {
        initSeed();
        return (T)rand() / (T)RAND_MAX;
    }

    // Returns a random number in the range [min..max]
    template <class T>
    static T randomValue(T min, T max)
    {
        initSeed();
        return randomValue<T>() * (max - min) + min;
    }

    // Returns a random number from a guassian distribution
    template <class T>
    static T randomGaussianValue(T mean = 0., T sigma = 1.)
    {
        initSeed();
        T x1, x2, w, y1;

        do {
            x1 = (T)2. * randomValue<T>() - (T)1.;
            x2 = (T)2. * randomValue<T>() - (T)1.;
            w = x1 * x1 + x2 * x2;
        } while (w >= (T)1. || w == (T)0.);

        w = sqrt(((T)-2.0 * log(w)) / w);
        y1 = x1 * w;

        return (mean + y1 * sigma);
    }

protected:
    static bool SET_RAND;
};

#endif // RANDOM_H
