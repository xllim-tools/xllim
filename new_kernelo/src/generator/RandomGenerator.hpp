#ifndef KERNELO_RANDOMGENERATOR_H
#define KERNELO_RANDOMGENERATOR_H

#include "Generator.hpp"

using namespace arma;

namespace DataGeneration{

    /**
     * @brief A data generator using Mersenne Twister engine
     *
     * @details this concrete strategy uses Mersenne Twister engine to generate data while following
     * the base strategy interface.
     */
    class RandomGenerator : public Generator {
    public:
        RandomGenerator(unsigned int seed);
        void execute(mat &x) final;

    private:
        unsigned int seed;
    };
}

#endif //KERNELO_RANDOMGENERATOR_H
