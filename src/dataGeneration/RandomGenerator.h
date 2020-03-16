/**
 * @file RandomGenerator.h
 * @brief RandomGenerator class definition
 * @author Sami DJOUADI
 * @version 1.0
 * @date 13/01/2020
 */

#ifndef KERNELO_RANDOMGENERATOR_H
#define KERNELO_RANDOMGENERATOR_H

#include "GeneratorStrategy.h"
#include <armadillo>

using namespace arma;

namespace DataGeneration{

    /**
     * @brief A data generator using Mersenne Twister engine
     *
     * @details this concrete strategy uses Mersenne Twister engine to generate data while following
     * the base strategy interface.
     */
    class RandomGenerator : public GeneratorStrategy {
    public:
        void execute(mat &x) final;
        explicit RandomGenerator(unsigned seed);

    private:
        unsigned seed;
    };
}

#endif //KERNELO_RANDOMGENERATOR_H
