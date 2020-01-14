//
// Created by reverse-proxy on 13‏/1‏/2020.
//

#ifndef KERNELO_RANDOMGENERATOR_H
#define KERNELO_RANDOMGENERATOR_H

#include "GeneratorStrategy.h"
#include <armadillo>

using namespace arma;

namespace DataGeneration{
    class RandomGenerator : GeneratorStrategy {
        static void execute(int n, int dimension, mat &x);
    };
}

#endif //KERNELO_RANDOMGENERATOR_H
