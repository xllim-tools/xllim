//
// Created by reverse-proxy on 13‏/1‏/2020.
//

#ifndef KERNELO_RANDOMGENERATOR_H
#define KERNELO_RANDOMGENERATOR_H

#include "GeneratorStrategy.h"
#include <armadillo>

using namespace arma;

namespace DataGeneration{
    class RandomGenerator : public GeneratorStrategy {
    public:
        void execute(mat &x, unsigned seed) final;
    };
}

#endif //KERNELO_RANDOMGENERATOR_H
