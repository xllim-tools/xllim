//
// Created by reverse-proxy on 13‏/1‏/2020.
//

#ifndef KERNELO_GENERATORSTRATEGY_H
#define KERNELO_GENERATORSTRATEGY_H

#include <armadillo>

using namespace arma;

namespace DataGeneration{
    class GeneratorStrategy{
    public:
        virtual void execute(mat &x, unsigned seed) = 0;
    };
}

#endif //KERNELO_GENERATORSTRATEGY_H