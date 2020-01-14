//
// Created by reverse-proxy on 13‏/1‏/2020.
//

#ifndef KERNELO_SOBOLGENERATOR_H
#define KERNELO_SOBOLGENERATOR_H

#include "GeneratorStrategy.h"
#include <armadillo>

using namespace arma;

namespace DataGeneration{
    class SobolGenerator : GeneratorStrategy {
    public:
        static void execute(int n, int dimension, mat &x);
    };
}



#endif //KERNELO_SOBOLGENERATOR_H
