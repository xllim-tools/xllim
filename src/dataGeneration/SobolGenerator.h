//
// Created by reverse-proxy on 13‏/1‏/2020.
//

#ifndef KERNELO_SOBOLGENERATOR_H
#define KERNELO_SOBOLGENERATOR_H

#include "GeneratorStrategy.h"
#include <armadillo>

using namespace arma;

namespace DataGeneration{
    class SobolGenerator : public GeneratorStrategy {
    public:
        void execute(int n, int dimension, double *x) final ;
    };
}



#endif //KERNELO_SOBOLGENERATOR_H
