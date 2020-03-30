//
// Created by reverse-proxy on 29‏/3‏/2020.
//

#ifndef KERNELO_ISPROPOSITION_H
#define KERNELO_ISPROPOSITION_H

#include <armadillo>

using namespace arma;

namespace importanceSampling{
    class ISProposition{
    public:
        virtual vec sample(unsigned L) = 0; // return a sample using the proposition law
        virtual double proposition_log_density(vec x_sample) = 0;
        virtual mat proposition_covariance() = 0;
    };
}

#endif //KERNELO_ISPROPOSITION_H
