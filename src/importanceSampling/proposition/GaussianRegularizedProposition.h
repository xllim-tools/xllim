//
// Created by reverse-proxy on 29‏/3‏/2020.
//

#ifndef KERNELO_GAUSSIANREGULARIZEDPROPOSITION_H
#define KERNELO_GAUSSIANREGULARIZEDPROPOSITION_H

#include "ISProposition.h"

namespace importanceSampling {
    class GaussianRegularizedProposition : public ISProposition{
    public:

        GaussianRegularizedProposition(vec &mean, mat &cov);

        vec sample() override ; // return a sample using the proposition law
        double proposition_log_density(vec x_sample) override;
        mat proposition_covariance() override;

    private:
        vec mean;
        mat cov;
    };
}


#endif //KERNELO_GAUSSIANREGULARIZEDPROPOSITION_H
