/**
 * @file GaussianRegularizedProposition.h
 * @brief GaussianRegularizedProposition class definition
 * @author Sami DJOUADI
 * @version 1.2
 * @date 29/03/2020
 */

#ifndef KERNELO_GAUSSIANREGULARIZEDPROPOSITION_H
#define KERNELO_GAUSSIANREGULARIZEDPROPOSITION_H

#include "ISProposition.h"

namespace importanceSampling {
    /**
     * @class GaussianRegularizedProposition
     */
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
