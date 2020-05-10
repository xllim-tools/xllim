/**
 * @file GaussianMixtureProposition.h
 * @brief GaussianMixtureProposition class definition
 * @author Sami DJOUADI
 * @version 1.2
 * @date 29/03/2020
 */

#ifndef KERNELO_GAUSSIANMIXTUREPROPOSITION_H
#define KERNELO_GAUSSIANMIXTUREPROPOSITION_H

#include "ISProposition.h"


namespace importanceSampling {
    /**
     * @class GaussianMixtureProposition
     */
    class GaussianMixtureProposition : public ISProposition{
    public:
        GaussianMixtureProposition(vec &weights, mat &means, cube &covariances);

        vec sample() override ; // return a sample using the proposition law
        double proposition_log_density(vec x_sample) override;
        mat proposition_covariance() override;

    private:
        gmm_full gmm;
    };
}


#endif //KERNELO_GAUSSIANMIXTUREPROPOSITION_H
