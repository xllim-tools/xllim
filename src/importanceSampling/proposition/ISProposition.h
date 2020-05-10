/**
 * @file ISProposition.h
 * @brief abstract class definition of the proposition law
 * @author Sami DJOUADI
 * @version 1.2
 * @date 29/03/2020
 */

#ifndef KERNELO_ISPROPOSITION_H
#define KERNELO_ISPROPOSITION_H

#include <armadillo>

using namespace arma;

namespace importanceSampling{
    /**
     * @class ISProposition
     * @details This is an abstract class that represents the proposition law used in the importance sampling.
     */
    class ISProposition{
    public:
        /**
         * This method generates a sample according to the proposition law
         * @return a vector of variables of L dimension
         */
        virtual vec sample() = 0;

        /**
         * This method computes the logarithm of the density of the proposition law
         * @param x_sample
         * @return value of the log density
         */
        virtual double proposition_log_density(vec x_sample) = 0;
        virtual mat proposition_covariance() = 0;


        unsigned getDimension(){
            return L;
        }

    protected :
        unsigned L;
    };
}

#endif //KERNELO_ISPROPOSITION_H
