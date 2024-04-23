#ifndef KERNELO_DATAGENERATOR_H
#define KERNELO_DATAGENERATOR_H

#include <armadillo>
#include "../../functionalModel/FunctionalModel.hpp"

using namespace arma;

namespace DataGeneration{
    /**
     * @class StatModel
     * @brief Abstract class representing the statistical model used in data generation component
     *
     * @details This is an interface of statistical models used for data generation and density calculation.
     * It contains two versions of data generation method ; The first one is exposed to the front-end,
     * and its signature has standard C++ types. On the other hand, the second version is used internally
     * and shouldn't be exposed because it uses armadillo's data structures.
     */
    class StatModel{
    public:
        /**
         * This method generates a complete learning data set and returns a pair of X (generated data) and
         * Y (calculated data using the functional model). It uses armadillo's data structures.
         * @param functionalModel is used to calculate Y and to define the problem dimensions
         * @param n : number of rows in the dat set
         * @return A pair of X (generated data) and Y (calculated data using the functional model)
         */
        virtual std::tuple<mat, mat> gen_data(unsigned int n) = 0;

        /**
         * This method computes the logarithm of the direct conditional density.
         * @param x : low dimensional tuple that is used to compute F(x) that stands for the mean of the gaussian low.
         * @param y : high dimensional tuple that its density is computed.
         * @param y_cov : the variance of the variables of y.
         * @return double : the logarithm of the direct conditional density.
         */
        // virtual double density_X_Y(const vec &x, const vec &y, const vec &y_cov) = 0;
    };
}

#endif //KERNELO_DATAGENERATOR_H
