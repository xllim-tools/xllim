/**
 * @file StatModel.h
 * @brief Statistic model abstract class definition
 * @author Sami DJOUADI
 * @version 1.0
 * @date 06/01/2020
 */

#ifndef KERNELO_DATAGENERATOR_H
#define KERNELO_DATAGENERATOR_H


#include <armadillo>
#include "../physicalModel/FunctionalModel.h"

using namespace arma;
using namespace Functional;

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
         * This method generates a complete learning data set and returns the n generated values in X and Y.
         * This method should be exposed for any integration with another front-end language. It is a wrapper
         * method of the version using armadillo's data structures.
         *
         * @param functionalModel : is used to calculate Y and to define the problem dimensions
         * @param n : number of rows in the dat set
         * @param x : generated values (like photometries in the context of space remote sensing)
         * @param y : calculated values with the functional model and using the generated values in X
         *            (like reflectances in the context of space remote sensing)
         */
        virtual void gen_data(int n, double *x, int x_dimension, double *y, int y_dimension) {
            std::tuple<mat, mat> data = gen_data(n);

            for(unsigned i=0 ; i<n ; i++){
                for(unsigned j=0 ; j<x_dimension; j++){
                    x[i*x_dimension+j] = std::get<0>(data)(i,j);
                }

                for(unsigned j=0 ; j<y_dimension; j++){
                    y[i*y_dimension+j] = std::get<1>(data)(i,j);
                }
            }
        };

        /**
         * This method enerates a complete learning data set and returns a pair of X (generated data) and
         * Y (calculated data using the functional model). It uses armadillo's data structures.
         * @param functionalModel is used to calculate Y and to define the problem dimensions
         * @param n : number of rows in the dat set
         * @return A pair of X (generated data) and Y (calculated data using the functional model)
         */
        virtual std::tuple<mat, mat> gen_data(int n) = 0;
        virtual double density_X_Y(mat x, mat y) = 0;
    };
}

#endif //KERNELO_DATAGENERATOR_H
