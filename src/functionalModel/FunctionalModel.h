/**
 * @file FunctionalModel.h
 * @brief Functional model abstract class definition
 * @author Sami DJOUADI
 * @version 1.0
 * @date 25/12/2019
 */

#ifndef KERNELO_FUNCTIONNALMODEL_H
#define KERNELO_FUNCTIONNALMODEL_H

#include <armadillo>
#include <utility>
#include <memory>

using namespace arma;

namespace Functional {
/**
 * @class FunctionalModel
 * @brief Abstract class representing the functional model
 *
 * This class is an interface of the functional model. It offers the
 * functional method "F" which requires that the parameters of X be 
 * in mathematical space. It contains normalization methods to transform
 * X from and to physical space. It also allows to retrieve the dimensions 
 * of the problem.
 *
 */
    class FunctionalModel {
    public:

        /**
         * This method calculates y = F(x) using armadillo library and writes
         * results on the vector y without allocating new memory. This method
         * is used only by the other components of the kernel.
         *
         * @param x : vector of the functional model parameters. (L dimension)
         * @param y : vector of results (D dimension)
         */
        virtual void F(rowvec x, rowvec &y) = 0;

        /**
         * This method calculates y = F(x) using the method above, after adapting
         * the data structure to armadillo structure: the function copy the results
         * from armadillo structure to standard array structure.
         *
         * @param x : pointer to the set of parameters of the physical model.
         * @param size_x : number of parameters (L dimension)
         * @param y : pointer  to the set of results
         * @param size_y : number of outputs (D dimension)
         */
        virtual void F(double *x, int size_x, double *y, int size_y) {
            //create an armadillo row vector pointing to the standard array
            rowvec x_arma = rowvec(&x[0], size_x);
            rowvec y_arma = rowvec(size_y);

            F(x_arma, y_arma);

            for (unsigned i = 0; i < size_y; i++) {
                y[i] = y_arma(i);
            }
        }

        /**
         * This method calculates y = F(x) considering x and y as matrix. It iterates over
         * the matrix line by line and calls the vector version of F.
         *
         * @param x : pointer to the matrix of the functional model parameters.
         * @param x_row_size : number of sets of parameters
         * @param x_col_size : number of parameters by set (L dimension)
         * @param y : pointer to the matrix of the results
         * @param y_row_size : number of sets of results
         * @param y_col_size : number of results by set (D dimension)
         */
        virtual void F(double *x, int x_row_size, int x_col_size, double *y, int y_row_size, int y_col_size) {
            //test
        };

        /**
         * This method returns the D dimension of the problem
         * @return the dimension D of the problem
         */
        virtual int get_D_dimension() = 0;

        /**
         * This method returns the L dimension of the problem
         * @return the dimension L of the problem
         */
        virtual int get_L_dimension() = 0;

        /**
         * This method transforms the values of x from the mathematical
         * space to the physical space.
         * @param x : the vector to normalize
         */
        virtual void to_physic(rowvec &x) = 0;

        /**
         * This method transforms the values of x from the physical
         * space to the mathematical space.
         * @param x : the vector to denormalize
         */
        virtual void from_physic(double *x, int size) = 0;
    };

}

#endif //KERNELO_FUNCTIONNALMODEL_H
