/**
 * @file FunctionnalModel.h
 * @brief Functional model pure abstract class definition
 * @author Sami DJOUADI
 * @version 1.0
 * @date 25/12/2019
 */

#ifndef UNTITLED_FUNCTIONNALMODEL_H
#define UNTITLED_FUNCTIONNALMODEL_H

#include <armadillo>
#include <utility>
#include <memory>

using namespace arma;

/**
 * @class FunctionnalModel
 * @brief Pure abstract class representing the functional model
 *
 * This class is an interface of the functional model. It offers the
 * functional method "F" in different signatures. It contains normalisation
 * and problem dimension methods.
 *
 */
class FunctionnalModel{
public:

    /**
     * This method calculates F(x,y) where x is a vector and return the results
     * in y vector
     * @param x
     * @param y vector of results
     */
    virtual void F(rowvec x, rowvec &y) = 0;

    /**
     * This method calculates F(x) where x is a vector and return the results
     * as vector
     * @param x
     * @return vector of results
     */
    virtual void F(double *x, int size_x, double *y, int size_y) {
        rowvec x_arma = rowvec(&x[0],size_x);
        rowvec y_arma = rowvec(size_y);
        F(x_arma,y_arma);
        for(unsigned i=0 ; i<size_y; i++){
            y[i] = y_arma(i);
        }
    }

    /**
     * This method calculates F(x) where x is a matrix and return the results
     * as matrix
     * @param x
     * @return matrix of results
     */
    virtual void F(double *x, int x_row_size, int x_col_size, double *y, int y_row_size, int y_col_size) = 0;

    /**
     * This method returns the D dimension of the problem
     * @return thd D dimension of the problem
     */
    virtual int get_D_dimension() = 0;

    /**
     * This method returns the L dimension of the problem
     * @return thd L dimension of the problem
     */
    virtual int get_L_dimension() = 0;

    /**
     * his method normalizes the vector
     * @param x the vector to normalize
     */
    virtual void to_physic(double *x, int size) = 0;

    /**
     * This method denormalizes the vector
     * @param x the vector to denormalize
     */
    virtual void from_physic(double *x, int size) = 0;
};

#endif //UNTITLED_FUNCTIONNALMODEL_H
