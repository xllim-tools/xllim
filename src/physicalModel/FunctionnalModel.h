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
    virtual void F(const std::vector<double> &x, std::vector<double> &y) = 0;

    /**
     * This method calculates F(x) where x is a vector and return the results
     * as vector
     * @param x
     * @return vector of results
     */
    virtual std::vector<double> F(const std::vector<double> &x) = 0;

    /**
     * This method calculates F(x) where x is a matrix and return the results
     * as matrix
     * @param x
     * @return matrix of results
     */
    virtual std::vector<std::vector<double>> F(const std::vector<std::vector<double>> &x) = 0;

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
     * This method returns a normalized vector of x
     * @param x the vector to normalize
     * @return normalized vector
     */
    virtual std::vector<double> nomalize(std::vector<double> x) = 0;

    /**
     * This method returns a  denormalized vector of x
     * @param x the vector to denormalize
     * @return denormalized vector
     */
    virtual std::vector<double> invNormalize(std::vector<double> x) = 0;
};

#endif //UNTITLED_FUNCTIONNALMODEL_H
