/**
 * @file IGLLiMLearning.h
 * @brief IGLLiMLearning interface definition
 * @author Sami DJOUADI
 * @version 1.1
 * @date 13/02/2020
 */

#ifndef KERNELO_IGLLIMLEARNING_H
#define KERNELO_IGLLIMLEARNING_H

#include <armadillo>
#include <memory>
#include "GLLiM.h"
#include "../covariances/Icovariance.h"
#include "GLLiMParameters.h"

using namespace arma;

namespace learningModel{

    /**
     * @class IGLLiMLearning
     * @details This is the interface of the GLLiM model. It implements the initialization and training methods using data
     * structures based on pointers that are called by third party language API for integration purposes.
     */
    class IGLLiMLearning{
    public:
        /**
         * @details This method can be used by third party language API to initialize the GLLiM model. It calls the internal
         * initialization method based on the data structures of the Armadillo library.
         * @param x : pointer to a matrix of low dimension data
         * @param x_rows : number of tuples of low dimension data
         * @param x_cols : size of a tuple of low dimension data
         * @param y : pointer to a matrix of high dimension data
         * @param y_rows : number of tuples of high dimension data
         * @param y_cols : size of a tuple of high dimension data
         */
        void train(double *x, int x_rows, int x_cols, double *y, int y_rows, int y_cols){
            mat x_arma(x_rows, x_cols);
            for(unsigned i=0; i<x_rows; i++){
                for(unsigned j=0; j<x_cols; j++){
                    x_arma(i,j) = x[j + i*x_cols];
                }
            }

            mat y_arma(y_rows, y_cols);
            for(unsigned i=0; i<y_rows; i++){
                for(unsigned j=0; j<y_cols; j++){
                    y_arma(i,j) = y[j + i*y_cols];
                }
            }

            train(x_arma, y_arma);
        }

        /**
         * @details This method can be used by third party language API to train the GLLiM model. It calls the internal
         * training method based on the data structures of the Armadillo library.
         * @param x : pointer to a matrix of low dimension data
         * @param x_rows : number of tuples of low dimension data
         * @param x_cols : size of a tuple of low dimension data
         * @param y : pointer to a matrix of high dimension data
         * @param y_rows : number of tuples of high dimension data
         * @param y_cols : size of a tuple of high dimension data
         */
        void initialize(double *x, int x_rows, int x_cols, double *y, int y_rows, int y_cols){
            mat x_arma(x_rows, x_cols);
            for(unsigned i=0; i<x_rows; i++){
                for(unsigned j=0; j<x_cols; j++){
                    x_arma(i,j) = x[j + i*x_cols];
                }
            }

            mat y_arma(y_rows, y_cols);
            for(unsigned i=0; i<y_rows; i++){
                for(unsigned j=0; j<y_cols; j++){
                    y_arma(i,j) = y[j + i*y_cols];
                }
            }

            initialize(x_arma, y_arma);
        }

        /**
         * @details This method exports the parameters of the GLLiM model wrapped in an object of the class @see GLLiM GLLiM
         * @param gllim : the object to fill with the parameters of the GLLiM model
         */
        virtual void getModel(GLLiM &gllim) = 0;

        /**
         * @details This method imports the parameters of the GLLiM model wrapped in an object of the class @see GLLiM GLLiM
         * @param gllim : @see GLLiM GLLiM
         */
        virtual void setModel(GLLiM &gllim) = 0;

        /**
         * @details This method imports the parameters of the inversed GLLiM model wrapped in an object of the class @see GLLiM GLLiM
         * @param gllim : @see GLLiM GLLiM
         */
        virtual void getInverse(GLLiM &gllim) = 0;

        /**
         * @brief Computation of the logarithm of the direct conditional density of the GLLiM model
         * @details See the formula 6 in Antoine Deleforge, Florence Forbes, and Radu Horaud. High-Dimensional Regression
         * with Gaussian Mixtures and Partially-Latent Response Variables. Statistics and Computing 25(5): 893-911, September 2015.
         * @param x : pointer to a vector of low dimension data
         * @param weights : pointer to a vector(K) of weights of the GMM describing the direct density of the GLLiM model
         * @param means : pointer to a matrix(D,K) of means of the GMM describing the direct density of the GLLiM model
         * @param covs : pointer to a cube(D,D,K) contaiming the covariance matrices of the GMM describing the direct density of the GLLiM model
         */
        virtual void directLogDensity(double *x, double *weights, double *means, double *covs) = 0;

        /**
         * @brief Computation of the logarithm of the inverse conditional density of the GLLiM model
         * @details See the formula 7 in Antoine Deleforge, Florence Forbes, and Radu Horaud. High-Dimensional Regression
         * with Gaussian Mixtures and Partially-Latent Response Variables. Statistics and Computing 25(5): 893-911, September 2015.
         * @param y : pointer to a vector of high dimension data
         * @param weights : pointer to a vector(K) of weights of the GMM describing the inverse density of the GLLiM model
         * @param means : pointer to a matrix(L,K) of means of the GMM describing the inverse density of the GLLiM model
         * @param covs : pointer to a cube(L,L,K) containing the covariance matrices of the GMM describing the inverse density of the GLLiM model
         */
        virtual void inverseLogDensity(double *y, double *weights, double *means, double *covs) = 0;

        /**
         * @details This method trains the GLLiM model using Armadillo data structure. Should be used for internal calls within the library.
         * @param x : a matrix of low dimension data
         * @param y : a matrix of high dimension data
         */
        virtual void train(const mat &x, const mat &y) = 0;

        /**
         * @details This method initializes the GLLiM model using Armadillo data structure. Should be used for internal calls within the library.
         * @param x : a matrix of low dimension data
         * @param y : a matrix of high dimension data
         */
        virtual void initialize(const mat &x, const mat &y) = 0;

        /**
         * @details This method compute the corresponding GMM using inverse conditional density of given observation and its measure error.
         * @param y_obs : a vector of low dimension data
         * @param cov_obs : the errors in the measure of the observation
         * @return An Armadillo's GMM data structure
         */
        virtual arma::gmm_full computeGMM(const vec &y_obs, const vec &cov_obs) = 0;
    };

}

#endif //KERNELO_IGLLIMLEARNING_H
