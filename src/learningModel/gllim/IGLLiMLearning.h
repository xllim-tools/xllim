//
// Created by reverse-proxy on 13‏/2‏/2020.
//

#ifndef KERNELO_IGLLIMLEARNING_H
#define KERNELO_IGLLIMLEARNING_H

#include <armadillo>
#include <memory>
#include "GLLiM.h"
#include "../covariances/Icovariance.h"
#include "GLLiMParameters.h"

using namespace arma;

namespace learningModel{

    class IGLLiMLearning{
    public:
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
        virtual void getModel(GLLiM &gllim) = 0;
        virtual void setModel(GLLiM &gllim) = 0;
        virtual void getInverse(GLLiM &gllim) = 0;
        virtual void directLogDensity(double *x, double *weights, double *means, double *covs) = 0;
        virtual void inverseLogDensity(double *y, double *weights, double *means, double *covs) = 0;
        virtual void train(const mat &x, const mat &y) = 0;
        virtual void initialize(const mat &x, const mat &y) = 0;
        virtual arma::gmm_full computeGMM(const vec &y_obs, const vec &cov_obs) = 0;
    };

}

#endif //KERNELO_IGLLIMLEARNING_H
