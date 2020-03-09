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
            mat x_arma = mat(&x[0], x_rows, x_cols);
            mat y_arma = mat(&y[0], y_rows, y_cols);

            train(x_arma, y_arma);
        }

        void initialize(double *x, int x_rows, int x_cols, double *y, int y_rows, int y_cols){
            mat x_arma = mat(&x[0], x_rows, x_cols);
            mat y_arma = mat(&y[0], y_rows, y_cols);

            initialize(x_arma, y_arma);
        }

        virtual void train(const mat &x, const mat &y) = 0;
        virtual void initialize(const mat &x, const mat &y) = 0;
        virtual void exportModel(GLLiM &gllim) = 0;
        virtual void importModel(GLLiM &gllim) = 0;
        virtual arma::gmm_full computeGMM(const vec &y_obs, const vec &cov_obs) = 0;

    };

}

#endif //KERNELO_IGLLIMLEARNING_H
