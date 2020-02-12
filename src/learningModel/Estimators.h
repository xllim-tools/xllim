//
// Created by reverse-proxy on 12‏/2‏/2020.
//

#ifndef KERNELO_ESTIMATORS_H
#define KERNELO_ESTIMATORS_H

#include <armadillo>
#include "Icovariance.h"
#include "GLLiMParameters.h"
#include "LearningConfig.h"

namespace learningModel{

    template <typename T = Icovariance, typename U = Icovariance>
    class Iestimator{
    public:
        virtual GLLiMParameters<T,U> estimate(mat x, mat y, GLLiMParameters<T,U> initial_theta) = 0;
    };

    template <typename T, typename U>
    class EmEstimator : public Iestimator<T,U>{
    public:
        explicit EmEstimator(EMLearningConfig config);
        void next_rnk(mat x, mat y, GLLiMParameters<T,U> theta, mat &next_rnk);
        void next_theta(mat x, mat y, mat rnk, GLLiMParameters<T,U> &next_theta);
    };

    template <typename T, typename U>
    class GmmEstimator: public Iestimator<T,U>{
    public:
        explicit GmmEstimator(GMMLearningConfig config);
        GmmEstimator(vec Rou_k, Col<vec> M_k, Col<mat> V_k);
        mat getPosterior();
        void train(mat x, int nb_iteration);

    private:
        vec Rou_k;
        Col<vec> M_k;
        Col<mat> V_k;
        mat posterior;

        GLLiMParameters<T,U> fromGMM();
        void toGMM(GLLiMParameters<T,U> theta);
    };

}

#endif //KERNELO_ESTIMATORS_H
