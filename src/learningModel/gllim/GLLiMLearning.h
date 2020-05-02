//
// Created by reverse-proxy on 13‏/2‏/2020.
//

#ifndef KERNELO_GLLIMLEARNING_H
#define KERNELO_GLLIMLEARNING_H

#include "IGLLiMLearning.h"
#include "../initializers/Initializers.h"
#include "../estimators/Estimators.h"
#include <memory>
#include <gtest/gtest.h>

namespace learningModel{
    template <typename T, typename U >
    class GLLiMLearning : public IGLLiMLearning {
    public:
        GLLiMLearning(std::shared_ptr<Iinitilizer<T,U>> initializer, std::shared_ptr<Iestimator<T,U>> estimator, unsigned K);
        void train(const mat &x, const mat &y) override;
        void initialize(const mat &x, const mat &y) override;
        void getModel(GLLiM &gllim) override ;
        void setModel(GLLiM &gllim) override ;
        GLLiMParameters<FullCovariance, FullCovariance> inverse(GLLiMParameters<T,U> &gllim_direct);
        arma::gmm_full computeGMM(const vec &y_obs, const vec &cov_obs) override ;

        void getInverse(GLLiM &gllim);
        void directLogDensity(double *x, double *weights, double *means, double *covs);
        void inverseLogDensity(double *y, double *weights, double *means, double *covs);

        std::shared_ptr<GLLiMParameters<T,U>> getParameters(){
            return gllim_parameters;
        }

        static_assert(std::is_base_of<Icovariance, T>(), "Type T must be Icovariance specialization");
        static_assert(std::is_base_of<Icovariance, T>(), "Type U must be Icovariance specialization");

        template <typename V, typename W>
        arma::gmm_full logDensity(std::shared_ptr<GLLiMParameters<V,W>> gllim, const vec &x);

    private:
        std::shared_ptr<Iinitilizer<T,U>> initializer;
        std::shared_ptr<Iestimator<T,U>> estimator;
        std::shared_ptr<GLLiMParameters<T,U>> gllim_parameters;
        std::shared_ptr<GLLiMParameters<FullCovariance, FullCovariance>> inverse_gllim_parameters;

        unsigned K;

    protected:
        void alterCovariance(GLLiMParameters<T, U> &gllim, const vec &cov_obs);

    };

}

#include "GLLiMLearning.tpp"



#endif //KERNELO_GLLIMLEARNING_H
