//
// Created by reverse-proxy on 13‏/2‏/2020.
//

#ifndef KERNELO_GLLIMLEARNING_H
#define KERNELO_GLLIMLEARNING_H

#include "IGLLiMLearning.h"
#include "../initializers/Initializers.h"
#include "../estimators/Estimators.h"
#include <memory>

namespace learningModel{
    template <typename T, typename U >
    class GLLiMLearning : public IGLLiMLearning {
    public:
        GLLiMLearning(std::shared_ptr<Iinitilizer<T,U>> initializer, std::shared_ptr<Iestimator<T,U>> estimator, unsigned gaussians);
        void train(const mat &x, const mat &y) override;
        void initialize(const mat &x, const mat &y) override;
        void exportModel(GLLiM &gllim) override ;
        void importModel(GLLiM &gllim) override ;
        GLLiMParameters<FullCovariance, FullCovariance> inverse(GLLiMParameters<T,U> &gllim_direct);
        arma::gmm_full computeGMM(const vec &y_obs, const vec &cov_obs) override ;

        static_assert(std::is_base_of<Icovariance, T>(), "Type T must be Icovariance specialization");
        static_assert(std::is_base_of<Icovariance, T>(), "Type U must be Icovariance specialization");

    private:
        std::shared_ptr<Iinitilizer<T,U>> initializer;
        std::shared_ptr<Iestimator<T,U>> estimator;
        std::shared_ptr<GLLiMParameters<T,U>> gllim_parameters;
        unsigned nb_gaussians;
    };
}



#endif //KERNELO_GLLIMLEARNING_H
