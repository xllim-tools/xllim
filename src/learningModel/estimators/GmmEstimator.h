//
// Created by reverse-proxy on 13‏/2‏/2020.
//

#ifndef KERNELO_GMMESTIMATOR_H
#define KERNELO_GMMESTIMATOR_H

#include "Estimators.h"

namespace learningModel{

    class GmmEstimator: public Iestimator<FullCovariance, FullCovariance>{

    public:
        explicit GmmEstimator(const std::shared_ptr<GMMLearningConfig>& config);
        GmmEstimator() = default;

        void estimate(
                const mat& x,
                const mat& y,
                std::shared_ptr<GLLiMParameters<FullCovariance, FullCovariance>> initial_theta) override ;
        mat getPosterior();
        void train(mat x, int nb_iteration);

    private:
        vec Rou;
        mat M;
        cube V;
        mat posterior;
        std::shared_ptr<GMMLearningConfig> config;

        GLLiMParameters<FullCovariance, FullCovariance> fromGMM(int K, int D, int L);
        void toGMM(std::shared_ptr<GLLiMParameters<FullCovariance, FullCovariance>> theta);
    };
}

#include "GmmEstimator.tpp"

#endif //KERNELO_GMMESTIMATOR_H
