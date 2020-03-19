//
// Created by reverse-proxy on 13‏/2‏/2020.
//

#ifndef KERNELO_GMMESTIMATOR_H
#define KERNELO_GMMESTIMATOR_H

#include "Estimators.h"
#include <gtest/gtest_prod.h>

namespace learningModel{

    class GmmEstimator: public Iestimator<FullCovariance, FullCovariance>{

    public:
        explicit GmmEstimator(const std::shared_ptr<GMMLearningConfig>& config);
        GmmEstimator();

        void execute(
                const mat& x,
                const mat& y,
                std::shared_ptr<GLLiMParameters<FullCovariance, FullCovariance>> initial_theta) override ;

        mat getPosterior();

        void train(
                const mat &data,
                const vec& weights,
                const mat &means,
                const cube &covariances);

    private:
        vec Rou;
        mat M;
        cube V;
        mat posterior;
        std::shared_ptr<GMMLearningConfig> config;

        FRIEND_TEST(GmmEstimatorTest, toGMM);
        FRIEND_TEST(GmmEstimatorTest, fromGMM);

        GLLiMParameters<FullCovariance, FullCovariance> fromGMM(int K, int D, int L);
        void toGMM(const std::shared_ptr<GLLiMParameters<FullCovariance, FullCovariance>>& theta);
    };
}

#include "GmmEstimator.tpp"

#endif //KERNELO_GMMESTIMATOR_H
