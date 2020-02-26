//
// Created by reverse-proxy on 26‏/2‏/2020.
//

#ifndef KERNELO_EMESTIMATOR_H
#define KERNELO_EMESTIMATOR_H

#include "Estimators.h"

namespace learningModel{

    template <typename T , typename U >
    class EmEstimator : public Iestimator<T,U>{
        static_assert(std::is_base_of<Icovariance, T>(), "Type T must be Icovariance specialization");
        static_assert(std::is_base_of<Icovariance, U>(), "Type U must be Icovariance specialization");

    public:
        explicit EmEstimator(const std::shared_ptr<EMLearningConfig>& config);
        void next_rnk(const mat& x, const mat& y, std::shared_ptr<GLLiMParameters<T, U>> theta, mat &next_rnk);
        void next_theta(const mat& x, const mat& y, const mat& r_nk, std::shared_ptr<GLLiMParameters<T, U>> &next_theta);

        void estimate(
                const mat& x,
                const mat& y,
                std::shared_ptr<GLLiMParameters<T, U>> initial_theta) override;

    private:
        std::shared_ptr<EMLearningConfig> config;
    };

}

#include "EmEstimator.tpp"


#endif //KERNELO_EMESTIMATOR_H
