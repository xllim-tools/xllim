//
// Created by reverse-proxy on 12‏/2‏/2020.
//

#ifndef KERNELO_ESTIMATORS_H
#define KERNELO_ESTIMATORS_H

#include <armadillo>
#include "../Icovariance.h"
#include "../GLLiMParameters.h"
#include "../LearningConfig.h"
#include <memory>

namespace learningModel{

    template <typename T , typename U >
    class Iestimator{

        static_assert(std::is_base_of<Icovariance, T>(), "Type T must be Icovariance specialization");
        static_assert(std::is_base_of<Icovariance, U>(), "Type U must be Icovariance specialization");

    public:
        virtual void estimate(const mat& x, const mat& y, std::shared_ptr<GLLiMParameters<T,U>> initial_theta) = 0;

    };
}

#endif //KERNELO_ESTIMATORS_H
