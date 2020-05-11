/**
 * @file Estimators.h
 * @brief Estimator interface
 * @author Sami DJOUADI
 * @version 1.1
 * @date 12/02/2020
 */

#ifndef KERNELO_ESTIMATORS_H
#define KERNELO_ESTIMATORS_H

#include <armadillo>
#include "../covariances/Icovariance.h"
#include "../gllim/GLLiMParameters.h"
#include "../configs/LearningConfig.h"
#include <memory>

namespace learningModel{

    /**
     * @class Iestimator
     * @brief This is the interface of an estimator of the GLLiM model.
     * @tparam T : the type of Gamma matrices must be a specialisation of @see Icovariance Icovariance.
     * @tparam U : the type of Sigma matrices must be a specialisation of @see Icovariance Icovariance.
     */
    template <typename T , typename U >
    class Iestimator{

        static_assert(std::is_base_of<Icovariance, T>(), "Type T must be Icovariance specialization");
        static_assert(std::is_base_of<Icovariance, U>(), "Type U must be Icovariance specialization");

    public:
        /**
         * @brief The method trains a GLLiM model.
         * @param x : a matrix of low dimension data
         * @param y : a matrix of high dimension data
         * @param initial_theta : is the set of initialized parameters of the GLLiM model (@see GLLiMParameters GLLiMParameters)
         */
        virtual void execute(const mat& x, const mat& y, std::shared_ptr<GLLiMParameters<T,U>> initial_theta) = 0;

    };
}

#endif //KERNELO_ESTIMATORS_H
