/**
 * @file Iinitilizer.h
 * @brief Iinitilizer class defenition
 * @author Sami DJOUADI
 * @version 1.1
 * @date 13/02/2020
 */

#ifndef KERNELO_INITIALIZERS_H
#define KERNELO_INITIALIZERS_H

#include <armadillo>
#include "../covariances/Icovariance.h"
#include "../gllim/GLLiMParameters.h"
#include <memory>

namespace learningModel{

    /**
     * @class Iinitilizer
     * @brief This is the interface of an initializer of GLLiM model.
     * @tparam T : the type of Gamma matrices must be a specialisation of @see Icovariance Icovariance.
     * @tparam U : the type of Sigma matrices must be a specialisation of @see Icovariance Icovariance.
     */
    template <typename T, typename U >
    class Iinitilizer{

        static_assert(std::is_base_of<Icovariance, T>(), "Type T must be Icovariance specialization");
        static_assert(std::is_base_of<Icovariance, U>(), "Type U must be Icovariance specialization");

    public:
        /**
         * @brief The method initializes a GLLiM model.
         * @param x : a matrix of low dimension data
         * @param y : a matrix of high dimension data
         * @param K : number of gaussians
         * @return std::shared_ptr<GLLiMParameters<T, U>>
         */
        virtual std::shared_ptr<GLLiMParameters<T, U>> execute(const mat &x, const mat &y, unsigned K) = 0;
    };
}

#endif //KERNELO_INITIALIZERS_H
