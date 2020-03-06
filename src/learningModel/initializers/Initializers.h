//
// Created by reverse-proxy on 13‏/2‏/2020.
//

#ifndef KERNELO_INITIALIZERS_H
#define KERNELO_INITIALIZERS_H

#include <armadillo>
#include "../covariances/Icovariance.h"
#include "../gllim/GLLiMParameters.h"
#include <memory>

namespace learningModel{

    template <typename T, typename U >
    class Iinitilizer{

        static_assert(std::is_base_of<Icovariance, T>(), "Type T must be Icovariance specialization");
        static_assert(std::is_base_of<Icovariance, U>(), "Type U must be Icovariance specialization");

    public:
        virtual std::shared_ptr<GLLiMParameters<T, U>> execute(const mat &x, const mat &y, unsigned nb_gaussians) = 0;
    };
}

#endif //KERNELO_INITIALIZERS_H
