//
// Created by reverse-proxy on 13‏/2‏/2020.
//

#ifndef KERNELO_INITIALIZERS_H
#define KERNELO_INITIALIZERS_H

#include <armadillo>
#include "Icovariance.h"
#include "GLLiMParameters.h"

namespace learningModel{

    template <typename T, typename U >
    class Iinitilizer{

        static_assert(!std::is_base_of<Icovariance, T>(), "Type T must be Icovariance specialization");
        static_assert(!std::is_base_of<Icovariance, T>(), "Type U must be Icovariance specialization");

    public:
        virtual void initialize(mat x, mat y, GLLiMParameters<T,U> &theta) = 0;
    };
}

#endif //KERNELO_INITIALIZERS_H
