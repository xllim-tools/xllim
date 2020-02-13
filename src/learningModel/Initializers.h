//
// Created by reverse-proxy on 13‏/2‏/2020.
//

#ifndef KERNELO_INITIALIZERS_H
#define KERNELO_INITIALIZERS_H

#include <armadillo>
#include "Icovariance.h"
#include "GLLiMParameters.h"

namespace learningModel{

    template <typename T = Icovariance, typename U = Icovariance>
    class Iinitilizer{
    public:
        virtual void initialize(mat x, mat y, GLLiMParameters<T,U> &theta) = 0;
    };
}

#endif //KERNELO_INITIALIZERS_H
