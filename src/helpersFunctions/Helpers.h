//
// Created by reverse-proxy on 16‏/3‏/2020.
//

#ifndef KERNELO_HELPERS_H
#define KERNELO_HELPERS_H

#include <armadillo>

using namespace arma;

namespace Helpers{

    double logSumExp(const vec &elements);
    double computeDeterminant(const mat& matrix);
    mat inverseMatrix(const mat& matrix);

}

#endif //KERNELO_HELPERS_H
