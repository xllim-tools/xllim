//
// Created by reverse-proxy on 12‏/2‏/2020.
//

#ifndef KERNELO_GLLIMPARAMETERS_H
#define KERNELO_GLLIMPARAMETERS_H

#include <armadillo>
#include "Icovariance.h"

using namespace arma;

namespace learningModel{

    template <typename T = Icovariance, typename U = Icovariance>
    class GLLiMParameters {

    public:
        GLLiMParameters(
                const vec& Pi_k,
                const vec& Ci_k,
                const Col<T>& Gamma_k,
                const Col<vec>& A_k,
                const mat& B_k,
                const Col<U>& Sigma_k){

            this->Pi_k = Pi_k;
            this->Ci_k = Ci_k;
            this->Gamma_k = Gamma_k;
            this->A_k = A_k;
            this->B_k = B_k;
            this->Sigma_k = Sigma_k;
        }

        vec Pi_k;
        vec Ci_k;
        Col<T> Gamma_k;
        Col<vec> A_k;
        mat B_k;
        Col<U> Sigma_k;
    };
}



#endif //KERNELO_GLLIMPARAMETERS_H
