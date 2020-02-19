//
// Created by reverse-proxy on 12‏/2‏/2020.
//

#ifndef KERNELO_GLLIMPARAMETERS_H
#define KERNELO_GLLIMPARAMETERS_H

#include <armadillo>
#include "Icovariance.h"

using namespace arma;

namespace learningModel{

    template <typename T, typename U>
    class GLLiMParameters {

    public:
        GLLiMParameters(
                const vec& Pi,
                const mat &C,
                const std::vector<T>& Gamma,
                const cube &A,
                const mat& B,
                const std::vector<U>& Sigma){

            this->Pi = Pi;
            this->C = C;
            this->Gamma = Gamma;
            this->A = A;
            this->B = B;
            this->Sigma = Sigma;
        }

        GLLiMParameters() = default;

        vec Pi;
        mat C;
        std::vector<T> Gamma;
        cube A;
        mat B;
        std::vector<U> Sigma;
    };


}



#endif //KERNELO_GLLIMPARAMETERS_H
