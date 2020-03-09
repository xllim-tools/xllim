//
// Created by reverse-proxy on 12‏/2‏/2020.
//

#ifndef KERNELO_GLLIMPARAMETERS_H
#define KERNELO_GLLIMPARAMETERS_H

#include <armadillo>
#include "../covariances/Icovariance.h"

using namespace arma;

namespace learningModel{

    template <typename T, typename U>
    class GLLiMParameters {

        static_assert(std::is_base_of<Icovariance, T>(), "Type T must be Icovariance specialization");
        static_assert(std::is_base_of<Icovariance, U>(), "Type U must be Icovariance specialization");

    public:

        GLLiMParameters(unsigned D, unsigned L, unsigned K){
            this->D = D;
            this->L = L;
            this->K = K;
            this->Pi = vec(K, fill::zeros);
            this->Gamma = std::vector<T>(K);
            this->Sigma = std::vector<U>(K);

            for(unsigned k=0; k<K; k++){
                this->Gamma[k] = T(L);
                this->Sigma[k] = U(D);
            }

            this->C = mat(L, K,fill::zeros);
            this->B = mat(D, K, fill::zeros);
            this->A = cube(D,L,K,fill::zeros);
        }

        GLLiMParameters(const GLLiMParameters &gllimParams){
            this->D = gllimParams.D;
            this->L = gllimParams.L;
            this->K = gllimParams.K;
            this->Pi = gllimParams.Pi;
            this->Gamma = gllimParams.Gamma;
            this->Sigma = gllimParams.Sigma;
            this->C = gllimParams.C;
            this->B = gllimParams.B;
            this->A = gllimParams.A;
        }

        vec Pi;
        mat C;
        std::vector<T> Gamma;
        cube A;
        mat B;
        std::vector<U> Sigma;
        unsigned K;
        unsigned L;
        unsigned D;


    };


}



#endif //KERNELO_GLLIMPARAMETERS_H
