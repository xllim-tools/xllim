/**
 * @file GLLiMParameters.h
 * @brief GLLiMParameters class definition
 * @author Sami DJOUADI
 * @version 1.1
 * @date 12/02/2020
 */

#ifndef KERNELO_GLLIMPARAMETERS_H
#define KERNELO_GLLIMPARAMETERS_H

#include <armadillo>
#include "../covariances/Icovariance.h"
#include "../../logging/Logger.h"

using namespace arma;

namespace learningModel{
    /**
     * @class GLLiMParameters
     * @details This class wraps the parameters of the GLLiM model using data structures from armadillo library. It mutates according
     * to the type of covariance matrices of Sigma and Gamma parameters.
     * @tparam T : the type of Gamma matrices must be a specialisation of @see Icovariance Icovariance.
     * @tparam U : the type of Sigma matrices must be a specialisation of @see Icovariance Icovariance.
     */
    template <typename T, typename U>
    class GLLiMParameters {

        static_assert(std::is_base_of<Icovariance, T>(), "Type T must be Icovariance specialization");
        static_assert(std::is_base_of<Icovariance, U>(), "Type U must be Icovariance specialization");

    public:
        /**
         * Constructor
         * @param D
         * @param L
         * @param K
         */
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

        /**
         * Constructor
         * @param gllimParams : GLLiMParameters
         */
        GLLiMParameters(const GLLiMParameters &gllimParams){
            this->D = gllimParams.D;
            this->L = gllimParams.L;
            this->K = gllimParams.K;
            gllimParams.Pi.print();
            this->Pi = gllimParams.Pi;
            this->Gamma = std::vector<T>(K);
            this->Sigma = std::vector<U>(K);
            Logging::Logger::GetInstance() -> log("step A1", Logging::Logger::level(Logging::INFO));
            for(unsigned k=0; k<this->K; k++){
                this->Gamma[k] = gllimParams.Gamma[k];
                this->Sigma[k] = gllimParams.Sigma[k];
            }
            Logging::Logger::GetInstance() -> log("step A3", Logging::Logger::level(Logging::INFO));
            this->C = gllimParams.C;
            this->B = gllimParams.B;
            this->A = gllimParams.A;
            Logging::Logger::GetInstance() -> log("step A4", Logging::Logger::level(Logging::INFO));

        }
        /**
         * Assignement operator redifinition
         * @param gllimParams : GLLiMParameters
         */
        GLLiMParameters &operator=(const GLLiMParameters &gllimParams){
            Logging::Logger::GetInstance() -> log("step B1", Logging::Logger::level(Logging::INFO));
            this->D = gllimParams.D;
            this->L = gllimParams.L;
            this->K = gllimParams.K;
            this->Pi = gllimParams.Pi;
            this->Gamma = gllimParams.Gamma;
            this->Sigma = gllimParams.Sigma;
            this->C = gllimParams.C;
            this->B = gllimParams.B;
            this->A = gllimParams.A;
            Logging::Logger::GetInstance() -> log("step B2", Logging::Logger::level(Logging::INFO));
            return *this;
        }

        GLLiMParameters(std::shared_ptr<GLLiMParameters> gllimParams){
            Logging::Logger::GetInstance() -> log("step A5", Logging::Logger::level(Logging::INFO));
            this->D = gllimParams->D;
            Logging::Logger::GetInstance() -> log("step A6", Logging::Logger::level(Logging::INFO));
            this->L = gllimParams->L;
            Logging::Logger::GetInstance() -> log("step A7", Logging::Logger::level(Logging::INFO));
            this->K = gllimParams->K;
            Logging::Logger::GetInstance() -> log("step A8", Logging::Logger::level(Logging::INFO));
            this->Pi = gllimParams->Pi;
            Logging::Logger::GetInstance() -> log("step A1", Logging::Logger::level(Logging::INFO));
            this->Gamma = std::vector<T>(K);
            this->Sigma = std::vector<U>(K);
            for(unsigned k=0; k<this->K; k++){
                this->Gamma[k] = gllimParams->Gamma[k];
                this->Sigma[k] = gllimParams->Sigma[k];
            }
            Logging::Logger::GetInstance() -> log("step A3", Logging::Logger::level(Logging::INFO));
            this->C = gllimParams->C;
            this->B = gllimParams->B;
            this->A = gllimParams->A;
            Logging::Logger::GetInstance() -> log("step A4", Logging::Logger::level(Logging::INFO));

        }

        vec Pi; /**< A vector of size K containing the weights of the gaussian distributions in the mixture */

        mat C; /**< A matrix of size (L,K) containing the means of the mixture of gaussian distribution that define low dimension data*/

        std::vector<T> Gamma; /**< A vector of K covariance matrices (L,L) of the mixture of gaussian distribution that define low dimension data*/

        cube A; /**< A cube of size (D,L,K), see the formula 2 in Antoine Deleforge, Florence Forbes, and Radu Horaud.
 * High-Dimensional Regression with Gaussian Mixtures and Partially-Latent Response Variables. Statistics and Computing 25(5): 893-911, September 2015. */

        mat B; /**< A matrix of size (D,K), see the formula 2 in Antoine Deleforge, Florence Forbes, and Radu Horaud.
 * High-Dimensional Regression with Gaussian Mixtures and Partially-Latent Response Variables. Statistics and Computing 25(5): 893-911, September 2015. */

        std::vector<U> Sigma;/**< A cube of size (D,D,K) containing the covariance matrices , see the formula 3 in Antoine Deleforge, Florence Forbes, and Radu Horaud.
 * High-Dimensional Regression with Gaussian Mixtures and Partially-Latent Response Variables. Statistics and Computing 25(5): 893-911, September 2015. */

        unsigned K; /**< The number of affine transformation which stands also for the number of gaussian distributions in the mixture */

        unsigned L; /**< Low dimension value */

        unsigned D; /**< High dimension value */


    };


}



#endif //KERNELO_GLLIMPARAMETERS_H
