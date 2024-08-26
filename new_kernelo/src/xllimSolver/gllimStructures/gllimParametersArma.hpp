#ifndef GLLIMPARAMETERSARMA_HPP
#define GLLIMPARAMETERSARMA_HPP

#include <armadillo>

using namespace arma;

struct GLLiMParametersBase
{
    virtual ~GLLiMParametersBase() = default;
};

template <typename TGamma, typename TSigma>
struct GLLiMParametersArma : public GLLiMParametersBase
{
    rowvec Pi;                   // A row vector of size K containing the weights of the gaussian distributions in the mixture
    cube A;                      // A cube of size (D,L,K)
    mat C;                       // A matrix of size (L,K) containing the means of the mixture of gaussian distribution that define low dimension data
    typename TGamma::Type Gamma; // A vector of size K containing the covariance matrices (L,L) of type  of the mixture of gaussian distribution that define low dimension data
    mat B;                       // A matrix of size (D,K)
    typename TSigma::Type Sigma; // A vector of size K containing the covariance matrices (D,D) of the mixture of gaussian distribution that define high dimension data

    // For more information on these parameters, see the formula 2 in Antoine Deleforge, Florence Forbes, and Radu Horaud.
    // High-Dimensional Regression with Gaussian Mixtures and Partially-Latent Response Variables. Statistics and Computing 25(5): 893-911, September 2015.

    GLLiMParametersArma(unsigned K, unsigned D, unsigned L) : Pi(K), A(D, L, K), C(L, K), B(D, K)
    {
        Gamma = TGamma::getTypeSize(K, L);
        Sigma = TSigma::getTypeSize(K, D);
    }
};

#endif // GLLIMPARAMETERSARMA_HPP
