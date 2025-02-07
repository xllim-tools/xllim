#ifndef GLLIMPARAMETERSARRAY_HPP
#define GLLIMPARAMETERSARRAY_HPP

#include <armadillo>

using namespace arma;

struct GLLiMParametersBase
{
    virtual ~GLLiMParametersBase() = default;
};

template <typename TGamma, typename TSigma>
struct GLLiMParametersArray : public GLLiMParametersBase
{
    rowvec Pi;                   // A vector of size K containing the weights of the gaussian distributions in the mixture
    cube A;                      // A cube of size (K,D,L)
    mat C;                       // A matrix of size (K,L) containing the means of the mixture of gaussian distribution that define low dimension data
    typename TGamma::Type Gamma; // An array of size (K,L,L) (depending on TGamma) containing the covariance matrices of type of the mixture of gaussian distribution that define low dimension data
    mat B;                       // A matrix of size (K,D)
    typename TSigma::Type Sigma; // An array of size (K,D,D) (depending on TGamma) containing the covariance matrices (D,D) of the mixture of gaussian distribution that define high dimension data

    // For more information on these parameters, see the formula 2 in Antoine Deleforge, Florence Forbes, and Radu Horaud.
    // High-Dimensional Regression with Gaussian Mixtures and Partially-Latent Response Variables. Statistics and Computing 25(5): 893-911, September 2015.

    GLLiMParametersArray(unsigned K, unsigned D, unsigned L) : A(K, D, L), C(K, L), B(K, D)
    {
        Pi = rowvec(K, fill::ones) * 1 / K;
        Gamma = TGamma::getTypeSize(K, L);
        Sigma = TSigma::getTypeSize(K, D);
    }
};

#endif // GLLIMPARAMETERSARRAY_HPP
