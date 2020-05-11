/**
 * @file GLLiM.h
 * @brief GLLiM class definition
 * @author Sami DJOUADI
 * @version 1.1
 * @date 04/03/2020
 */

#ifndef KERNELO_GLLIM_H
#define KERNELO_GLLIM_H

namespace learningModel{

    /**
     * @class GLLiM
     * @details This class wraps the parameters of the GLLiM model in order to export it to third party language API .
     * The data structures are stored as pointers in order to facilitate the integration with third party language API.
     */
    class GLLiM {
    public:
        unsigned K; /**< The number of affine transformation which stands also for the number of gaussian distributions in the mixture */

        unsigned L; /**< Low dimension value */

        unsigned D; /**< High dimension value */

        double *Pi; /**< A vector of size K containing the weights of the gaussian distributions in the mixture */

        double *C; /**< A matrix of size (L,K) containing the means of the mixture of gaussian distribution that define low dimension data*/

        double *B; /**< A matrix of size (D,K), see the formula 2 in Antoine Deleforge, Florence Forbes, and Radu Horaud.
 * High-Dimensional Regression with Gaussian Mixtures and Partially-Latent Response Variables. Statistics and Computing 25(5): 893-911, September 2015. */

        double *Gamma; /**< A cube of size (L,L,K) containing the covariance matrices of the mixture of gaussian distribution that define low dimension data*/

        double *Sigma; /**< A cube of size (D,D,K) containing the covariance matrices , see the formula 3 in Antoine Deleforge, Florence Forbes, and Radu Horaud.
 * High-Dimensional Regression with Gaussian Mixtures and Partially-Latent Response Variables. Statistics and Computing 25(5): 893-911, September 2015. */

        double *A; /**< A cube of size (D,L,K), see the formula 2 in Antoine Deleforge, Florence Forbes, and Radu Horaud.
 * High-Dimensional Regression with Gaussian Mixtures and Partially-Latent Response Variables. Statistics and Computing 25(5): 893-911, September 2015. */
    };

}

#endif //KERNELO_GLLIM_H
