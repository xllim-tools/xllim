/**
 * @file PredictionResult.h
 * @brief Internal structures of the results of the prediction
 * @author Sami DJOUADI
 * @version 1.2
 * @date 26/04/2020
 */

#ifndef KERNELO_PREDICTIONRESULT_H
#define KERNELO_PREDICTIONRESULT_H

#include <armadillo>

using namespace arma;

namespace prediction{
    /**
     * @struct MeanPredictionResult
     */
    struct MeanPredictionResult{
        vec mean; /**< The mean of the GMM which stands for the prediction*/
        vec variance; /**< The variance of the prediction*/
        vec gmm_weights; /**< The weights of the components of the GMM*/
        mat gmm_means;/**< The means of each component in the GMM*/
        cube gmm_covs;/**< The covariance matrices of each component in the GMM*/
    };

    /**
     * @struct CenterPredictionResult
     */
    struct CenterPredictionResult{
        vec weights; /**< The weights of the centers*/
        mat means; /**< The centers that stands for the predictions*/
        cube covs; /**< The covariance matrices of the centers*/

    };

    /**
     * @struct PredictionResult
     */
    struct PredictionResult{
        MeanPredictionResult meanPredResult; /**< @see MeanPredictionResult MeanPredictionResult*/
        CenterPredictionResult centerPredResult; /**< @see CenterPredictionResult CenterPredictionResult*/
    };
}

#endif //KERNELO_PREDICTIONRESULT_H
