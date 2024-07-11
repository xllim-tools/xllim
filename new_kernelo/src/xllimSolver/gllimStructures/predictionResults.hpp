#ifndef PREDICTIONRESULTS_HPP
#define PREDICTIONRESULTS_HPP

#include <armadillo>

using namespace arma;

struct MeanPredictionResult
{
    mat mean;        // The mean of the GMM which stands for the prediction (N_obs, D)
    cube variance;   // The variance of the prediction (N_obs, D, D)
    mat gmm_weights; // The weights of the components of the GMM (N_obs, K)
    cube gmm_means;  // The means of each component in the GMM (N_obs, D, K)
    cube gmm_covs;   // The covariance matrices of each component in the GMM (D, D, K)

    MeanPredictionResult(unsigned N_obs, unsigned D, unsigned K) : mean(N_obs, D), variance(N_obs, D, D), gmm_weights(N_obs, K), gmm_means(N_obs, D, K), gmm_covs(D, D, K) {}
};

struct CenterPredictionResult
{
    vec weights; // The weights of the centers
    mat means;   // The centers that stands for the predictions
    cube covs;   // The covariance matrices of the centers
};

struct PredictionResult
{
    MeanPredictionResult meanPredResult;     // @see MeanPredictionResult MeanPredictionResult
    CenterPredictionResult centerPredResult; // @see CenterPredictionResult CenterPredictionResult

    PredictionResult(unsigned N_obs, unsigned D, unsigned K) : meanPredResult(N_obs, D, K) {}
};

#endif // PREDICTIONRESULTS_HPP
