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
    cube gmm_covs;   // The covariance matrices of each component in the GMM (D, D, K). The covariance is indenpendent from the observation.

    MeanPredictionResult(unsigned N_obs, unsigned D, unsigned K) : mean(N_obs, D), variance(N_obs, D, D), gmm_weights(N_obs, K), gmm_means(N_obs, D, K), gmm_covs(D, D, K) {}
};

struct CenterPredictionResult
{
    mat weights;            // The weights of the merged GMM with shape (N_obs, K_merged)
    cube means;             // The means of the merged GMM with shape (N_obs, D, K_merged). It corresponds to the centers that stands for the predictions
    std::vector<cube> covs; // The covariance of the merged GMM with shape (N_obs, D, D, K_merged). It is constructed from other gaussians means thus it depends on observations.// ! TODO field

    CenterPredictionResult(unsigned N_obs, unsigned D, unsigned K_merged) : weights(N_obs, K_merged), means(N_obs, D, K_merged), covs(N_obs, cube(D, D, K_merged))
    {
        // cube template_cube(D, D, K_merged); // Initialize a single cube
        // covs.fill(template_cube);           // Fill it with the template cube
    }
};

struct PredictionResult
{
    MeanPredictionResult meanPredResult;     // @see MeanPredictionResult MeanPredictionResult
    CenterPredictionResult centerPredResult; // @see CenterPredictionResult CenterPredictionResult

    PredictionResult(unsigned N_obs, unsigned D, unsigned K, unsigned K_merged = 0) : meanPredResult(N_obs, D, K), centerPredResult(N_obs, D, K_merged) {}
};

#endif // PREDICTIONRESULTS_HPP
