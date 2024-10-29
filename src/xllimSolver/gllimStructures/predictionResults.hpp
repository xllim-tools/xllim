#ifndef PREDICTIONRESULTS_HPP
#define PREDICTIONRESULTS_HPP

#include <armadillo>

using namespace arma;

struct FullGMMResult
{
    mat mean;      // The mean of the full GMM which stands for the prediction (N_obs, D)
    cube variance; // The variance of the prediction (N_obs, D, D)
    mat weights;   // The weights of the components of the GMM (N_obs, K)
    cube means;    // The means of each component in the GMM (N_obs, D, K)
    cube covs;     // The covariance matrices of each component in the GMM (D, D, K). The covariance is indenpendent from the observation.

    FullGMMResult(unsigned N_obs, unsigned D, unsigned K) : mean(N_obs, D), variance(N_obs, D, D), weights(N_obs, K), means(N_obs, D, K), covs(D, D, K) {}
};

struct MergedGMMResult
{
    mat mean;               // The mean of the merged GMM which stands for the prediction (N_obs, D)
    cube variance;          // The variance of the prediction from the merged GMM (N_obs, D, D)
    mat weights;            // The weights of the merged GMM with shape (N_obs, K_merged)
    cube means;             // The means of the merged GMM with shape (N_obs, D, K_merged). It corresponds to the centers that stands for the predictions
    std::vector<cube> covs; // The covariance of the merged GMM with shape (N_obs, D, D, K_merged). It is constructed from other gaussians means thus it depends on observations.

    MergedGMMResult(unsigned N_obs, unsigned D, unsigned K_merged) : mean(N_obs, D), variance(N_obs, D, D), weights(N_obs, K_merged), means(N_obs, D, K_merged), covs(N_obs, cube(D, D, K_merged))
    {
        for (size_t n = 0; n < N_obs; n++)
        {
            for (size_t k = 0; k < K_merged; k++)
            {
                covs[n].slice(k).eye();
            }
        }
    }
};

struct PredictionResult
{
    FullGMMResult fullGMM;     // @see FullGMMResult
    MergedGMMResult mergedGMM; // @see MergedGMMResult

    PredictionResult(unsigned N_obs, unsigned D, unsigned K, unsigned K_merged = 0) : fullGMM(N_obs, D, K), mergedGMM(N_obs, D, K_merged) {}
};

#endif // PREDICTIONRESULTS_HPP
