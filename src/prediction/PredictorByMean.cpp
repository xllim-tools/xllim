//
// Created by reverse-proxy on 26‏/3‏/2020.
//

#include "PredictorByMean.h"

using namespace prediction;

PredictorByMean::PredictorByMean(const std::shared_ptr<learningModel::IGLLiMLearning>& learningModel) {
    this->learningModel = learningModel;
}

void PredictorByMean::predict(const vec &y_obs, const vec& cov_obs) {
    arma::gmm_full gmm_obs = learningModel->computeGMM(y_obs, cov_obs);

    vec mixtureMean = computeMixtureMean(gmm_obs.hefts, gmm_obs.means);
    vec mixtureCov = computeMixtureCov(gmm_obs.hefts, gmm_obs.means, gmm_obs.fcovs);

}

vec PredictorByMean::computeMixtureMean(const vec &weights, const mat &means) {
    vec barycenter(means.n_rows, fill::zeros);
    for(unsigned k=0; k<weights.n_rows; k++){
        barycenter += means.col(k) * weights(k);
    }
    return barycenter;
}

mat PredictorByMean::computeMixtureCov(const vec &weights, const mat &means, const cube &covs) {
    unsigned L = means.n_rows;
    vec mean_mean_mixture = computeMixtureMean(weights, means);
    mat mean_cov_mixture(L,L,fill::zeros);

    for(unsigned k=0; k<weights.n_rows; k++){
        mean_cov_mixture += covs.slice(k) + means.col(k)  * means.col(k) .t() * weights(k);
    }
    mean_cov_mixture -= mean_mean_mixture * mean_mean_mixture.t();

    return mean_cov_mixture;
}
