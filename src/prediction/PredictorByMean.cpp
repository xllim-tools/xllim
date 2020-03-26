//
// Created by reverse-proxy on 26‏/3‏/2020.
//

#include "PredictorByMean.h"

using namespace prediction;

PredictorByMean::PredictorByMean(const std::shared_ptr<learningModel::IGLLiMLearning>& learningModel) {
    this->learningModel = learningModel;
}

vec PredictorByMean::predict(const vec &y_obs, const vec& cov_obs) {
    arma::gmm_full gmm_obs = learningModel->computeGMM(y_obs, cov_obs);
    return computeMeansBarycenter(gmm_obs.hefts, gmm_obs.means);
}

vec PredictorByMean::computeMeansBarycenter(const vec &weights, const mat &means) {
    vec barycenter(means.n_rows, fill::zeros);
    for(unsigned k=0; k<weights.n_rows; k++){
        barycenter += means.col(k) * weights(k);
    }
    return barycenter;
}
