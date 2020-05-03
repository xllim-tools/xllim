//
// Created by reverse-proxy on 29‏/3‏/2020.
//

#include "GaussianMixtureProposition.h"
#include "../../helpersFunctions/Helpers.h"
#include "../../helpersFunctions/Helpers.h"

#define LOG_2_PI log(2* datum::pi)

using namespace importanceSampling;

GaussianMixtureProposition::GaussianMixtureProposition(vec &weights, mat &means, cube &covariances) {
    gmm.set_params(means, covariances, weights.t());
    L = means.n_rows;
}

vec GaussianMixtureProposition::sample() {
    return gmm.generate();
}

mat GaussianMixtureProposition::proposition_covariance() {

    unsigned L = gmm.means.n_rows;
    unsigned K = gmm.hefts.n_cols;

    vec mean_mean_mixture(L, fill::zeros);
    mat mean_cov_mixture(L,L,fill::zeros);

    for(unsigned k=0; k<K; k++){
        mean_mean_mixture += gmm.hefts(k) * gmm.means.col(k);
        mean_cov_mixture += gmm.fcovs.slice(k) + gmm.means.col(k) * gmm.means.col(k).t() * gmm.hefts(k);
    }
    mean_cov_mixture -= mean_mean_mixture * mean_mean_mixture.t();

    return mean_cov_mixture;
}

double GaussianMixtureProposition::proposition_log_density(vec x_sample) {

    unsigned K = gmm.hefts.n_cols;
    unsigned L = gmm.means.n_rows;

    vec densities(K, fill::zeros);

    for(unsigned k=0; k<K; k++){
        vec x_u = x_sample - gmm.means.col(k);
        mat cov = gmm.fcovs.slice(k);
        cov.diag() += 1e-08;
        double density_k = -0.5 * (L * LOG_2_PI + log(Helpers::computeDeterminant(cov)) + dot((rowvec(x_u.t()) * Helpers::inverseMatrix(cov)).t() , x_u));
        densities(k) = density_k + log(gmm.hefts(k));
    }

    return Helpers::logSumExp(densities);
}
