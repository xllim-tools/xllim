//
// Created by reverse-proxy on 29‏/3‏/2020.
//

#include "ImportanceSampler.h"
#include "../helpersFunctions/Helpers.h"

#include <utility>

using namespace importanceSampling;


ISResult ImportanceSampler::importanceSampling(
        const std::shared_ptr<ISTarget>& isTarget,
        const std::shared_ptr<ISProposition>& isProposition,
        const vec &y_obs,
        const vec &y_cov,
        unsigned L,
        unsigned N_samples) {

    mat X_samples(L, N_samples, fill::zeros);
    vec weights_samples(N_samples, fill::zeros);

    ISResult result{};
    result.mean = vec(L, fill::zeros);
    result.covariance = vec(L, fill::zeros);

    // Start importance sampling and gather diagnostic results
    result.diagnostic = diagnostic(X_samples, weights_samples, y_obs, y_cov, isProposition, std::move(isTarget));

    // Compute mean predictor
    for(unsigned n=0; n<N_samples ; n++){
        result.mean += X_samples.col(n) * weights_samples(n);
    }

    // Compute variance predictor
    for(unsigned n=0; n<N_samples ; n++){
        result.covariance += weights_samples(n) * pow(X_samples.col(n) - result.mean, 2);
    }

    return result;
}

ISDiagnostic ImportanceSampler::diagnostic(
        mat &samples,
        vec &weights,
        const vec &y_obs,
        const vec &y_cov,
        const std::shared_ptr<ISProposition>& isProposition,
        const std::shared_ptr<ISTarget>& isTarget) {

    unsigned N_samples = samples.n_cols, L_samples = samples.n_rows;
    double max_target_log_density = -datum::inf;

    vec target_log_densities(N_samples);
    vec proposition_log_densities(N_samples);

    ISDiagnostic diagnostic{};
    diagnostic.nb_effective_sample = 0;

    for(unsigned n=0; n<N_samples; n++){

        // sample X_n
        samples.col(n) = isProposition->sample(L_samples);

        // compute target density
        //target_log_densities = this->statModel->density_X_Y()

        // we save the number of effective samples that their weight is not null
        if(target_log_densities(n) != -datum::inf){
            diagnostic.nb_effective_sample++;
        }
        if(target_log_densities(n) > max_target_log_density){
            max_target_log_density = target_log_densities(n);
        }

        // compute proposition density
        proposition_log_densities(n) = isProposition->proposition_log_density(samples.col(n));
    }

    double sum_weights_2 = 0, sum_weights = 0;

    weights = target_log_densities - proposition_log_densities; // verify numerical stability
    sum_weights = Helpers::logSumExp(weights);
    sum_weights_2 = Helpers::logSumExp(2 * weights);

    diagnostic.effective_sample_size = exp(2 * sum_weights - sum_weights_2);
    diagnostic.qn = exp(weights.max() - sum_weights);

    weights -= sum_weights;
    weights = exp(weights);

    return diagnostic;
}
