/**
 * @file ImportanceSampler.cpp
 * @brief ImportanceSampler calss implementation
 * @author Sami DJOUADI
 * @version 1.2
 * @date 29/03/2020
 */

#include "ImportanceSampler.h"
#include "../helpersFunctions/Helpers.h"

#include <utility>

using namespace importanceSampling;


ISResult ImportanceSampler::execute(
        std::shared_ptr<ISProposition> isProposition,
        const vec &y_obs,
        const vec &y_cov) {

    unsigned L = isProposition->getDimension();

    mat X_samples(L, N_Samples, fill::zeros);
    vec weights_samples(N_Samples, fill::zeros);

    ISResult result{};
    result.mean = vec(L, fill::zeros);
    result.covariance = vec(L, fill::zeros);

    // Start importance sampling and gather diagnostic results
    result.diagnostic = diagnostic(X_samples, weights_samples, y_obs, y_cov, isProposition);

    // Compute mean predictor
    for(unsigned n=0; n<N_Samples ; n++){
        result.mean += X_samples.col(n) * weights_samples(n);
    }

    // Compute variance predictor
    for(unsigned n=0; n<N_Samples ; n++){
        result.covariance += weights_samples(n) * pow(X_samples.col(n) - result.mean, 2);
    }

    return result;
}

void ImportanceSampler::execute(
        std::shared_ptr<ISProposition> isProposition,
        double *y_obs,
        double *y_cov,
        unsigned size,
        std::shared_ptr<ImportanceSamplingResult> resultExport){

    vec y_obs_arma(&y_obs[0],size, false, true);
    vec var_obs_arma(&y_cov[0],size, false, true);

    ISResult result = this->execute(isProposition, y_obs_arma, var_obs_arma);

    resultExport->diagnostic.nb_effective_sample = result.diagnostic.nb_effective_sample;
    resultExport->diagnostic.effective_sample_size = result.diagnostic.effective_sample_size;
    resultExport->diagnostic.qn = result.diagnostic.qn;

    for(unsigned j=0 ; j<isProposition->getDimension(); j++){
        resultExport->mean[j] = result.mean(j);
        resultExport->covariance[j] = result.covariance(j);
    }
}

ISDiagnostic ImportanceSampler::diagnostic(
        mat &samples,
        vec &weights,
        const vec &y_obs,
        const vec &y_cov,
        std::shared_ptr<ISProposition> isProposition) {

    unsigned N_samples = samples.n_cols, L_samples = samples.n_rows;
    double max_target_log_density = -datum::inf;

    vec target_log_densities(N_samples);
    vec proposition_log_densities(N_samples);

    ISDiagnostic diagnostic{};
    diagnostic.nb_effective_sample = 0;

    for(unsigned n=0; n<N_samples; n++){

        // sample X_n
        samples.col(n) = isProposition->sample();
        /*bool finish;
        do{
            finish = true;
            samples.col(n) = isProposition->sample(L_samples);
            for(auto x_s: samples.col(n)){
                if(x_s > 1 || x_s < 0){
                    finish = false;
                }
            }
        }while(!finish);*/

        // compute target density
        target_log_densities(n) = isTarget->target_log_density(samples.col(n), y_obs, y_cov);

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

ImportanceSampler::ImportanceSampler(unsigned N_Samples, std::shared_ptr<ISTarget> isTarget) {
    this->N_Samples = N_Samples;
    this->isTarget = isTarget;
}
