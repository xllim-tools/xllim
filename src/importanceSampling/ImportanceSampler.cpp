//
// Created by reverse-proxy on 29‏/3‏/2020.
//

#include "ImportanceSampler.h"
#include "../helpersFunctions/Helpers.h"

#include <utility>

using namespace importanceSampling;

ImportanceSampler::ImportanceSampler(std::shared_ptr<DataGeneration::StatModel> statModel,
                                     std::shared_ptr<ISProposition> isProposition) :
                                     IimportanceSampler(std::move(statModel),std::move(isProposition)) {}

void ImportanceSampler::importanceSampling() {
    unsigned L; // need to get L
    mat X_samples(L, N_samples, fill::zeros);
    vec weights_samples(N_samples, fills::zeros);

    vec is_mean(L, fill::zeros);
    vec is_variance(L, fill::zeros);

    // Start importance sampling and gather diagnostic results
    diagnostic(X_samples, weights_samples);

    // Compute mean predictor
    for(unsigned n=0; n<N_samples ; n++){
        is_mean += X_samples.col(n) * weights_samples(n);
    }

    // Compute variance predictor
    for(unsigned n=0; n<N_samples ; n++){
        is_variance += weights_samples(n) * pow(X_samples.col(n) - is_mean, 2);
    }



}

void ImportanceSampler::diagnostic(mat &samples, vec &weights) {
    unsigned N_samples = samples.n_cols, L_samples = samples.n_rows;
    unsigned nb_effective_sample = 0;
    double max_target_log_density = -datum::inf;

    vec target_log_densities(N_samples);
    vec proposition_log_densities(N_samples);

    for(unsigned n=0; n<N_samples; n++){

        // sample X_n
        samples.col(n) = this->isProposition->sample(L_samples);

        // compute target density
        //target_log_densities = this->statModel->density_X_Y()

        // we save the number of effective samples that their weight is not null
        if(target_log_densities(n) != -datum::inf){
            nb_effective_sample++;
        }
        if(target_log_densities(n) > max_target_log_density){
            max_target_log_density = target_log_densities(n);
        }

        // compute proposition density
        proposition_log_densities(n) = isProposition->proposition_log_density(samples.col(n));
    }

    double max_weight = 0, sum_weights_2 = 0, sum_weights = 0;

    weights = target_log_densities - proposition_log_densities; // verify numerical stability
    sum_weights = Helpers::logSumExp(weights);
    sum_weights_2 = Helpers::logSumExp(2 * weights);

    double effective_sample_size = exp(2 * sum_weights - sum_weights_2);
    weights -= sum_weights;
    weights = exp(weights);
}
