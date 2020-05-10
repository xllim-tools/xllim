/**
 * @file GaussianRegularizedProposition.cpp
 * @brief GaussianRegularizedProposition class implementation
 * @author Sami DJOUADI
 * @version 1.2
 * @date 29/03/2020
 */

#include "GaussianRegularizedProposition.h"
#include "../../helpersFunctions/Helpers.h"

#define LOG_2_PI log(2* datum::pi)

using namespace importanceSampling;
using namespace arma;

vec GaussianRegularizedProposition::sample() {
    vec temp(L , fill::randn);
    vec x_sample = this->mean;
    for(unsigned l = 0; l < L; l++){
        x_sample += this->cov.col(l) * temp(l);
    }
    return x_sample;
}

mat GaussianRegularizedProposition::proposition_covariance() {
    return this->cov;
}

double GaussianRegularizedProposition::proposition_log_density(vec x_sample) {
    return 0;
}

GaussianRegularizedProposition::GaussianRegularizedProposition(vec &mean, mat &cov) {
    this->mean = mean;
    this->cov = cov;
    L = mean.n_rows;
}


