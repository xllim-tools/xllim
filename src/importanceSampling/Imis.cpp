#include "Imis.h"
#include "../helpersFunctions/Helpers.h"
#include "ImportanceSampler.h"

#include <utility>

using namespace importanceSampling;
using namespace arma;

// [TODO] il y a deux méthodes "execute" pour appliquer IS. Je ne sais pas vraiment dans quel cas l'une ou l'autre est utilisé

ISResult Imis::execute(
        std::shared_ptr<ISProposition> isProposition,
        const vec &y_obs,
        const vec &y_cov) {

    unsigned L = isProposition->getDimension();

    mat X_samples(L, N_tot, arma::fill::zeros);
    vec weights_samples(N_tot, arma::fill::zeros);

    ISResult result{};
    result.mean = vec(L, arma::fill::zeros);
    result.covariance = vec(L, arma::fill::zeros);

    // Start importance sampling and gather diagnostic results
    result.diagnostic = diagnostic(X_samples, weights_samples, y_obs, y_cov, isProposition);

    // Compute mean predictor
    for(unsigned n=0; n<N_tot ; n++){
        result.mean += X_samples.col(n) * weights_samples(n);
    }

    // Compute variance predictor
    for(unsigned n=0; n<N_tot ; n++){
        result.covariance += weights_samples(n) * pow(X_samples.col(n) - result.mean, 2);
    }

    return result;
}

void Imis::execute(
        std::shared_ptr<ISProposition> isProposition,
        double *y_obs,
        double *y_cov,
        unsigned size,
        std::shared_ptr<ImportanceSamplingResult> resultExport){

    vec y_obs_arma(&y_obs[0],size, false, true);
    vec var_obs_arma(&y_cov[0],size, false, true);

    ISResult result = this->execute(isProposition, y_obs_arma, var_obs_arma);

    resultExport->diagnostic->nb_effective_sample = result.diagnostic.nb_effective_sample;
    resultExport->diagnostic->effective_sample_size = result.diagnostic.effective_sample_size;
    resultExport->diagnostic->qn = result.diagnostic.qn;

    for(unsigned j=0 ; j<isProposition->getDimension(); j++){
        resultExport->mean[j] = result.mean(j);
        resultExport->covariance[j] = result.covariance(j);
    }
}

ISDiagnostic Imis::diagnostic(
        mat &samples,
        vec &weights,
        const vec &y_obs,
        const vec &y_cov,
        std::shared_ptr<ISProposition> isProposition) {

    // Setup useful classes, matrix and vectors
    ISDiagnostic diagnostic{};
    vec target_log_densities(N_tot, arma::fill::zeros);
    vec proposition_log_densities(N_tot, arma::fill::zeros);
    vec proposition_log_densities_0(N_tot, arma::fill::zeros);
    unsigned L = isProposition->getDimension();
    mat means(L, J, arma::fill::zeros);
    field<mat> chols(J);

    // Initial stage : standard IS generating N_0 weighted samples
    ImportanceSampler is_sampler = ImportanceSampler(N_0,isTarget);
    mat is_samples(L, N_0);
    vec is_weights(N_0), is_target_log_densities(N_0), is_proposition_log_densities(N_0);
    is_sampler.execute(is_samples, is_weights, is_target_log_densities, is_proposition_log_densities, y_obs, y_cov, isProposition);
    samples.cols(0,N_0-1) = is_samples ;
    weights.subvec(0,N_0-1) = is_weights;
    target_log_densities.subvec(0,N_0-1) = is_target_log_densities;
    proposition_log_densities.subvec(0,N_0-1) = is_proposition_log_densities;

    /* The covariance matrix from the initial proposition law is used for each IMIS iteration
    The advandage is that the inverse of the covariance matrix is only computed once.
    However to improve IMIS precision the Mahalanobis distance should be calculated at each step with Covariance of each new proposition law */
    mat proposition_covariance = isProposition->proposition_covariance();
    mat Cov_proposition_0 = trimatl(proposition_covariance.i());

    size_t j_step, N_j, N_j1; // declared outside so can use later

    // IMIS steps
    for(j_step = 0; j_step < J; j_step++){

        N_j = (N_0 + j_step*B);
        N_j1 = N_j + B;
        
        // a) Find highest weigth
        uword i_max = weights.subvec(0,N_j-1).index_max();
        vec x_max = samples.col(i_max);

        // b) Find the B inputs with smallest Mahalanobis distance to x_max
        vec mahalanobis_dist(N_j, arma::fill::zeros);
        for(unsigned n = 0; n < N_j; n++){
            vec x_tmp = samples.col(n) - x_max;
            vec Cov_x_tmp = Cov_proposition_0 * x_tmp;
            mahalanobis_dist(n) = arma::dot(x_tmp, Cov_x_tmp);
            
        }
        uvec neighboors_idx = arma::sort_index(mahalanobis_dist);

        // d) Compute associated covariance
        /*  Raftery & Bao propose the formula
            w = (ws[id] + (1 / Nk)) / 2 (average between importance and 1/Nk)
            but, according to Fasalio et al 2016, not weighting increases stability
            w = 1*/
        mat Sigma_j(L, L, arma::fill::zeros);
        for(unsigned id = 0; id < B; id++){
            vec u_tmp = x_max - samples.col(neighboors_idx[id]);
            Sigma_j +=  u_tmp * u_tmp.t();
        }
        Sigma_j /= B;
        mat Chol_j = Helpers::safe_cholesky(Sigma_j); // safe cholesky

        means.col(j_step) = x_max; // save the mean ...
        chols(j_step) = Chol_j; // ... and the cholesky decomposition of the variance

        // e) Generate B new samples
        samples.cols(N_j, N_j1-1) = mvnrnd(x_max, Sigma_j, B);
        
        // h) Update proposition law

        /* Update current points [0:N_j-1]:
            for existing points, we can update the weigths without computing the whole mixture using
            prop_j1 = (N_j / N_j1) * prop_j + (B / N_j1) * phi_j1 */
        vec log_phi_j1 = Helpers::dmvnrm_arma_fast_chol(samples.cols(0,N_j-1).t(), x_max.t(), Chol_j); // from https://gallery.rcpp.org/articles/dmvnorm_arma/, Chol_j must be trimatu
        for(unsigned n = 0; n < N_j; n++){
            proposition_log_densities(n) = Helpers::weightedLogSumExp(proposition_log_densities(n), log_phi_j1(n), N_j, B) - log(N_j1);
        } // [TODO] manage to vectorize this function

        /* Update new points [N_j:N_j1-1]:
            for new points, we have to compute the whole mixture (of j_step first components) using
            prop_j1 = (N_0 / N_j1) * prop_0 + (B / N_j1) * SUM(phi_j, 1:j) */
        vec sum_phi(B);
        for(unsigned n = 0; n < j_step+1; n++){
            sum_phi += Helpers::dmvnrm_arma_fast_chol(samples.cols(N_j,N_j1-1).t(), means.col(n).t(), chols(n), false);
        }
        for(unsigned n = N_j; n < N_j1; n++){
            proposition_log_densities_0(n) = isProposition->proposition_log_density(samples.col(n));
            proposition_log_densities(n) = Helpers::weightedLogSumExp(proposition_log_densities_0(n), log(sum_phi(n-N_j)), N_0, B) - log(N_j1);
            target_log_densities(n) = isTarget->target_log_density(samples.col(n), y_obs, y_cov);
        } // [TODO] manage to vectorize these 3 functions

        // i) Update all weights
        weights.subvec(0,N_j1-1) = target_log_densities.subvec(0,N_j1-1) - proposition_log_densities.subvec(0,N_j1-1); // Careful: here we manipulate log(weights)
        weights.subvec(0,N_j1-1) -= Helpers::logSumExp(weights.subvec(0,N_j1-1));
        weights.subvec(0,N_j1-1) = exp(weights.subvec(0,N_j1-1)); // back to real weights
    } // End of IMIS steps

    // QUOI mettre ?
    diagnostic.nb_effective_sample = N_0;
    diagnostic.effective_sample_size = B;
    diagnostic.qn = J;

    return diagnostic;
}

Imis::Imis(unsigned N_0, unsigned B, unsigned J, std::shared_ptr<ISTarget> isTarget) {
    this->N_0 = N_0;
    this->B = B;
    this->J = J;
    this->N_tot = N_0 + B*J;
    this->isTarget = isTarget;
}
