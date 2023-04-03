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

    mat X_samples(L, N_0, arma::fill::zeros);
    vec weights_samples(N_0, arma::fill::zeros);

    ISResult result{};
    result.mean = vec(L, arma::fill::zeros);
    result.covariance = vec(L, arma::fill::zeros);

    // Start importance sampling and gather diagnostic results
    result.diagnostic = diagnostic(X_samples, weights_samples, y_obs, y_cov, isProposition);

    // Compute mean predictor
    for(unsigned n=0; n<N_0 ; n++){
        result.mean += X_samples.col(n) * weights_samples(n);
    }

    // Compute variance predictor
    for(unsigned n=0; n<N_0 ; n++){
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


// [TODO] il s'agit de l'algorithme. Il est utilisé au sein de la fonction diagnostic qui renvoie
// (nb_effective_sample, effective_sample_size, qn)
// Je ne pense pas que cette structure soit totalement adapté à l'IMIS ou bien à de futures méthodes (Langevin, ...)
// Mais bon c'est comme ça que c'est fait dans le code Julia aussi.
// On pourrait par exemple faire une fonction imis à part qui renvoie (X,w) ...
// ... et la fonction diagnostic utilise ce résultat pour évaluer ess, es, qn

// IDEE: faire comme le code C avec l'algo et diagnostic dans la même fonction. 
//      On implémente tous les param (entropy, maxWheight,...) mais on renvoie que ess,es,qn
//      Demander à Sylvain/Florence quels sont les param de diagnostique intéressants et envisager un refactory de tous Important Sampling
//      
ISDiagnostic Imis::diagnostic(
        mat &samples,
        vec &weights,
        const vec &y_obs,
        const vec &y_cov,
        std::shared_ptr<ISProposition> isProposition) {

    ISDiagnostic diagnostic{};

// fnIMIS(const size_t InitSamples, const size_t B, const size_t FinalResamples, const size_t MaxIter, const size_t NumParam, unsigned long int rng_seed, const char * runName)

    // Declare and configure GSL RNG 
    // [TODO] change to Armadillo
    unsigned long int rng_seed = 77930367; // seed to compare

    // Setup IMIS arrays
    // malloc/free c'est du C. Il faut utiliser new/delete
    // gsl_matrix * Xmat = gsl_matrix_alloc(InitSamples + StepSamples*MaxIter, NumParam);
    // double * prior_all = (double*) malloc(sizeof(double) * (InitSamples + StepSamples*MaxIter));
    // ...
    unsigned N_tot = N_0 + J*B;
    vec target_log_densities(N_tot);
    vec proposition_log_densities(N_tot);
    vec proposition_log_densities_0(N_tot);
    unsigned L = isProposition->getDimension();
    mat means(L, J, arma::fill::zeros);
    field<mat> chols(J);
    // TODO] [IMPORTANT] Mettre toutes les définitions ici. Il ne faut pas de "mat", "vec",... dans la boucle


    // Initial stage : standard IS generating N_0 weighted samples
    // ImportanceSampler::execute(X_samples, weights_samples, y_obs, y_cov, isProposition);
    // ---------- standard IS ------------
    double max_target_log_density = -datum::inf;
    diagnostic.nb_effective_sample = 0;
    for(unsigned n=0; n<N_0; n++){
        // sample X_n
        samples.col(n) = isProposition->sample();
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
    // ------------------------------------------

    // Chol_proposition_0 for Mahalanobis distance computing
    mat proposition_covariance = isProposition->proposition_covariance();
    mat Cov_proposition_0 = trimatl(proposition_covariance.i());

    // IMIS steps
    // at each step we need the full mixture to compute qk for the new samples 
    // the weights are constant (thus easy to compute), dont need to store them
    size_t j_step = 0, N_j, N_j1; // declared outside so can use later

    for(j_step = 0; j_step < J; j_step++){

        N_j = (N_0 + j_step*B);
        N_j1 = N_j + B;
        
        // ----------- new gaussian ---------
        // a) Find highest weigth
        uword i_max = weights.index_max();
        vec x_max = samples.col(i_max);

        // b) Find the B inputs with smallest Mahalanobis distance to x_max
        // GetMahalanobis_diag(Xmat, center_all[imisStep],  prior_invCov_diag, numImisSamples, NumParam, distance);
        // qsort(distance, numImisSamples, sizeof(struct dst), cmp_dst);
        vec mahalanobis_dist(N_j);
        for(unsigned n = 0; n < N_j; n++){
            vec x_tmp = samples.col(n) - x_max;
            vec Chol_x_tmp = Cov_proposition_0 * x_tmp;
            mahalanobis_dist(n) = arma::dot(x_tmp, Chol_x_tmp);
            // mahalanobis_dist(n) = Mahalanobis(samples.col(n), x_max, cov);
            // Here the covariance matrix from the initial proposition law is used for each IMIS iteration
            // The advandage is that the inverse of the covariance matrix is only computed once.
            // However to improve IMIS precision the Mahalanobis distance shoulb be calculated with Covariance of each new proposition law
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
        std::cout << Sigma_j << std::endl;
        std::cout << arma::size(Sigma_j) << std::endl;
        std::cout << "HEYYYY before safe_chol" << std::endl;
        mat Chol_j = Helpers::safe_cholesky(Sigma_j); // safe cholesky
        // Probleme : boucle infinie ! (parfois)
        std::cout << Chol_j << std::endl;
        std::cout << arma::size(Chol_j) << std::endl;
        std::cout << "HEYYYY after safe_chol" << std::endl;
        // ----------------------------------
        means.col(j_step) = x_max; // save the mean ...
        chols(j_step) = Chol_j; // ... and the cholesky decomposition of the variance

        // e) Generate B new samples
        std::cout << "Generate B new samples" << std::endl;
        samples.cols(N_j, N_j1) = mvnrnd(x_max, Sigma_j, B); // Pas besoin de Cholesky ici !?
        
        // h) Update proposition law

        /* Update current points [0:N_j-1]:
            for existing points, we can update the weigths without computing the whole mixture using
            prop_j1 = (N_j / N_j1) * prop_j + (B / N_j1) * phi_j1 */
        std::cout << "Update current points [0:N_j-1]" << std::endl;
        vec log_phi_j1 = Helpers::dmvnrm_arma_fast_chol(samples.cols(0,N_j-1), x_max, Chol_j); // from https://gallery.rcpp.org/articles/dmvnorm_arma/
        for(unsigned n = 0; n < N_j; n++){
            std::cout << n << std::endl;
            proposition_log_densities(n) = Helpers::weightedLogSumExp(proposition_log_densities(n), log_phi_j1(n), N_j, B) - log(N_j1);
        } // [TODO] manage to vectorize this function

        /* Update new points [N_j:N_j1-1]:
            for new points, we have to compute the whole mixture (of j_step first components) using
            prop_j1 = (N_0 / N_j1) * prop_0 + (B / N_j1) * SUM(phi_j, 1:j) */
        std::cout << "Update current points [N_j:N_j1-1]" << std::endl;
        vec sum_phi(B);
        for(unsigned n = 0; n < j_step+1; n++){
            std::cout << n << std::endl;
            sum_phi += Helpers::dmvnrm_arma_fast_chol(samples.cols(N_j,N_j1-1), means.col(n), chols(n), false);
            std::cout << sum_phi << std::endl;
        }
        for(unsigned n = N_j; n < N_j1; n++){
            std::cout << n << std::endl;
            std::cout << N_j << std::endl;
            std::cout << N_j1 << std::endl;
            std::cout << size(sum_phi) << std::endl;
            std::cout << size(proposition_log_densities) << std::endl;
            std::cout << size(proposition_log_densities_0) << std::endl;
            proposition_log_densities_0(n) = isProposition->proposition_log_density(samples.col(n));
            std::cout << "proposition_log_density OK" << std::endl;
            proposition_log_densities(n) = Helpers::weightedLogSumExp(proposition_log_densities_0(n), log(sum_phi(n-N_j)), N_0, B) - log(N_j1);
            std::cout << "weightedLogSumExp OK" << std::endl;
            target_log_densities(n) = isTarget->target_log_density(samples.col(n), y_obs, y_cov);
        } // [TODO] manage to vectorize these 3 functions

        // i) Update all weights
        std::cout << "Update all weights" << std::endl;
        weights = target_log_densities - proposition_log_densities; // Careful: here we manipulate log(weights)
        weights -= Helpers::logSumExp(weights);
        weights = exp(weights); // back to real weights

    } // End of IMIS steps

    diagnostic.nb_effective_sample = N_0;
    diagnostic.effective_sample_size = B;
    diagnostic.qn = J;

    return diagnostic;
    // diagnostic.nb_effective_sample = ;
    // diagnostic.effective_sample_size = ;
    // diagnostic.qn = ;
}

Imis::Imis(unsigned N_0, unsigned B, unsigned J, std::shared_ptr<ISTarget> isTarget) {
    this->N_0 = N_0;
    this->B = B;
    this->J = J;
    this->isTarget = isTarget;
}
