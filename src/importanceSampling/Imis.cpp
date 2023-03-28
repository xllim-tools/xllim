#include "Imis.h"
#include "../helpersFunctions/Helpers.h"

#include <utility>

using namespace importanceSampling;

// [TODO] il y a deux méthodes "execute" pour appliquer IS. Je ne sais pas vraiment dans quel cas l'une ou l'autre est utilisé

ISResult Imis::execute(
        std::shared_ptr<ISProposition> isProposition,
        const vec &y_obs,
        const vec &y_cov) {

    unsigned L = isProposition->getDimension();

    mat X_samples(L, N_0, fill::zeros);
    vec weights_samples(N_0, fill::zeros);

    ISResult result{};
    result.mean = vec(L, fill::zeros);
    result.covariance = vec(L, fill::zeros);

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

    // [TODO] insérer la fonction en C puis corriger/débugger
    // void fnIMIS(const size_t InitSamples, const size_t StepSamples, const size_t FinalResamples, const size_t MaxIter, const size_t NumParam, unsigned long int rng_seed, const char * runName)
    // {
    // }

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
