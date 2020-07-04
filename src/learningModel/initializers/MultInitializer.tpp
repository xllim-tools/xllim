/**
 * @file MultInitializer.tpp
 * @brief MultInitializer class implementation
 * @author Sami DJOUADI
 * @version 1.1
 * @date 03/03/2020
 */

#include "../estimators/GmmEstimator.h"
#include "../estimators/EmEstimator.h"

#include "MultInitializer.h"

using namespace learningModel;

template<typename T, typename U>
MultInitializer<T, U>::MultInitializer(const std::shared_ptr<MultInitConfig> &config) {
    this->config = config;
}

template<typename T, typename U>
std::shared_ptr <GLLiMParameters<T, U>> MultInitializer<T, U>::execute(const mat &x, const mat &y, unsigned K) {
    unsigned L = x.n_cols;
    unsigned D = y.n_cols;
    unsigned N = x.n_rows;
    double best_log_likelihood = -(datum::inf);
    double log_likelihood;
    std::shared_ptr<GLLiMParameters <T, U>> best_theta(new GLLiMParameters <T, U>(D,L,K));;
    std::shared_ptr<GLLiMParameters <T, U>> local_theta(new GLLiMParameters <T, U>(D,L,K));
    mat best_log_rnk(N,K, fill::zeros);
    mat log_rnk(N,K, fill::zeros);

    mat m(L,K);
    vec rho(K);
    mat cov(L,L);
    GmmEstimator gmmEstimator;
    EmEstimator<T,U> emEstimator;

    Logging::Logger::GetInstance() -> log("Start Multi initialization", level(Logging::INFO));

    for(unsigned exp=0; exp<config->nb_experiences; exp++){
        Logging::Logger::GetInstance() -> log("Initialisation : " + std::to_string(exp), level(Logging::INFO));
        // generate a mean for the GMM using a data generator strategy
        Logging::Logger::GetInstance() -> log("\tGenerate GMM means", level(Logging::INFO));
        config->generator->execute(m);

        // use the same weight for all the clusters
        rho = ones(K)/K;

        // Create a cube of K covariance matrices with a homothety constraint
        Logging::Logger::GetInstance() -> log("\tGenerate GMM covariance matrices", level(Logging::INFO));
        cov = zeros(L,L);
        cov.diag() += sqrt(1.0/(pow(K, 1.0/L)));
        cube v(L,L,K);
        v.each_slice() = cov;

        // train a GMM over nb_iter iteration
        Logging::Logger::GetInstance() -> log("\tTrain the GMM model", level(Logging::INFO));
        gmmEstimator = GmmEstimator(config->gmmLearningConfig);
        gmmEstimator.train(x.t(),rho,m,v);

        // compute log_rnk using the posterior of the GMM after the training
        log_rnk = gmmEstimator.getPosterior();


        // Compute theta of the GLLiM using the log_posterior of the GMM
        Logging::Logger::GetInstance() -> log("\tCompute Initial theta vector of the GLLiM model", level(Logging::INFO));
        emEstimator = EmEstimator<T,U>(config->emLearningConfig);
        emEstimator.next_theta(x.t(),y.t(),log_rnk,local_theta);

        Logging::Logger::GetInstance() -> log("\tTrain the initial GLLiM model", level(Logging::INFO));
        for(unsigned iter=0; iter<config->nb_iter_EM; iter++){
            emEstimator.next_rnk(x.t(),y.t(),local_theta,log_rnk);
            emEstimator.next_theta(x.t(),y.t(),log_rnk,local_theta);
        }

        log_likelihood = emEstimator.log_likelihood(log_rnk);

        if(log_likelihood > best_log_likelihood){
            best_theta = local_theta;
            best_log_likelihood = log_likelihood;
            best_log_rnk = log_rnk;
        }

        Logging::Logger::GetInstance() -> log("\tCurrent log likelihood : " + std::to_string(log_likelihood) +
                                              ", Best log likelihood : " + std::to_string(best_log_likelihood),
                                              level(Logging::INFO));
    }
    Logging::Logger::GetInstance() -> log("Finish Multi initialization", level(Logging::INFO));
    return best_theta;
}




