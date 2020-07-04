/**
 * @file FixedInitializer.tpp
 * @brief FixedInitializer class implementation
 * @author Sami DJOUADI
 * @version 1.1
 * @date 23/03/2020
 */

#include "../estimators/GmmEstimator.h"
#include "../estimators/EmEstimator.h"

#include "FixedInitializer.h"

using namespace learningModel;

template<typename T, typename U>
FixedInitializer<T, U>::FixedInitializer(const std::shared_ptr<FixedInitConfig> &config) {
    this->config = config;
}

template<typename T, typename U>
std::shared_ptr <GLLiMParameters<T, U>> FixedInitializer<T, U>::execute(const mat &x, const mat &y, unsigned K) {
    unsigned L = x.n_cols;
    unsigned D = y.n_cols;

    Logging::Logger::GetInstance() -> log("Start Fixed initialization", level(Logging::INFO));

    std::shared_ptr<GLLiMParameters <T, U>> theta(new GLLiMParameters <T, U>(D,L,K));


    // generate a mean for the GMM using a data generator strategy
    Logging::Logger::GetInstance() -> log("\tGenerate GMM means", level(Logging::INFO));
    mat m(L,K);
    config->generator->execute(m);


    // use the same weight for all the clusters
    vec rho = ones(K)/K;

    // Create a cube of K covariance matrices with a homothety constraint
    Logging::Logger::GetInstance() -> log("\tGenerate GMM covariance matrices", level(Logging::INFO));
    mat cov(L,L,fill::zeros);
    cov.diag() += sqrt(1.0/(pow(K, 1.0/L)));
    cube v(L,L,K);
    v.each_slice() = cov;


    // train the GMM model
    Logging::Logger::GetInstance() -> log("\tTrain the GMM model", level(Logging::INFO));
    GmmEstimator gmmEstimator = GmmEstimator(config->gmmLearningConfig);
    gmmEstimator.train(x.t(),rho,m,v);

    // compute log_rnk using the posterior of the GMM after the training
    mat log_rnk(x.n_rows,K);
    log_rnk = gmmEstimator.getPosterior();

    // Compute theta of the GLLiM using the log_posterior of the GMM
    Logging::Logger::GetInstance() -> log("\tCompute Initial theta vector of the GLLiM model", level(Logging::INFO));
    EmEstimator<T,U> emEstimator = EmEstimator<T,U>(config->emLearningConfig);
    emEstimator.next_theta(x.t(),y.t(),log_rnk,theta);

    Logging::Logger::GetInstance() -> log("\tFinish Fixed initialization", level(Logging::INFO));

    return theta;
}



