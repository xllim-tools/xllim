#include "../estimators/GmmEstimator.h"
#include "../estimators/EmEstimator.h"

#include "FixedInitializer.h"

using namespace learningModel;

template<typename T, typename U>
FixedInitializer<T, U>::FixedInitializer(const std::shared_ptr<FixedInitConfig> &config) {
    this->config = config;
}

template<typename T, typename U>
std::shared_ptr <GLLiMParameters<T, U>> FixedInitializer<T, U>::execute(const mat &x, const mat &y, unsigned nb_gaussians) {
    unsigned K = nb_gaussians;
    unsigned L = x.n_cols;
    unsigned D = y.n_cols;

    std::shared_ptr<GLLiMParameters <T, U>> theta(new GLLiMParameters <T, U>(D,L,K));


    // generate a mean for the GMM using a data generator strategy
    mat m(L,K);
    config->generator->execute(m, config->seed);

    // use the same weight for all the clusters
    vec rho = ones(K)/K;

    // Create a cube of K covariance matrices with a homothety constraint
    mat cov(L,L,fill::zeros);
    cov.diag() += sqrt(1.0/(pow(K, 1.0/L)));
    cube v(L,L,K);
    v.each_slice() = cov;

    // train a GMM over one iteration
    GmmEstimator gmmEstimator = GmmEstimator(std::make_shared<GMMLearningConfig>(config->gmmLearningConfig));
    gmmEstimator.train(x.t(),rho,m,v);

    // compute log_rnk using the posterior of the GMM after the training
    mat log_rnk(x.n_rows,K);
    log_rnk = gmmEstimator.getPosterior();

    // Compute theta of the GLLiM using the log_posterior of the GMM
    EmEstimator<T,U> emEstimator = EmEstimator<T,U>(std::make_shared<EMLearningConfig>(config->emLearningConfig));
    emEstimator.next_theta(x.t(),y.t(),log_rnk,theta);

    return theta;
}



