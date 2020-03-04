#include "../estimators/GmmEstimator.h"
#include "../estimators/EmEstimator.h"

#include "MultInitializer.h"

using namespace learningModel;

template<typename T, typename U>
MultInitializer<T, U>::MultInitializer(const std::shared_ptr<MultInitConfig> &config) {
    this->config = config;
}

template<typename T, typename U>
std::shared_ptr <GLLiMParameters<T, U>> MultInitializer<T, U>::execute(const mat &x, const mat &y, unsigned nb_gaussians) {
    unsigned K = nb_gaussians;
    unsigned L = x.n_cols;
    unsigned D = y.n_cols;
    unsigned N = x.n_rows;
    double best_log_likelihood = -(datum::inf);
    double log_likelihood;
    std::shared_ptr<GLLiMParameters <T, U>> best_theta(new GLLiMParameters <T, U>(D,L,K));;
    std::shared_ptr<GLLiMParameters <T, U>> local_theta(new GLLiMParameters <T, U>(D,L,K));
    mat best_log_rnk(N,K, fill::zeros);
    mat log_rnk(N,K, fill::zeros);

    Col<unsigned> seeds(config->nb_experiences, fill::randn); // will be changed later


    for(unsigned exp=0; exp<config->nb_experiences; exp++){
        // generate a mean for the GMM using a data generator strategy
        mat m(L,K);
        config->generator->execute(m, seeds(exp) );

        // use the same weight for all the clusters
        vec rho = ones(K)/K;

        // Create a cube of K covariance matrices with a homothety constraint
        mat cov(L,L,fill::zeros);
        cov.diag() += sqrt(1.0/(pow(K, 1.0/L)));
        cube v(L,L,K);
        v.each_slice() = cov;

        // train a GMM over nb_iter iteration
        GmmEstimator gmmEstimator = GmmEstimator(std::make_shared<GMMLearningConfig>(config->gmmLearningConfig));
        gmmEstimator.train(x.t(),rho,m,v);

        // compute log_rnk using the posterior of the GMM after the training
        log_rnk = gmmEstimator.getPosterior();

        // Compute theta of the GLLiM using the log_posterior of the GMM
        EmEstimator<T,U> emEstimator = EmEstimator<T,U>(std::make_shared<EMLearningConfig>(config->emLearningConfig));
        emEstimator.next_theta(x.t(),y.t(),log_rnk,local_theta);

        for(unsigned iter=0; iter<config->nb_iter_EM; iter++){
            emEstimator.next_rnk(x.t(),y.t(),local_theta,log_rnk);
            emEstimator.next_theta(x.t(),y.t(),log_rnk,local_theta);
            std::cout << emEstimator.log_likelihood(log_rnk) << std::endl;
        }

        log_likelihood = emEstimator.log_likelihood(log_rnk);

        std::cout << log_likelihood << std::endl;

        if(log_likelihood > best_log_likelihood){
            best_theta = local_theta;
            best_log_likelihood = log_likelihood;
            best_log_rnk = log_rnk;
        }
    }

    return best_theta;
}




