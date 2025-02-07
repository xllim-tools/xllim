#ifndef JGMM_HPP
#define JGMM_HPP

#include "gllim.hpp" // Only need GLLiMParameters => separate GLLiMParameters in different file

class JGMM
{
public:
    JGMM();
    void train(const mat &x, const mat &y, GLLiMParameters<FullCovariance,FullCovariance> &theta, unsigned kmeans_iteration, unsigned em_iteration, double floor, int verbose = 1);
    mat getPosterior();

private:
    unsigned L_;
    unsigned D_;
    unsigned K_;
    gmm_full jgmm_; // better than (weights + covs + means) ?
    // vec jgmm_weights;      // The weights of the GMM equivalent to the GLLiM model.
    // mat jgmm_means;        // The means of the GMM equivalent to the GLLiM model.
    // cube jgmm_covariances; // The covariance matrices of the GMM equivalent to the GLLiM model.
    mat posterior_; // the posterior from the training of the GMM

    void GLLiMParameterstoJGMM(GLLiMParameters<FullCovariance,FullCovariance> &initial_theta);
    void JGMMtoGLLiMParameters(GLLiMParameters<FullCovariance, FullCovariance> &theta);
};

#endif // JGMM_HPP
