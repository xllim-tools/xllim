#ifndef UTILS_HPP
#define UTILS_HPP

#include <armadillo>
// #include <omp.h>
using namespace arma;

namespace utils
{
    double logSumExp(const vec &x);
    vec logSumExp(const mat &x, const int axis);
    double weightedLogSumExp(const double &log_p1, const double &log_p2, const unsigned &c1, const unsigned &c2);
    vec weightedLogSumExp(const unsigned &c1, const vec &log_p1, const unsigned &c2, const vec &log_p2);
    mat proposition_covariance(gmm_full &gmm);
    vec Mahalanobis(const mat &x, const vec &center, const mat &cov);
    vec MahalanobisWithInvertedCov(const mat &x, const vec &center, const mat &inverted_cov);
    // double logDensity(const vec &x, const vec &mean, const mat &covariance);
    // double logDensity(const vec &x, const vec &mean, const vec &diag_covariance);
    // double logDensity(const vec &x, const vec &weight, const mat &mean, const cube &covariance);
    // mat logDensity(const mat &x, const rowvec &weight, const mat &mean, const cube &covariance); // full
    // mat logDensity(const mat &x, const rowvec &weight, const mat &mean, const mat &covariance); // diag
    // void inplace_tri_mat_mult(arma::rowvec &x, arma::mat const &trimat);
    // double mvnrm_arma_fast_chol(arma::rowvec const &x,arma::rowvec const &mean, arma::mat &chol, bool const logd = true);
    // vec dmvnrm_arma_fast_chol(arma::mat const &x, arma::rowvec const &mean, arma::mat &chol, bool const logd = true);
    // vec dmvnrm_arma_fast_chol_diag(arma::mat const &x, arma::rowvec const &mean, const arma::vec &chol, bool const logd = true);
    // mat safe_cholesky(mat &Sigma);

    // It implements the regularization algorithm to get predictions more adapter to a context that requires 
    // regularity between the predictions of different observations.
    // arg : series - cube of shape series is with shape (L, N_obs, K_merged) = (L, nb_wavelengths, nb_centers)
    // return : chosen permutation - Mat<unsigned int> of shape (K, N_obs)
    // ! The complexity is O(N_obs*K!*K)
    umat regularize(const cube &series);

    // Set the RNG (Random Number Generator) for an armadillo random generation method.
    // This methods takes into account if seed=0 are value
    void set_seed_armadillo(unsigned seed);
}

#endif // UTILS_HPP
