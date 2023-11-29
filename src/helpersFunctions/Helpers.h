//
// Created by reverse-proxy on 16‏/3‏/2020.
//

#ifndef KERNELO_HELPERS_H
#define KERNELO_HELPERS_H

#include <armadillo>

using namespace arma;

namespace Helpers{

    double logSumExp(const vec &elements);
    double computeDeterminant(const mat& matrix);
    mat inverseMatrix(const mat& matrix);
    double weightedLogSumExp(const double & log_p1, const double & log_p2, const unsigned & c1, const unsigned & c2);
    vec dmvnrm_arma_fast_chol(mat const &x, rowvec const &mean, mat &sigma, bool const logd = true);
    double mvnrm_arma_fast_chol(rowvec const &x, rowvec const &mean, mat &sigma, bool const logd = true);
    mat safe_cholesky(mat & Sigma);
    // arma::vec Mahalanobis(arma::mat const &x, arma::vec const &center, arma::mat const &cov);
}

#endif //KERNELO_HELPERS_H
