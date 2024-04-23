#ifndef UTILS_HPP
#define UTILS_HPP

namespace utils {
    double logSumExp(const vec &elements);
    vec logSumExp(const mat &elements);
    double weightedLogSumExp(const double &log_p1, const double &log_p2, const unsigned &c1, const unsigned &c2);
    double logDensity(const mat &elements);
    vec logDensity(const mat &elements);
    // arma::vec Mahalanobis(arma::mat const &x, arma::vec const &center, arma::mat const &cov);
}

#endif // UTILS_HPP
