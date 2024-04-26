#include "utils.hpp"

using namespace utils;

static double const log2pi = std::log(2.0 * M_PI);

double utils::logSumExp(const vec &x)
{
    double max = x.max();
    if (max == -datum::inf)
    {
        return max;
    }
    else
    {
        return max + log(sum(exp(x - max)));
    }
}

// TODO essayer avec template<typename T> pour que T soit vec ou rowvec
vec utils::logSumExp(const mat &x, const int axis = 0)
{
    // Transform x according to axis direction
    mat x_shifted(x);
    if (axis == 1)
    {
        x_shifted = x.t();
    }

    // Subtract the largest in each column
    rowvec x_max = max(x_shifted);
    x_shifted = x_shifted.each_row() - x_max;
    rowvec s = x_max + log(sum(exp(x_shifted)));

    // Handle numerical issues
    uvec i = find_nonfinite(x_max);
    if (i.n_elem > 0)
    {
        s.elem(i) = x_max.elem(i);
    }

    return s.t();
}

/* performs the operation log(c1 * exp(log_p1) + c2 * exp(log_p2)) with numerical stability */
double utils::weightedLogSumExp(
    const double &log_p1, const double &log_p2, const unsigned &c1, const unsigned &c2)
{

    double result;
    double m(std::max(log_p1, log_p2));
    if (m == -datum::inf)
    {
        return m;
    }
    else
    {
        result = log(c1 * exp(log_p1 - m) + c2 * exp(log_p2 - m)) + m;
        return result;
    }
}

double utils::logDensity(const vec &x, const vec &mean, const mat &covariance)
{
    return 0.0;
}

double utils::logDensity(const vec &x, const vec &weight, const mat &mean, const cube &covariance)
{
    // TODO: check if diamat with .is_diagmat() and apply woodbury
    return 0.0;
}

mat utils::logDensity(const mat &x, const vec &weight, const mat &mean, const cube &covariance)
{
    // TODO: check if diamat with .is_diagmat() and apply woodbury
    return mat(x.n_cols, weight.n_cols, fill::value(222));
}

/* C++ version of the dtrmv BLAS function */
void inplace_tri_mat_mult(arma::rowvec &x, arma::mat const &trimat)
{
    arma::uword const n = trimat.n_cols;

    for (unsigned j = n; j-- > 0;)
    {
        double tmp(0.);
        for (unsigned i = 0; i <= j; ++i)
            tmp += trimat.at(i, j) * x[i];
        x[j] = tmp;
    }
}

/* The Multivariate Normal density function */
vec dmvnrm_arma_fast_chol(arma::mat const &x, arma::rowvec const &mean, arma::mat &chol, bool const logd /*= true*/)
{
    using arma::uword;
    uword const n = x.n_rows,
                xdim = x.n_cols;
    arma::vec out(n);
    // arma::mat const rooti = arma::inv(trimatu(Helpers::safe_cholesky(sigma)));
    arma::mat const rooti = arma::inv(chol);
    double const rootisum = arma::sum(log(rooti.diag())),
                 constants = -(double)xdim / 2.0 * log2pi,
                 other_terms = rootisum + constants;

    arma::rowvec z;
    // #pragma omp parallel for schedule(static) private(z)
    for (uword i = 0; i < n; i++)
    {
        z = (x.row(i) - mean);
        inplace_tri_mat_mult(z, rooti);
        out(i) = other_terms - 0.5 * arma::dot(z, z);
    }

    if (logd)
        return out;
    return exp(out);
}

/* Perfoms a cholesky decomposition; if needed add a diagonal regularization term
    to increase numerical stability. */
mat safe_cholesky(mat &Sigma)
{
    mat Chol(arma::size(Sigma));
    bool success = false;
    while (success == false)
    {
        success = arma::chol(Chol, Sigma);
        if (success == false)
        {
            Sigma += eye(Sigma.n_rows, Sigma.n_rows) * 1e-8;
            // success = true;
        }
    }
    return Chol;
}
