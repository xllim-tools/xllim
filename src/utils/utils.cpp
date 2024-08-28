#include "utils.hpp"

using namespace utils;

static double const log2pi = std::log(2.0 * M_PI);

/* performs the operation log( Σ exp(logx_i) ) with numerical stability */
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

/* performs the operation log( Σ exp(logx_i) ) with numerical stability */
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

/* performs the operation log(c1 * exp(log_p1) + c2 * exp(log_p2)) with numerical stability */
vec utils::weightedLogSumExp(const unsigned &c1, const vec &log_p1, const unsigned &c2, const vec &log_p2)
{
    // Ensure the input vectors are of the same size
    if (log_p1.n_elem != log_p2.n_elem)
    {
        throw std::invalid_argument("Input vectors must have the same length.");
    }
    vec result(log_p1.n_elem, fill::value(-datum::inf));
    vec max_log_p = max(log_p1, log_p2);

    // Check for -inf in max_log_p
    uvec is_not_inf = find(max_log_p != -datum::inf);

    // Handle elements where max_log_p was -inf by setting them to -inf
    result.elem(is_not_inf) = log(c1 * exp(log_p1 - max_log_p) + c2 * exp(log_p2 - max_log_p)) + max_log_p;
    return result;
}

mat utils::proposition_covariance(gmm_full &gmm)
{
    unsigned L = gmm.means.n_rows;
    unsigned K = gmm.hefts.n_cols;

    vec mean_mean_mixture(L);
    mat mean_cov_mixture(L, L);

    for (unsigned k = 0; k < K; k++)
    {
        mean_mean_mixture += gmm.hefts(k) * gmm.means.col(k);
        mean_cov_mixture += (gmm.fcovs.slice(k) + gmm.means.col(k) * gmm.means.col(k).t()) * gmm.hefts(k);
    }
    mean_cov_mixture -= mean_mean_mixture * mean_mean_mixture.t();

    return mean_cov_mixture;
}

vec utils::Mahalanobis(const mat &x, const vec &center, const mat &cov)
{
    mat x_centered = x;
    x_centered.each_col() -= center;
    solve(x_centered, trimatl(chol(cov).t()), x_centered);
    x_centered.for_each([](mat::elem_type &val)
                        { val = val * val; });
    return sum(x_centered, 0).t();
}

vec utils::MahalanobisWithInvertedCov(const mat &x, const vec &center, const mat &inverted_cov)
{
    unsigned N = x.n_cols;
    vec mahalanobis_dist(N);
    for (unsigned n = 0; n < N; n++)
    {
        vec x_tmp = x.col(n) - center;
        vec Cov_x_tmp = inverted_cov * x_tmp;
        mahalanobis_dist(n) = dot(x_tmp, Cov_x_tmp);
    }
    return mahalanobis_dist;
}

// double utils::logDensity(const vec &x, const vec &mean, const mat &covariance)
// {
//     mat chol(covariance);
//     return utils::mvnrm_arma_fast_chol(x.t(), mean.t(), chol, true);
// }

// // diag covariance
// double utils::logDensity(const vec &x, const vec &mean, const vec &diag_covariance)
// {
//     using arma::uword;
//     uword const xdim = x.n_cols;
//     double out;
//     arma::vec const rooti = 1/sqrt(diag_covariance);
//     double const rootisum = arma::sum(log(rooti)),
//                  constants = -(double)xdim / 2.0 * log2pi,
//                  other_terms = rootisum + constants;

//     vec z = (x - mean) % rooti;
//     out = other_terms - 0.5 * arma::dot(z, z);
//     return out;
// }

// double utils::logDensity(const vec &x, const vec &weight, const mat &mean, const cube &covariance)
// {
//     // TODO: check if diamat with .is_diagmat() and apply woodbury
//     return 0.0;
// }

// mat utils::logDensity(const mat &x, const rowvec &weight, const mat &mean, const cube &covariance)
// {
//     // TODO: check if diamat with .is_diagmat() and apply woodbury
//     cube chol(covariance);
//     mat results(x.n_cols, weight.n_cols);
//     #pragma omp parallel for
//     for (unsigned k = 0; k < weight.n_cols; k++)
//     {
//         vec density_k = utils::dmvnrm_arma_fast_chol(x.t(), mean.col(k).t(), chol.slice(k), true);
//         results.col(k) = log(weight(k)) + density_k;
//     }
//     return results;
// }

// // diag covariance
// mat utils::logDensity(const mat &x, const rowvec &weight, const mat &mean, const mat &covariance)
// {
//     // TODO: check if diamat with .is_diagmat() and apply woodbury
//     // cube chol(covariance);
//     mat results(x.n_cols, weight.n_cols);
//     #pragma omp parallel for
//     for (unsigned k = 0; k < weight.n_cols; k++)
//     {
//         // mat cov = arma::diagmat(covariance.col(k));
//         vec density_k = utils::dmvnrm_arma_fast_chol_diag(x.t(), mean.col(k).t(), covariance.col(k), true);
//         // density_k.print("density_k");
//         results.col(k) = log(weight(k)) + density_k;
//     }
//     return results;
// }

// /* C++ version of the dtrmv BLAS function */
// void utils::inplace_tri_mat_mult(arma::rowvec &x, arma::mat const &trimat)
// {
//     arma::uword const n = trimat.n_cols;

//     for (unsigned j = n; j-- > 0;)
//     {
//         double tmp(0.);
//         for (unsigned i = 0; i <= j; ++i)
//             tmp += trimat.at(i, j) * x[i];
//         x[j] = tmp;
//     }
// }

// /* The Multivariate Normal density function */
// double utils::mvnrm_arma_fast_chol(arma::rowvec const &x, arma::rowvec const &mean, arma::mat &chol, bool const logd /*= true*/)
// {
//     using arma::uword;
//     uword const xdim = x.n_cols;
//     double out;
//     // arma::mat const rooti = arma::inv(arma::trimatu(safe_cholesky(chol)));
//     arma::mat const rooti = arma::inv(arma::trimatu(arma::chol(chol)));
//     double const rootisum = arma::sum(log(rooti.diag())),
//                  constants = -(double)xdim / 2.0 * log2pi,
//                  other_terms = rootisum + constants;

//     // #pragma omp parallel for schedule(static) private(z)
//     arma::rowvec z = (x - mean);
//     inplace_tri_mat_mult(z, rooti);
//     out = other_terms - 0.5 * arma::dot(z, z);

//     if (logd)
//         return out;
//     return exp(out);
// }

// /* The Multivariate Normal density function */
// vec utils::dmvnrm_arma_fast_chol(arma::mat const &x, arma::rowvec const &mean, arma::mat &chol, bool const logd /*= true*/)
// {
//     using arma::uword;
//     uword const n = x.n_rows,
//                 xdim = x.n_cols;
//     arma::vec out(n);
//     // arma::mat const rooti = arma::inv(trimatu(utils::safe_cholesky(chol)));
//     arma::mat const rooti = arma::inv(arma::trimatu(arma::chol(chol)));
//     double const rootisum = arma::sum(log(rooti.diag())),
//                  constants = -(double)xdim / 2.0 * log2pi,
//                  other_terms = rootisum + constants;

//     arma::rowvec z;
// #pragma omp parallel for schedule(static) private(z)
//     for (uword i = 0; i < n; i++)
//     {
//         z = (x.row(i) - mean);
//         inplace_tri_mat_mult(z, rooti);
//         out(i) = other_terms - 0.5 * arma::dot(z, z);
//     }

//     if (logd)
//         return out;
//     return exp(out);
// }

// /* The Multivariate Normal density function */
// vec utils::dmvnrm_arma_fast_chol_diag(arma::mat const &x, arma::rowvec const &mean, const arma::vec &chol, bool const logd /*= true*/)
// {
//     using arma::uword;
//     uword const n = x.n_rows,
//                 xdim = x.n_cols;
//     arma::vec out(n);
//     // arma::mat const rooti = arma::inv(trimatu(utils::safe_cholesky(chol)));
//     arma::vec const rooti = 1/sqrt(chol);
//     double const rootisum = arma::sum(log(rooti)),
//                  constants = -(double)xdim / 2.0 * log2pi,
//                  other_terms = rootisum + constants;

//     arma::rowvec z;
// #pragma omp parallel for// schedule(static) private(z)
//     for (uword i = 0; i < n; i++)
//     {
//         z = (x.row(i) - mean) % rooti.t();
//         // inplace_tri_mat_mult(z, rooti);
//         out(i) = other_terms - 0.5 * arma::dot(z, z);
//     }

//     if (logd)
//         return out;
//     return exp(out);
// }

// /* Perfoms a cholesky decomposition; if needed add a diagonal regularization term
//     to increase numerical stability. */
// mat utils::safe_cholesky(mat &Sigma)
// {
//     mat Chol(arma::size(Sigma));
//     bool success = false;
//     while (success == false)
//     {
//         success = arma::chol(Chol, Sigma);
//         if (success == false)
//         {
//             Sigma += eye(Sigma.n_rows, Sigma.n_rows) * 1e-8;
//             // success = true;
//         }
//     }
//     return Chol;
// }
