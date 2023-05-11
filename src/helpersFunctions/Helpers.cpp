//
// Created by reverse-proxy on 16‏/3‏/2020.
//

#include "Helpers.h"
// #include <omp.h>

using namespace Helpers;

static double const log2pi = std::log(2.0 * M_PI);

double Helpers::logSumExp(const arma::vec & elements) {
    double result = 0;
    double max = elements.max();

    if(max == -datum::inf){
        return max;
    }else{
        for(unsigned i=0; i<elements.n_rows; i++){
            result += exp(elements(i) - max);
        }
        result = log(result) + max;
        return result;
    }
}


double Helpers::computeDeterminant(const mat& matrix){
    if(matrix.n_rows <= 3){
        return det(matrix);
    }else{
        mat R;
        if(chol(R,matrix)){
            return 2 * sum(log(R.diag()));
        }else{
            return det(matrix);
        }
    }

}

mat Helpers::inverseMatrix(const mat& matrix){
    if(matrix.n_rows <= 3){
        return inv(matrix);
    }else {
        mat R;
        if(chol(R,matrix)){
            mat R_inv = inv(chol(matrix));
            return R_inv * R_inv.t();
        }else{
            return inv(matrix);
        }
    }
}

/* performs the operation log(c1 * exp(log_p1) + c2 * exp(log_p2)) with numerical stability */
double Helpers::weightedLogSumExp(
    const double & log_p1, const double & log_p2, const unsigned & c1, const unsigned & c2) {
    
    double result;
    double m(std::max(log_p1, log_p2));
    if(m == -datum::inf){
        return m;
    }else{
        result = log(c1 * exp(log_p1 - m) + c2 * exp(log_p2 - m)) + m;
        return result;
    }
}

/* C++ version of the dtrmv BLAS function */
void inplace_tri_mat_mult(arma::rowvec &x, arma::mat const &trimat){
  arma::uword const n = trimat.n_cols;
  
  for(unsigned j = n; j-- > 0;){
    double tmp(0.);
    for(unsigned i = 0; i <= j; ++i)
      tmp += trimat.at(i, j) * x[i];
    x[j] = tmp;
  }
}

/* The Multivariate Normal density function */
vec Helpers::dmvnrm_arma_fast_chol(arma::mat const &x,arma::rowvec const &mean, arma::mat &chol, bool const logd /*= true*/) { 
    using arma::uword;
    uword const n = x.n_rows, 
                xdim = x.n_cols;
    arma::vec out(n);
    // arma::mat const rooti = arma::inv(trimatu(Helpers::safe_cholesky(sigma)));
    arma::mat const rooti = arma::inv(chol);
    double const rootisum = arma::sum(log(rooti.diag())), 
                constants = -(double)xdim/2.0 * log2pi, 
              other_terms = rootisum + constants;
    
    arma::rowvec z;
    // #pragma omp parallel for schedule(static) private(z)
    for (uword i = 0; i < n; i++) {
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
mat Helpers::safe_cholesky(mat & Sigma){
    mat Chol(arma::size(Sigma));
    bool success = false;
    while (success == false)
    {
        success = arma::chol(Chol, Sigma);
        if(success == false)
            {
            Sigma += eye(Sigma.n_rows,Sigma.n_rows) * 1e-8;
            // success = true;
            }
    }
    return Chol;
}

// arma::vec Mahalanobis(arma::mat const &x, arma::vec const &center, arma::mat const &cov) {
//     arma::mat x_cen = x.t();
//     x_cen.each_col() -= center;
//     arma::solve(x_cen, arma::trimatl(chol(cov).t()), x_cen);
//     x_cen.for_each( [](arma::mat::elem_type& val) { val = val * val; } );
//     return arma::sum(x_cen, 0).t();    
// }