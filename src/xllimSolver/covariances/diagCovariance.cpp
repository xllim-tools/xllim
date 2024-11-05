#include "covariance.hpp"

// ==================== Constructors ====================

DiagCovariance::DiagCovariance(const vec &var) : variances_(var) {}

DiagCovariance::DiagCovariance(const arma::subview_col<double> &var) : variances_(var) {}

DiagCovariance::DiagCovariance(const arma::subview_row<double> &var) : variances_(var.t()) {}

DiagCovariance::DiagCovariance(const mat &cov) : variances_(cov.diag()) {}

DiagCovariance::DiagCovariance(unsigned dimension) : variances_(vec(dimension, fill::ones)) {}

// ==================== Class methods ====================

double DiagCovariance::log_det() const
{
    return sum(log(variances_));
}

DiagCovariance DiagCovariance::inv() const
{
    vec inv = 1.0 / variances_;
    return DiagCovariance(inv);
}

void DiagCovariance::rank_one_update(const vec &v, double alpha)
{
    variances_ += pow(v, 2) * alpha;
}

void DiagCovariance::fill(const double scalar)
{
    variances_.fill(scalar);
}

void DiagCovariance::print(const std::string &str) const
{
    variances_.t().print(str);
}

void DiagCovariance::print() const
{
    variances_.t().print();
}

DiagCovariance DiagCovariance::head(unsigned L_t) const
{
    return DiagCovariance(variances_.head(L_t));
}

DiagCovariance DiagCovariance::tail(unsigned L_w) const
{
    return DiagCovariance(variances_.tail(L_w));
}

mat DiagCovariance::get_mat() const
{
    return diagmat(variances_);
}

vec DiagCovariance::get_vec() const
{
    return variances_;
}

// ==================== Assignement operators ====================

DiagCovariance &DiagCovariance::operator=(const DiagCovariance &cov)
{
    variances_ = cov.variances_;
    return *this;
}

DiagCovariance &DiagCovariance::operator=(const mat &cov)
{
    variances_ = cov.diag();
    return *this;
}

DiagCovariance &DiagCovariance::operator=(const vec &var)
{
    variances_ = var;
    return *this;
}

DiagCovariance &DiagCovariance::operator=(const arma::subview_row<double> &var)
{
    variances_ = var.t();
    return *this;
}

DiagCovariance &DiagCovariance::operator+=(const mat &cov)
{
    variances_ += cov.diag();
    return *this;
}

DiagCovariance &DiagCovariance::operator+=(double scalar)
{
    variances_ += scalar;
    return *this;
}

// ==================== Arithmetic operators ====================

mat operator+(const mat &y, const DiagCovariance &x)
{
    mat result = y + x.get_mat();
    ;
    return result;
}

mat operator+(const DiagCovariance &x, const mat &y)
{
    mat result = y + x.get_mat();
    return result;
}

mat operator-(const mat &y, const DiagCovariance &x)
{
    mat result = y - x.get_mat();
    return result;
}

mat operator-(const DiagCovariance &x, const mat &y)
{
    mat result = x.get_mat() - y;
    return result;
}

mat operator*(const mat &y, const DiagCovariance &x)
{
    mat result = mat(y.n_rows, y.n_cols);
    for (unsigned i = 0; i < y.n_rows; i++)
    {
        result.row(i) = y.row(i) % x.get_vec().t();
    }
    return result;
}

mat operator*(const arma::subview_cols<double> &y, const DiagCovariance &x)
{
    mat result = mat(y.n_rows, y.n_cols);
    for (unsigned i = 0; i < y.n_rows; i++)
    {
        result.row(i) = y.row(i) % x.get_vec().t();
    }
    return result;
}

mat operator*(const DiagCovariance &x, const mat &y)
{
    mat result = mat(y.n_rows, y.n_cols);
    for (unsigned i = 0; i < y.n_cols; i++)
    {
        result.col(i) = y.col(i) % x.get_vec();
    }
    return result;
}

vec operator*(const DiagCovariance &x, const vec &y)
{
    vec result = x.get_vec() % y;
    return result;
}

rowvec operator*(const rowvec &y, const DiagCovariance &x)
{
    rowvec result = y % x.get_vec().t();
    return result;
}
