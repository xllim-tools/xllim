#include "covariance.hpp"

// ==================== Constructors ====================

FullCovariance::FullCovariance(const mat &cov) : covariances_(cov) {}

FullCovariance::FullCovariance(unsigned dimension) : covariances_(mat(dimension, dimension, fill::eye)) {}

// ==================== Class methods ====================

double FullCovariance::log_det() const
{
    double result;
    bool success = arma::log_det_sympd(result, covariances_);
    if (success)
    {
        return result;
    }
    else
    {
        return arma::log_det(covariances_).real();
    }
}

FullCovariance FullCovariance::inv() const
{
    mat inv;
    bool success = arma::inv_sympd(inv, covariances_);
    if (success)
    {
        return FullCovariance(inv);
    }
    else
    {
        return FullCovariance(arma::inv(covariances_));
    }
}

void FullCovariance::rank_one_update(const vec &v, double alpha)
{
    for (unsigned i = 0; i < v.n_rows; i++)
    {
        covariances_.col(i) += v * v(i) * alpha;
    }
}

void FullCovariance::rank_one_update_head(unsigned L_t, const vec &v, double alpha)
{
    for (unsigned i = 0; i < v.n_rows; i++)
    {
        covariances_.submat(0, 0, L_t - 1, L_t - 1).col(i) += v * v(i) * alpha;
    }
}

void FullCovariance::fill(const double scalar)
{
    covariances_.fill(scalar);
}

void FullCovariance::fill_head(unsigned L_t, const double scalar)
{
    covariances_.submat(0, 0, L_t - 1, L_t - 1).fill(scalar);
}

void FullCovariance::print(const std::string &str) const
{
    covariances_.print(str);
}

void FullCovariance::print() const
{
    covariances_.print();
}

FullCovariance FullCovariance::head(unsigned L_t) const
{
    return FullCovariance(covariances_.submat(0, 0, L_t - 1, L_t - 1));
}

FullCovariance FullCovariance::tail(unsigned L_w) const
{
    return FullCovariance(covariances_.submat(covariances_.n_rows - L_w, covariances_.n_rows - L_w, covariances_.n_rows - 1, covariances_.n_rows - 1));
}

mat FullCovariance::get_mat() const
{
    return covariances_;
}

// ==================== Assignement operators ====================

FullCovariance &FullCovariance::operator=(const FullCovariance &cov)
{
    covariances_ = cov.covariances_;
    return *this;
}

FullCovariance &FullCovariance::operator=(const mat &cov)
{
    covariances_ = cov;
    return *this;
}

FullCovariance &FullCovariance::operator+=(const mat &cov)
{
    covariances_ += cov;
    return *this;
}

FullCovariance &FullCovariance::operator+=(double scalar)
{
    covariances_ += scalar;
    return *this;
}

// ==================== Arithmetic operators ====================

mat operator+(const mat &y, const FullCovariance &x)
{
    mat result = y + x.covariances_;
    return result;
}

mat operator+(const FullCovariance &x, const mat &y)
{
    mat result = y + x.covariances_;
    return result;
}

mat operator-(const mat &y, const FullCovariance &x)
{
    return y - x.covariances_;
}

mat operator-(const FullCovariance &x, const mat &y)
{
    return x.covariances_ - y;
}

mat operator*(const mat &y, const FullCovariance &x)
{
    mat result = y * x.covariances_;
    return result;
}

mat operator*(const arma::subview_cols<double> &y, const FullCovariance &x)
{
    mat result = y * x.covariances_;
    return result;
}

mat operator*(const FullCovariance &x, const mat &y)
{
    mat result = x.covariances_ * y;
    return result;
}

vec operator*(const FullCovariance &x, const vec &y)
{
    vec result = x.covariances_ * y;
    return result;
}

rowvec operator*(const rowvec &y, const FullCovariance &x)
{
    rowvec result = y * x.covariances_;
    return result;
}
