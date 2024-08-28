#include "covariance.hpp"

// ==================== Constructors ====================

FullCovariance::FullCovariance(const mat &cov) : covariances_(cov) {}

FullCovariance::FullCovariance(unsigned dimension) : covariances_(mat(dimension, dimension, fill::eye)) {}

// ==================== Class methods ====================

double FullCovariance::log_det() const
{
    return log_det_sympd(covariances_);
}

FullCovariance FullCovariance::inv() const
{
    mat inv = inv_sympd(covariances_);
    return FullCovariance(inv);
}

void FullCovariance::rank_one_update(const vec &v, double alpha)
{
    for (unsigned i = 0; i < v.n_rows; i++)
    {
        covariances_.col(i) += v * v(i) * alpha;
    }
}

void FullCovariance::fill(const double scalar)
{
    covariances_.fill(scalar);
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
    mat result = y + x.get_mat();
    return result;
}

mat operator+(const FullCovariance &x, const mat &y)
{
    mat result = y + x.get_mat();
    return result;
}

mat operator-(const mat &y, const FullCovariance &x)
{
    return y - x.get_mat();
}

mat operator-(const FullCovariance &x, const mat &y)
{
    return x.get_mat() - y;
}

mat operator*(const mat &y, const FullCovariance &x)
{
    mat result = y * x.get_mat();
    return result;
}

mat operator*(const arma::subview_cols<double> &y, const FullCovariance &x)
{
    mat result = y * x.get_mat();
    return result;
}

mat operator*(const FullCovariance &x, const mat &y)
{
    mat result = x.get_mat() * y;
    return result;
}

vec operator*(const FullCovariance &x, const vec &y)
{
    vec result = x.get_mat() * y;
    return result;
}

rowvec operator*(const rowvec &y, const FullCovariance &x)
{
    rowvec result = y * x.get_mat();
    return result;
}
