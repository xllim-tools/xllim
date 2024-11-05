#include "covariance.hpp"

// ==================== Constructors ====================

IsoCovariance::IsoCovariance(double variance, unsigned dimension) : scalar_(variance), size_(dimension) {}

IsoCovariance::IsoCovariance(const mat &covariance) : scalar_(accu(covariance.diag()) / covariance.n_cols), size_(covariance.n_cols) {}

IsoCovariance::IsoCovariance(unsigned dimension) : scalar_(1.0), size_(dimension) {}

// ==================== Class methods ====================

double IsoCovariance::log_det() const
{
    return log(scalar_ * size_);
}

IsoCovariance IsoCovariance::inv() const
{
    double inv = 1.0 / scalar_;
    return IsoCovariance(inv, size_);
}

void IsoCovariance::rank_one_update(const vec &v, double alpha)
{
    scalar_ += alpha * accu(pow(v, 2)) / size_;
}

void IsoCovariance::rank_one_update_head(unsigned L_t, const vec &v, double alpha)
{
    // This case should not be used. If Covariance is Isotropic than covariances keep the same. The observed/latent dimension is not considered here (L_t=L).
    scalar_ += alpha * accu(pow(v, 2)) / size_;
}

void IsoCovariance::fill(const double scalar)
{
    scalar_ = scalar;
}

void IsoCovariance::fill_head(unsigned L_t, const double scalar)
{
    // This case should not be used. If Covariance is Isotropic than covariances keep the same. The observed/latent dimension is not considered here (L_t=L).
    scalar_ = scalar;
}

void IsoCovariance::print(const std::string &str) const
{
    std::cout << str << "\n\tVariance = " << std::to_string(scalar_) << ", Dimension = " << std::to_string(size_) << std::endl;
}

void IsoCovariance::print() const
{
    std::cout << "Variance = " << std::to_string(scalar_) << ", Dimension = " << std::to_string(size_) << std::endl;
}

IsoCovariance IsoCovariance::head(unsigned L_t) const
{
    return IsoCovariance(scalar_, L_t);
}

IsoCovariance IsoCovariance::tail(unsigned L_w) const
{
    return IsoCovariance(scalar_, L_w);
}

mat IsoCovariance::get_mat() const
{
    vec diag(size_, fill::value(scalar_));
    return diagmat(diag);
}

vec IsoCovariance::get_vec() const
{
    return vec(size_, fill::value(scalar_));
}

double IsoCovariance::get_scalar() const
{
    return scalar_;
}

unsigned IsoCovariance::get_size() const
{
    return size_;
}

// ==================== Assignement operators ====================

IsoCovariance &IsoCovariance::operator=(const IsoCovariance &cov)
{
    scalar_ = cov.scalar_;
    size_ = cov.size_;
    return *this;
}

IsoCovariance &IsoCovariance::operator=(const mat &cov)
{
    scalar_ = accu(cov.diag()) / cov.n_cols;
    size_ = cov.n_cols;
    return *this;
}

IsoCovariance &IsoCovariance::operator=(const arma::subview_col<double> &cov_scalar)
{
    scalar_ = accu(cov_scalar) / cov_scalar.n_rows;
    return *this;
}

IsoCovariance &IsoCovariance::operator=(double scalar)
{
    scalar_ = scalar;
    return *this;
}

IsoCovariance &IsoCovariance::operator+=(const mat &cov)
{
    scalar_ += accu(cov.diag()) / cov.n_cols;
    return *this;
}

IsoCovariance &IsoCovariance::operator+=(double scalar)
{
    scalar_ += scalar;
    return *this;
}

// ==================== Arithmetic operators ====================

mat operator+(const mat &y, const IsoCovariance &x)
{
    mat result = y + x.get_mat();
    return result;
}

mat operator+(const IsoCovariance &x, const mat &y)
{
    mat result = y + x.get_mat();
    return result;
}

mat operator-(const mat &y, const IsoCovariance &x)
{
    mat result = y - x.get_mat();
    return result;
}

mat operator-(const IsoCovariance &x, const mat &y)
{
    mat result = x.get_mat() - y;
    return result;
}

mat operator*(const mat &y, const IsoCovariance &x)
{
    mat result = y * x.get_scalar();
    return result;
}

mat operator*(const arma::subview_cols<double> &y, const IsoCovariance &x)
{
    mat result = y * x.get_scalar();
    return result;
}

mat operator*(const IsoCovariance &x, const mat &y)
{
    mat result = y * x.get_scalar();
    return result;
}

vec operator*(const IsoCovariance &x, const vec &y)
{
    vec result = x.get_scalar() * y;
    return result;
}

rowvec operator*(const rowvec &y, const IsoCovariance &x)
{
    rowvec result = y * x.get_scalar();
    return result;
}
