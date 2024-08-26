#include "covariance.hpp"

// ==================== Constructors ====================

IsoCovariance::IsoCovariance(double variance, unsigned dimension) : scalar(variance), size(dimension) {}

IsoCovariance::IsoCovariance(const mat &covariance) : scalar(accu(covariance.diag()) / covariance.n_cols), size(covariance.n_cols) {}

IsoCovariance::IsoCovariance(unsigned dimension) : scalar(1.0), size(dimension) {}

// ==================== Class methods ====================

double IsoCovariance::log_det() const
{
    return log(this->scalar * this->size);
}

IsoCovariance IsoCovariance::inv() const
{
    double inv = 1.0 / this->scalar;
    return IsoCovariance(inv, this->size);
}

void IsoCovariance::rank_one_update(const vec &v, double alpha)
{
    this->scalar += alpha * accu(pow(v, 2)) / size;
}

void IsoCovariance::fill(const double scalar)
{
    this->scalar = scalar;
}

void IsoCovariance::print(const std::string &str) const
{
    std::cout << str << "\n\tVariance = " << std::to_string(this->scalar) << ", Dimension = " << std::to_string(this->size) << std::endl;
}

void IsoCovariance::print() const
{
    std::cout << "Variance = " << std::to_string(this->scalar) << ", Dimension = " << std::to_string(this->size) << std::endl;
}

IsoCovariance IsoCovariance::head(unsigned L_t) const
{
    return IsoCovariance(scalar, L_t);
}

IsoCovariance IsoCovariance::tail(unsigned L_w) const
{
    return IsoCovariance(scalar, L_w);
}

mat IsoCovariance::get_mat() const
{
    vec diag(this->size, fill::value(this->scalar));
    return diagmat(diag);
}

vec IsoCovariance::get_vec() const
{
    return vec(this->size, fill::value(this->scalar));
}

double IsoCovariance::get_scalar() const
{
    return this->scalar;
}

unsigned IsoCovariance::get_size() const
{
    return this->size;
}

// ==================== Assignement operators ====================

IsoCovariance &IsoCovariance::operator=(const IsoCovariance &cov)
{
    this->scalar = cov.scalar;
    this->size = cov.size;
    return *this;
}

IsoCovariance &IsoCovariance::operator=(const mat &cov)
{
    this->scalar = accu(cov.diag()) / cov.n_cols;
    this->size = cov.n_cols;
    return *this;
}

IsoCovariance &IsoCovariance::operator=(double scalar)
{
    this->scalar = scalar;
    return *this;
}

IsoCovariance &IsoCovariance::operator+=(const mat &cov)
{
    this->scalar += accu(cov.diag()) / cov.n_cols;
    return *this;
}

IsoCovariance &IsoCovariance::operator+=(double scalar)
{
    this->scalar += scalar;
    return *this;
}

// ==================== Arithmetic operators ====================

mat operator+(const mat &y, const IsoCovariance &x)
{
    mat result = y + x.get_mat();
    ;
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
