#include "covariance.hpp"

// ==================== Constructors ====================

FullCovariance::FullCovariance(const mat &cov) : covariances(cov) {}

FullCovariance::FullCovariance(unsigned dimension) : covariances(mat(dimension, dimension, fill::eye)) {}

// ==================== Class methods ====================

double FullCovariance::log_det() const
{
    return log_det_sympd(this->covariances);
}

FullCovariance FullCovariance::inv() const
{
    mat inv = inv_sympd(this->covariances);
    return FullCovariance(inv);
}

void FullCovariance::rank_one_update(const vec &v, double alpha)
{
    for (unsigned i = 0; i < v.n_rows; i++)
    {
        this->covariances.col(i) += v * v(i) * alpha;
    }
}

void FullCovariance::fill(const double scalar)
{
    this->covariances.fill(scalar);
}

void FullCovariance::print(const std::string &str) const
{
    this->covariances.print(str);
}

void FullCovariance::print() const
{
    this->covariances.print();
}

mat FullCovariance::get_mat() const
{
    return this->covariances;
}

// ==================== Assignement operators ====================

FullCovariance &FullCovariance::operator=(const FullCovariance &cov)
{
    this->covariances = cov.covariances;
    return *this;
}

FullCovariance &FullCovariance::operator=(const mat &cov)
{
    this->covariances = cov;
    return *this;
}

FullCovariance &FullCovariance::operator+=(const mat &cov)
{
    this->covariances += cov;
    return *this;
}

FullCovariance &FullCovariance::operator+=(double scalar)
{
    this->covariances += scalar;
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
