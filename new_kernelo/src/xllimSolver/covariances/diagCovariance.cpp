#include "covariance.hpp"

// ==================== Constructors ====================

DiagCovariance::DiagCovariance(const vec &var) : variances(var) {}

DiagCovariance::DiagCovariance(const arma::subview_row<double> &var) : variances(var.t()) {}

DiagCovariance::DiagCovariance(const mat &cov) : variances(cov.diag()) {}

DiagCovariance::DiagCovariance(unsigned dimension) : variances(vec(dimension, fill::zeros)) {}

// ==================== Class methods ====================

double DiagCovariance::log_det() const
{
    return sum(log(this->variances));
}

DiagCovariance DiagCovariance::inv() const
{
    vec inv = 1.0 / this->variances;
    return DiagCovariance(inv);
}

void DiagCovariance::rankOneUpdate(const vec &v, double alpha) const
{
    // TODO
    // variances += pow(v, 2) * alpha;
}

void DiagCovariance::print(const std::string &str) const
{
    variances.t().print(str);
}

void DiagCovariance::print() const
{
    variances.t().print();
}

mat DiagCovariance::get_mat() const
{
    return diagmat(this->variances);
}

vec DiagCovariance::get_vec() const
{
    return this->variances;
}

// ==================== Assignement operators ====================

DiagCovariance &DiagCovariance::operator=(const DiagCovariance &cov)
{
    this->variances = cov.variances;
    return *this;
}

DiagCovariance &DiagCovariance::operator=(const mat &cov)
{
    this->variances = cov.diag();
    return *this;
}

DiagCovariance &DiagCovariance::operator=(const vec &var)
{
    this->variances = var;
    return *this;
}

DiagCovariance &DiagCovariance::operator+=(const mat &cov)
{
    this->variances += cov.diag();
    return *this;
}

DiagCovariance &DiagCovariance::operator+=(double scalar)
{
    this->variances += scalar;
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
