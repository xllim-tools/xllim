/**
 * @file IsoCovariance.cpp
 * @brief IsoCovairance class implementation
 * @author Sami DJOUADI
 * @version 1.1
 * @date 26/02/2020
 */

#include "Icovariance.h"

using namespace learningModel;

IsoCovariance::IsoCovariance(double scalar, unsigned size){
    this->scalar = scalar;
    this->size = size;
}

IsoCovariance::IsoCovariance(const mat &covariance){
    this->scalar = accu(covariance.diag()) / covariance.n_cols;
    this->size = covariance.n_cols;
}

IsoCovariance::IsoCovariance(unsigned dimension) {
    scalar = 0;
    size = dimension;
}

IsoCovariance &IsoCovariance::operator=(const IsoCovariance &cov) {
    scalar = cov.scalar;
    size = cov.size;
    return *this;
}

IsoCovariance &IsoCovariance::operator=(const arma::mat & cov) {
    this->scalar = accu(cov.diag()) / cov.n_cols;
    this->size = cov.n_cols;
    return *this;
}

IsoCovariance &IsoCovariance::operator=(double scalar) {
    this->scalar = scalar;
    return *this;
}

IsoCovariance &IsoCovariance::operator+=(double scalar) {
    this->scalar += scalar;
}

IsoCovariance &IsoCovariance::operator+=(const arma::mat & cov) {
    scalar += accu(cov.diag()) / cov.n_cols;
}

mat learningModel::operator+(const mat &y, const IsoCovariance &x) {
    mat result = y;
    result.diag() += x.scalar;
    return result;
}

mat learningModel::operator+(const IsoCovariance &x, const mat &y) {
    return y + x;
}

mat learningModel::operator*(const mat &y, const IsoCovariance &x) {
    return y * x.scalar;
}

mat learningModel::operator*(const IsoCovariance &x, const mat &y) {
    return y * x.scalar;
}

vec learningModel::operator*(const IsoCovariance &x, const vec &y){
    return y * x.scalar;
}

rowvec learningModel::operator*(const rowvec &y, const IsoCovariance &x){
    return y * x.scalar;
}

IsoCovariance IsoCovariance::inv(){
    return IsoCovariance(1.0 / scalar, size);
}

double IsoCovariance::log_det() {
    mat full(size, size, fill::zeros);
    full.diag() += scalar;
    return Helpers::computeDeterminant(full);
}

void IsoCovariance::rankOneUpdate(const arma::vec & v, double alpha) {
    scalar += accu(pow(v, 2) * alpha) / size;
}

void IsoCovariance::print() {
    std::cout << "IsoCovariance : " << scalar << " size : " << size << std::endl;
}

mat IsoCovariance::getFull() const {
    mat full(size, size, fill::zeros);
    full.diag() += scalar;
    return full;
}

mat learningModel::operator-(const mat &y, const IsoCovariance &x) {
    mat result = y;
    result.diag() -= x.scalar;
    return result;
}

mat learningModel::operator-(const IsoCovariance &x, const mat &y) {
    mat result = x.getFull();
    result -= y;
    return result;
}












