//
// Created by reverse-proxy on 26‏/2‏/2020.
//

#include "Icovariance.h"

using namespace learningModel;

IsoCovariance::IsoCovariance(double covariance, unsigned size){
    this->covariance = covariance;
    this->size = size;
}

IsoCovariance::IsoCovariance(const mat &covariance){
    this->covariance = accu(covariance.diag())/covariance.n_cols;
    this->size = covariance.n_cols;
}

IsoCovariance::IsoCovariance() {
    covariance = 0;
    size = 1;
}

IsoCovariance &IsoCovariance::operator=(const IsoCovariance &cov) {
    covariance = cov.covariance;
    size = cov.size;
}

IsoCovariance &IsoCovariance::operator=(const arma::mat & cov) {
    this->covariance = accu(cov.diag())/cov.n_cols;
    this->size = cov.n_cols;
}

IsoCovariance &IsoCovariance::operator=(double scalar) {
    covariance = scalar;
}

IsoCovariance &IsoCovariance::operator+=(double scalar) {
    covariance += scalar;
}

IsoCovariance &IsoCovariance::operator+=(const arma::mat & cov) {
    covariance += accu(cov.diag())/cov.n_cols;
}

mat learningModel::operator+(const mat &y, const IsoCovariance &x) {
    return y + x.covariance;
}

mat learningModel::operator+(const IsoCovariance &x, const mat &y) {
    return y + x.covariance;
}

mat learningModel::operator*(const mat &y, const IsoCovariance &x) {
    return y * x.covariance;
}

mat learningModel::operator*(const IsoCovariance &x, const mat &y) {
    return y + x.covariance;
}

vec learningModel::operator*(const IsoCovariance &x, const vec &y){
    return y * x.covariance;
}

rowvec learningModel::operator*(const rowvec &y, const IsoCovariance &x){
    return y * x.covariance;
}

IsoCovariance IsoCovariance::inv(){
    return IsoCovariance(1.0/covariance, size);
}

double IsoCovariance::det() {
    return pow(covariance, size);
}

void IsoCovariance::rankOneUpdate(const arma::vec & v, double alpha) {
    covariance += accu(pow(v,2) * alpha)/size;
}

void IsoCovariance::print() {
    std::cout << "IsoCovariance : " << covariance << " size : " << size << std::endl;
}












