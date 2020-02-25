//
// Created by reverse-proxy on 17‏/2‏/2020.
//

#include "Icovariance.h"

using namespace learningModel;
using namespace arma;


FullCovariance::FullCovariance(const mat &covariance){
    this->covariance = covariance;
}

FullCovariance &FullCovariance::operator=(const FullCovariance &cov) {
    covariance = cov.covariance;
}

FullCovariance &FullCovariance::operator=(const mat &cov){
    covariance = cov;
}

FullCovariance &FullCovariance::operator=(double scalar) {
    for(unsigned i=0; i<covariance.n_rows; i++)
        for(unsigned j=0; j<covariance.n_rows; j++)
            covariance(i,j) = scalar;
}

double FullCovariance::det() {
    double result = arma::det(covariance);
    if(result < 0)
        result = 0;
    return result;
}

FullCovariance FullCovariance::inv(bool print) {
    mat inv = arma::inv(covariance);
    if(print)
        inv.print();
    return FullCovariance(inv);
}

mat learningModel::operator+(const mat &y, const FullCovariance &x) {
    return y + x.covariance;
}

mat learningModel::operator+(const FullCovariance &x, const mat &y) {
    return y + x.covariance;
}

mat learningModel::operator*(const mat &y, const FullCovariance &x) {
    return y * x.covariance;
}

mat learningModel::operator*(const FullCovariance &x, const mat &y) {
    return x.covariance * y;
}

FullCovariance &FullCovariance::operator+=(const mat &cov) {
    covariance += cov;
}

FullCovariance &FullCovariance::operator+=(double scalar) {
    covariance += scalar;
}

vec learningModel::operator*(const FullCovariance &x, const vec &y) {
    return x.covariance * y;
}

rowvec learningModel::operator*(const rowvec &y, const FullCovariance &x) {
    return y * x.covariance;
}

void FullCovariance::print() {
    covariance.print("covariance");
}







