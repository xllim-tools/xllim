//
// Created by reverse-proxy on 17‏/2‏/2020.
//

#include "Icovariance.h"

using namespace learningModel;


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
    covariance.fill(scalar);
}

double FullCovariance::det() {
    double result = arma::det(covariance);
    if(result < 0)
        result = 0;
    return result;
}

FullCovariance FullCovariance::inv() {
    mat inv = arma::inv(covariance);
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

void FullCovariance::rankOneUpdate(const vec &v, double alpha) {
    for(unsigned c=0; c < v.n_rows; c++){
        covariance.col(c) += v * v(c) * alpha;
    }
}

FullCovariance::FullCovariance(unsigned dimension) {
    covariance = mat(dimension,dimension,fill::zeros);
}







