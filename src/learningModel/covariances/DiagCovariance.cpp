//
// Created by reverse-proxy on 17‏/2‏/2020.
//

#include "Icovariance.h"

using namespace learningModel;

DiagCovariance::DiagCovariance(const vec &covariance){
    this->covariance = covariance;
}

DiagCovariance::DiagCovariance(const mat &covariance){
    this->covariance = covariance.diag();
}

DiagCovariance::DiagCovariance(unsigned dimension) {
    this->covariance = vec(dimension, fill::zeros);
}

DiagCovariance &DiagCovariance::operator=(const DiagCovariance &cov) {
    covariance = cov.covariance;
}

DiagCovariance &DiagCovariance::operator=(const mat &cov){
    covariance = cov.diag();
}

DiagCovariance &DiagCovariance::operator=(double scalar) {
    covariance.fill(scalar);
}

mat learningModel::operator+(const mat &y, const DiagCovariance &x) {
    mat result = y;
    result.diag() += x.covariance;
    return result;
}

mat learningModel::operator+(const DiagCovariance &x, const mat &y) {
    return y + x;
}

mat learningModel::operator*(const mat &y, const DiagCovariance &x) {
    mat result = mat(y.n_rows,y.n_cols);
    for(unsigned i=0; i<x.covariance.n_rows; i++){
        result.col(i) = y.col(i) * x.covariance.row(i);
    }
    return result;
}

mat learningModel::operator*(const DiagCovariance &x, const mat &y) {
    mat result = mat(y.n_rows,y.n_cols);
    for(unsigned i=0; i<y.n_cols; i++){
        result.col(i) = y.col(i) % x.covariance;
    }
    return result;
}

DiagCovariance DiagCovariance::inv() {
    vec inv = 1.0/covariance;
    return DiagCovariance(inv);
}

double DiagCovariance::det() {
    double det = prod(covariance);
    if(det < 0)
        return 0;
    return det;
}

DiagCovariance &DiagCovariance::operator+=(double scalar) {
    covariance += scalar;
}

DiagCovariance &DiagCovariance::operator+=(const mat &cov) {
    covariance += cov.diag();
}

void DiagCovariance::rankOneUpdate(const vec &v, double alpha) {
    covariance += pow(v,2) * alpha;
}

void DiagCovariance::print() {
    covariance.t().print();
}

vec learningModel::operator*(const DiagCovariance &x, const vec &y) {
    return x.covariance % y;
}

rowvec learningModel::operator*(const rowvec &y, const DiagCovariance &x) {
    return y % x.covariance.t();
}







