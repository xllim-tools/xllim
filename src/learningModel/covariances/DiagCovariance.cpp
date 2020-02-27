//
// Created by reverse-proxy on 17‏/2‏/2020.
//

#include "Icovariance.h"

using namespace learningModel;

DiagCovariance::DiagCovariance(const vec &covariance){
    this->covariance = covariance;
}

DiagCovariance &DiagCovariance::operator=(const DiagCovariance &cov) {
    covariance = cov.covariance;
}

DiagCovariance &DiagCovariance::operator=(const mat &cov){
    covariance = cov.diag();
}

mat learningModel::operator+(const mat &y, const DiagCovariance &x) {
    mat result = y;
    for(unsigned i=0; i<x.covariance.n_rows; i++){
        result(i,i) += x.covariance(i);
    }
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
    vec inv = vec(covariance.n_rows);
    for(unsigned i=0; i<covariance.n_rows; i++){
        inv(i) = 1/covariance(i);
    }
    return DiagCovariance(inv);
}

double DiagCovariance::det() {
    return prod(covariance);
}



