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

