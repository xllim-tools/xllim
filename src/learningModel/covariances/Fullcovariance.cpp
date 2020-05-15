/**
 * @file FullCovariance.cpp
 * @brief FullCovairance class implementation
 * @author Sami DJOUADI
 * @version 1.1
 * @date 17/02/2020
 */

#include "Icovariance.h"

using namespace learningModel;


FullCovariance::FullCovariance(const mat &covariance){
    this->covariance = covariance;
}

FullCovariance &FullCovariance::operator=(const FullCovariance &cov) {
    covariance = cov.covariance;
    return *this;
}

FullCovariance &FullCovariance::operator=(const mat &cov){
    covariance = cov;
    return *this;
}

FullCovariance &FullCovariance::operator=(double scalar) {
    covariance.fill(scalar);
    return *this;
}

double FullCovariance::det() {
    double result = Helpers::computeDeterminant(covariance);
    if(result < 0)
        result = 0;
    return result;
}

FullCovariance FullCovariance::inv() {
    mat inv = Helpers::inverseMatrix(covariance);
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

mat FullCovariance::getFull() const{
    return covariance;
}

mat learningModel::operator-(const mat &y, const FullCovariance &x) {
    return y - x.covariance;
}

mat learningModel::operator-(const FullCovariance &x, const mat &y) {
    return x.covariance - y;
}






