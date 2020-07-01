/**
 * @file DiagCovariance.cpp
 * @brief DiagCovairance class implementation
 * @author Sami DJOUADI
 * @version 1.1
 * @date 17/02/2020
 */

#include "Icovariance.h"

using namespace learningModel;

DiagCovariance::DiagCovariance(const vec &covariance){
    this->variances = covariance;
}

DiagCovariance::DiagCovariance(const mat &covariance){
    this->variances = covariance.diag();
}

DiagCovariance::DiagCovariance(unsigned dimension) {
    this->variances = vec(dimension, fill::zeros);
}

DiagCovariance &DiagCovariance::operator=(const DiagCovariance &cov) {
    variances = cov.variances;
    return *this;
}

DiagCovariance &DiagCovariance::operator=(const mat &cov){
    variances = cov.diag();
    return *this;
}

DiagCovariance &DiagCovariance::operator=(double scalar) {
    variances.fill(scalar);
    return *this;
}

mat learningModel::operator+(const mat &y, const DiagCovariance &x) {
    mat result = y;
    result.diag() += x.variances;
    return result;
}

mat learningModel::operator+(const DiagCovariance &x, const mat &y) {
    return y + x;
}

mat learningModel::operator*(const mat &y, const DiagCovariance &x) {
    mat result = mat(y.n_rows,y.n_cols);
    for(unsigned i=0; i<x.variances.n_rows; i++){
        result.col(i) = y.col(i) * x.variances.row(i);
    }
    return result;
}

mat learningModel::operator*(const DiagCovariance &x, const mat &y) {
    mat result = mat(y.n_rows,y.n_cols);
    for(unsigned i=0; i<y.n_cols; i++){
        result.col(i) = y.col(i) % x.variances;
    }
    return result;
}

DiagCovariance DiagCovariance::inv() {
    vec inv = 1.0 / variances;
    return DiagCovariance(inv);
}

double DiagCovariance::log_det() {
    mat full(variances.n_rows, variances.n_rows, fill::zeros);
    full.diag() += variances;
    return Helpers::computeDeterminant(full);
}

DiagCovariance &DiagCovariance::operator+=(double scalar) {
    variances += scalar;
}

DiagCovariance &DiagCovariance::operator+=(const mat &cov) {
    variances += cov.diag();
}

void DiagCovariance::rankOneUpdate(const vec &v, double alpha) {
    variances += pow(v, 2) * alpha;
}

void DiagCovariance::print() {
    variances.t().print();
}

vec learningModel::operator*(const DiagCovariance &x, const vec &y) {
    return x.variances % y;
}

rowvec learningModel::operator*(const rowvec &y, const DiagCovariance &x) {
    return y % x.variances.t();
}

mat DiagCovariance::getFull() const {
    mat full(variances.n_rows, variances.n_rows, fill::zeros);
    full.diag() += variances;
    return full;
}

mat learningModel::operator-(const mat &y, const DiagCovariance &x) {
    mat result = y;
    result.diag() -= x.variances;
    return result;
}

mat learningModel::operator-(const DiagCovariance &x, const mat &y) {
    mat result = x.getFull();
    result -= y;
    return result;
}








