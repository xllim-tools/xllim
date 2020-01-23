/**
 * @file Hapke93Model.cpp
 * @brief 1993 Hapke model class implmentation
 * @author Sami DJOUADI
 * @version 1.0
 * @date 25/12/2019
 */

#include "Hapke93Model.h"
using namespace Functional;
using namespace HapkeEnumeration;

Hapke93Model::Hapke93Model(const double *geometries, int row_size, int col_size,
                           const std::shared_ptr<HapkeAdapter> &adapter,
                           double theta_bar_scaling)
        : HapkeModel(geometries, row_size, col_size, adapter, theta_bar_scaling) {}

rowvec Hapke93Model::calculate_H(const rowvec &x , double omega) {
    double y = sqrt(1 - omega);
    double temp = (1-y)/(1+y);
    rowvec result = 1 / (1 - (1 - y) * x % (temp +(1 - (0.5 + x) * temp) % log((1+x)/x)));
    return result;
}

double Hapke93Model::set_coef() {
    return 1;
}

rowvec Hapke93Model::define_different_part(const rowvec &photometry, rowvec mue, rowvec mu0e) {
    rowvec result = rowvec(configuredGeometries.n_rows);
    result = (1 + calculate_B(adapter->get_b0(),adapter->get_h())) % calculate_P(photometry(B), adapter->get_c()) + calculate_H(mu0e, photometry(OMEGA)) % calculate_H(mue , photometry(OMEGA)) - 1;
    return result;
}


