//
// Created by reverse-proxy on 25‏/12‏/2019.
//

#include "Hapke93Model.h"

using namespace HapkeEnumeration;

Hapke93Model::Hapke93Model(const double *geometries, int row_size, int col_size, const std::shared_ptr<HapkeAdapter>& adapter)
        : HapkeModel(geometries, row_size, col_size, adapter) {}

rowvec Hapke93Model::calculate_H(const rowvec &x , double omega) {
    double y = sqrt(1 - omega);
    double temp = (1-y)/(1+y);
    rowvec result = 1 / (1 - (1 - y) * x % (temp +(1 - (0.5 + x) * temp) % log((1+x)/x)));
    return result;
}

double Hapke93Model::set_coef() {
    return 1;
}

rowvec Hapke93Model::define_different_part(const rowvec &x, rowvec mue, rowvec mu0e) {
    rowvec result = rowvec(configuredGeometries.n_rows);
    result = (1 + calculate_B(adapter->get_b0(),adapter->get_h())) % calculate_P(x(B), adapter->get_c()) + calculate_H(mu0e, x(OMEGA)) % calculate_H(mue , x(OMEGA)) - 1;
    return result;
}


