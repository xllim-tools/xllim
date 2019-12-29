//
// Created by reverse-proxy on 27‏/12‏/2019.
//

#include "Hapke02Model.h"

#include <utility>

using namespace HapkeEnumeration;

Hapke02Model::Hapke02Model(mat geometries) : HapkeModel(std::move(geometries)) {}


double Hapke02Model::set_coef() {
    return 1;
}

rowvec Hapke02Model::define_different_part(const rowvec &x, rowvec mue, rowvec mu0e) {
    rowvec result = rowvec(configuredGeometries.n_rows);
    result = (1 + calculate_B(x(B0),x(H))) % calculate_P(x(B), x(C)) + (calculate_H(mu0e, x(OMEGA)) % calculate_H(mue , x(OMEGA))) - 1;
    return result;
}

rowvec Hapke02Model::calculate_H(const rowvec &x, double omega) {
    double y = sqrt(1 - omega);
    rowvec result = (1 + 2 * x)/(1 + 2 * x * y);
    return result;
}
