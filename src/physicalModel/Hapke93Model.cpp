//
// Created by reverse-proxy on 25‏/12‏/2019.
//

#include "Hapke93Model.h"

using namespace HapkeEnumeration;

Hapke93Model::Hapke93Model(std::vector<std::vector<double>> &geometries): HapkeModel(geometries){}

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
    result = (1 + calculate_B(x(B0),x(H))) % calculate_P(x(B), x(C)) + calculate_H(mu0e, x(OMEGA)) % calculate_H(mue , x(OMEGA)) - 1;
    return result;
}


