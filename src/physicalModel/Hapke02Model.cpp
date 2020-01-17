/**
 * @file Hapke02Model.cpp
 * @brief 1993 Hapke model class implmentation
 * @author Sami DJOUADI
 * @version 1.0
 * @date 27/12/2019
 */
#include "Hapke02Model.h"

#include <utility>

using namespace Functional;
using namespace HapkeEnumeration;


Hapke02Model::Hapke02Model(const double *geometries, int row_size, int col_size, const std::shared_ptr<HapkeAdapter>& adapter)
        : HapkeModel(geometries, row_size,col_size, adapter) {}


double Hapke02Model::set_coef() {
    return 1;
}

rowvec Hapke02Model::define_different_part(const rowvec &photometry, rowvec mue, rowvec mu0e) {
    rowvec result = rowvec(configuredGeometries.n_rows);
    result = (1 + calculate_B(adapter->get_b0(),adapter->get_h())) % calculate_P(photometry(B), adapter->get_c()) + (calculate_H(mu0e, photometry(OMEGA)) % calculate_H(mue , photometry(OMEGA))) - 1;
    //cout<< "diff part " << result(0) <<endl;
    return result;
}

rowvec Hapke02Model::calculate_H(const rowvec &x, double omega) {
    double y = sqrt(1 - omega);
    rowvec result = (1 + 2 * x)/(1 + 2 * x * y);
    //cout<< "H " << result(0) <<endl;
    return result;
}
