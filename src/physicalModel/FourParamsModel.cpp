//
// Created by reverse-proxy on 14‏/1‏/2020.
//

#include "FourParamsModel.h"
#include "Enumeration.h"

using namespace HapkeEnumeration;

FourParamsModel::FourParamsModel(double b0, double h) {
    this->b0 = b0;
    this->h = h;
}

void FourParamsModel::adaptModel(rowvec &x) {
    this->c = x(C);
}

int FourParamsModel::get_dimension_L() {
    return 4;
}
