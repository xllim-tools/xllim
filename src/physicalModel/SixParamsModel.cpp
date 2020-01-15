//
// Created by reverse-proxy on 14‏/1‏/2020.
//

#include "SixParamsModel.h"
#include "Enumeration.h"

using namespace HapkeEnumeration;

void SixParamsModel::adaptModel(rowvec &x) {
    this->b0 = x(B0);
    this->h = x(H);
    this->c = x(C);
}

int SixParamsModel::get_dimension_L() {
    return 6;
}

SixParamsModel::SixParamsModel() = default;
