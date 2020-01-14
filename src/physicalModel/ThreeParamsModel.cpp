//
// Created by reverse-proxy on 14‏/1‏/2020.
//

#include "ThreeParamsModel.h"
#include "Enumeration.h"

using namespace HapkeEnumeration;

void ThreeParamsModel::adaptModel(rowvec &x) {
    this->c = 3.29 * exp(-17.4 * pow(x(B),2) + 0.092)/2;
}

ThreeParamsModel::ThreeParamsModel(double b0, double h): HapkeAdapter(){
    this->b0 = b0;
    this->h = h;
}

int ThreeParamsModel::get_dimension_L() {
    return 3;
}
