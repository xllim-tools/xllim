/**
 * @file ThreeParamsModel.cpp
 * @brief Class implementation of the 3 parameters Hapke model adapter
 * @author Sami DJOUADI
 * @version 1.0
 * @date 14/12/2019
 */

#include "ThreeParamsModel.h"

using namespace Functional;
using namespace HapkeEnumeration;

void ThreeParamsModel::adaptModel(rowvec &photometry) {
    this->c = (3.29 * exp(-17.4 * pow(photometry(B), 2)) + 0.092) / 2;
}

ThreeParamsModel::ThreeParamsModel(double b0, double h): HapkeAdapter(){
    this->b0 = b0;
    this->h = h;
}

int ThreeParamsModel::get_dimension_L() {
    return 3;
}
