/**
 * @file FourParamsModel.cpp
 * @brief Class implementation of the 4 parameters Hapke model adapter
 * @author Sami DJOUADI
 * @version 1.0
 * @date 14/12/2019
 */

#include "FourParamsModel.h"
//#include "../../Enumeration.h"

using namespace Functional;
//using namespace HapkeEnumeration;

FourParamsModel::FourParamsModel(double b0, double h) {
    this->b0 = b0;
    this->h = h;
}

void FourParamsModel::adaptModel(rowvec &photometry) {
    this->c = photometry(C);
}

int FourParamsModel::get_dimension_L() {
    return 4;
}
