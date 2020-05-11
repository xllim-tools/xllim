/**
 * @file SixParamsModel.cpp
 * @brief Class implementation of the 6 parameters Hapke model adapter
 * @author Sami DJOUADI
 * @version 1.0
 * @date 14/12/2019
 */

#include "SixParamsModel.h"
#include "../../Enumeration.h"

using namespace Functional;
using namespace HapkeEnumeration;

void SixParamsModel::adaptModel(rowvec &photometry) {
    this->b0 = photometry(B0);
    this->h = photometry(H);
    this->c = photometry(C);
}

int SixParamsModel::get_dimension_L() {
    return 6;
}

SixParamsModel::SixParamsModel() = default;
