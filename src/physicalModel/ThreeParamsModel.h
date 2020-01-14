//
// Created by reverse-proxy on 14‏/1‏/2020.
//

#ifndef KERNELO_THREEPARAMSMODEL_H
#define KERNELO_THREEPARAMSMODEL_H

#include "HapkeAdapter.h"

class ThreeParamsModel: public HapkeAdapter {
public:
    ThreeParamsModel(double b0, double h);
    void adaptModel(rowvec &x) override ;
    int get_dimension_L() override ;
};


#endif //KERNELO_THREEPARAMSMODEL_H
