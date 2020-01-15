//
// Created by reverse-proxy on 14‏/1‏/2020.
//

#ifndef KERNELO_FOURPARAMSMODEL_H
#define KERNELO_FOURPARAMSMODEL_H

#include "HapkeAdapter.h"

class FourParamsModel : public HapkeAdapter{
public:
    FourParamsModel(double b0, double h);
    void adaptModel(rowvec &x) override ;
    int get_dimension_L() override ;
};


#endif //KERNELO_FOURPARAMSMODEL_H
