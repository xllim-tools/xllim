//
// Created by reverse-proxy on 14‏/1‏/2020.
//

#ifndef KERNELO_SIXPARAMSMODEL_H
#define KERNELO_SIXPARAMSMODEL_H

#include "HapkeAdapter.h"

class SixParamsModel: public HapkeAdapter {
public:
    void adaptModel(rowvec &x) override ;
    int get_dimension_L() override ;
    SixParamsModel();
};


#endif //KERNELO_SIXPARAMSMODEL_H
