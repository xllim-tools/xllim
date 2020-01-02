//
// Created by reverse-proxy on 27‏/12‏/2019.
//

#ifndef UNTITLED_HAPKE02MODEL_H
#define UNTITLED_HAPKE02MODEL_H

#include "HapkeModel.h"

class Hapke02Model : public HapkeModel {
public:
    Hapke02Model(std::vector<std::vector<double>> &geometries);

private:
    double set_coef() override ;
    rowvec define_different_part(const rowvec &x, rowvec mue, rowvec mu0e) override ;
    static rowvec calculate_H(const rowvec &x , double omega);
};



#endif //UNTITLED_HAPKE02MODEL_H
