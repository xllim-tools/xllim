//
// Created by reverse-proxy on 25‏/12‏/2019.
//

#ifndef UNTITLED_HAPKE93MODEL_H
#define UNTITLED_HAPKE93MODEL_H

#include "HapkeModel.h"
#include "Enumeration.h"

class Hapke93Model : public HapkeModel {
public:
    Hapke93Model();

private:
    double set_coef() override ;
    rowvec define_different_part(const rowvec &x, rowvec mue, rowvec mu0e) override ;
    static rowvec calculate_H(const rowvec &x , double omega);
};


#endif //UNTITLED_HAPKE93MODEL_H
