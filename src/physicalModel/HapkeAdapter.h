//
// Created by reverse-proxy on 14‏/1‏/2020.
//

#ifndef KERNELO_HAPKEADAPTER_H
#define KERNELO_HAPKEADAPTER_H

#include <armadillo>

using namespace arma;

class HapkeAdapter{
public:
    virtual void adaptModel(rowvec &x) = 0;
    virtual int get_dimension_L() = 0;

    double get_b0(){
        return b0;
    };

    double get_h() {
        return h;
    };

    double get_c() {
        return c;
    };


protected:
    double c;
    double b0;
    double h;
};

#endif //KERNELO_HAPKEADAPTER_H


