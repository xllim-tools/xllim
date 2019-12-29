//
// Created by reverse-proxy on 25‏/12‏/2019.
//

#ifndef UNTITLED_FUNCTIONNALMODEL_H
#define UNTITLED_FUNCTIONNALMODEL_H

#include <armadillo>
#include <utility>
#include <memory>

using namespace arma;

class FunctionnalModel{
public:
    virtual void F(const rowvec &x, rowvec y) = 0;
    virtual rowvec F(const rowvec &x) = 0;
    virtual mat F(const mat &x) = 0;
};

#endif //UNTITLED_FUNCTIONNALMODEL_H
