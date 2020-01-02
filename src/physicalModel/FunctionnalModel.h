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
    virtual void F(const std::vector<double> &x, std::vector<double> &y) = 0;
    virtual std::vector<double> F(const std::vector<double> &x) = 0;
    virtual std::vector<std::vector<double>> F(const std::vector<std::vector<double>> &x) = 0;
    virtual int get_D_dimension() = 0;
    virtual int get_L_dimension() = 0;
    virtual std::vector<double> nomalize(std::vector<double> x) = 0;
    virtual std::vector<double> invNormalize(std::vector<double> x) = 0;
};

#endif //UNTITLED_FUNCTIONNALMODEL_H
