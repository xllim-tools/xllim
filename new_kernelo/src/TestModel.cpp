// #include <pybind11/pybind11.h>

#include "TestModel.hpp"

#define TestModel_L_dimension 4
#define TestModel_D_dimension 9

using namespace Functional;

TestModel::TestModel() {
    this->A = mat(TestModel_D_dimension, TestModel_L_dimension, fill::zeros);
    this->A = { {1, 2, 2, 1},
                {0, 0.5, 0, 0},
                {0, 0, 1, 0},
                {0, 0, 0, 3},
                {0.2, 0, 0, 0},
                {0, -0.5, 0, 0},
                {-0.2, 0, -1, 0},
                {-1, 0, 2, 0},
                {0, 0, 0, -0.7} };
    this->A *= 0.5;
}

void TestModel::F(rowvec x, rowvec &y) {
    rowvec Hx(TestModel_L_dimension, fill::ones);
    rowvec Gx(TestModel_L_dimension, fill::ones);

    // Fill Hx
    Hx(0) = x(0);
    Hx(1) = x(1);
    Hx(2) = 4. * pow(x(2) - 0.5, 2) ;
    Hx(3) = x(3) ;

    // Fill Gx
    Gx(0) = exp(Hx(0));
    Gx(1) = exp(Hx(1));
    Gx(2) = exp(Hx(2));
    Gx(3) = exp(Hx(3));

    // compute y
    mat C = this->A * Gx.t();
    C = C.t();
    y = C.row(0);
}

int TestModel::get_D_dimension() {
    return TestModel_D_dimension;
}

int TestModel::get_L_dimension() {
    return TestModel_L_dimension;
}

void TestModel::to_physic(rowvec &x) {

}

void TestModel::to_physic(double *x, unsigned int size) {

}

void TestModel::from_physic(double *x, unsigned int size) {

}