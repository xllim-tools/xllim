// #include <pybind11/pybind11.h>

#include "TestModel.hpp"

#define TestModel_L_dimension 4
#define TestModel_D_dimension 9

TestModel::TestModel()
{
    A_ = mat(TestModel_D_dimension, TestModel_L_dimension, fill::zeros); //! Seg fault on this line when calling ctest (checked with address sanitizer)
    A_ = {{1, 2, 2, 1},
               {0, 0.5, 0, 0},
               {0, 0, 1, 0},
               {0, 0, 0, 3},
               {0.2, 0, 0, 0},
               {0, -0.5, 0, 0},
               {-0.2, 0, -1, 0},
               {-1, 0, 2, 0},
               {0, 0, 0, -0.7}};
    A_ *= 0.5;
}

void TestModel::F(vec x, vec &y)
{
    vec Hx(TestModel_L_dimension, fill::ones);
    vec Gx(TestModel_L_dimension, fill::ones);

    // Fill Hx
    Hx(0) = x(0);
    Hx(1) = x(1);
    Hx(2) = 4. * pow(x(2) - 0.5, 2);
    Hx(3) = x(3);

    // Fill Gx
    Gx(0) = exp(Hx(0));
    Gx(1) = exp(Hx(1));
    Gx(2) = exp(Hx(2));
    Gx(3) = exp(Hx(3));

    // compute y
    y = A_ * Gx;
}

unsigned TestModel::getDimensionY()
{
    return TestModel_D_dimension;
}

unsigned TestModel::getDimensionX()
{
    return TestModel_L_dimension;
}

void TestModel::toPhysic(vec &x)
{
    x *= 2;
}

void TestModel::fromPhysic(vec &x)
{
    x /= 2;
}
