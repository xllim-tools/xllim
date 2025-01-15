// /usr/bin/g++ -fdiagnostics-color=always -g test_new_gllim.cpp ../../src/xllimSolver/gllim.cpp ../../src/xllimSolver/covariances/* ../../src/utils/utils.cpp -o test_new_gllim.out -lpython3.10 -larmadillo -llapack -lblas -O0

#include "../../src/xllimSolver/gllim.hpp"

int main()
{
    GLLiM<FullCovariance, DiagCovariance> gllim(5, 9, 4, "full", "diag");
    GLLiMParameters<FullCovariance, DiagCovariance> theta = gllim.getParams();
    theta.Sigma[0].print("Sigma");

    return 0;
}