//
// Created by reverse-proxy on 16‏/3‏/2020.
//

#include "Helpers.h"

using namespace Helpers;

double Helpers::logSumExp(const arma::vec & elements) {
    double result = 0;
    double max = elements.max();

    if(max == -datum::inf){
        return max;
    }else{
        for(unsigned i=0; i<elements.n_rows; i++){
            result += exp(elements(i) - max);
        }
        result = log(result) + max;
        return result;
    }
}


double Helpers::computeDeterminant(const mat& matrix){
    if(matrix.n_rows <= 3){
        return det(matrix);
    }else{
        mat R;
        if(chol(R,matrix)){
            return 2 * sum(log(R.diag()));
        }else{
            return det(matrix);
        }
    }

}

mat Helpers::inverseMatrix(const mat& matrix){
    if(matrix.n_rows <= 3){
        return inv(matrix);
    }else {
        mat R;
        if(chol(R,matrix)){
            mat R_inv = inv(chol(matrix));
            return R_inv * R_inv.t();
        }else{
            return inv(matrix);
        }
    }
}
