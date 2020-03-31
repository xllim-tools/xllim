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
