//
// Created by reverse-proxy on 2‏/4‏/2020.
//

#ifndef KERNELO_ISRESULT_H
#define KERNELO_ISRESULT_H

#include "ISDiagnostic.h"
#include <armadillo>

using namespace arma;

namespace importanceSampling{
    struct ISResult{
        ISDiagnostic diagnostic{};
        vec covariance;
        vec mean;
    };
}


#endif //KERNELO_ISRESULT_H
