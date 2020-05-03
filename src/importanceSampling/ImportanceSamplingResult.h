//
// Created by reverse-proxy on 27‏/4‏/2020.
//

#ifndef KERNELO_IMPORTANCESAMPLINGRESULT_H
#define KERNELO_IMPORTANCESAMPLINGRESULT_H

#include "ImportanceSamplingDiagnostic.h"

namespace importanceSampling{
    struct ImportanceSamplingResult{
        ImportanceSamplingDiagnostic diagnostic;
        double *covariance;
        double *mean;
    };
}

#endif //KERNELO_IMPORTANCESAMPLINGRESULT_H
