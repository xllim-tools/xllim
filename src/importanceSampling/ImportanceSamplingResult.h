/**
 * @file ImportanceSamplingResult.h
 * @author Sami DJOUADI
 * @version 1.2
 * @date 26/04/2020
 */

#ifndef KERNELO_IMPORTANCESAMPLINGRESULT_H
#define KERNELO_IMPORTANCESAMPLINGRESULT_H

#include "ImportanceSamplingDiagnostic.h"

namespace importanceSampling{
    /**
     * @struct ImportanceSamplingResult
     * @brief This struct wraps the results of the importance sampling algorithm. The struct is meant for integration purposes
     * with a third language API.
     */
    class ImportanceSamplingResult{
    public:
        ImportanceSamplingDiagnostic diagnostic;
        double *covariance;
        double *mean;
    };
}

#endif //KERNELO_IMPORTANCESAMPLINGRESULT_H
