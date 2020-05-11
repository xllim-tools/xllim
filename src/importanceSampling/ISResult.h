/**
 * @file ISResult.h
 * @author Sami DJOUADI
 * @version 1.2
 * @date 22/04/2020
 */

#ifndef KERNELO_ISRESULT_H
#define KERNELO_ISRESULT_H

#include "ISDiagnostic.h"
#include <armadillo>

using namespace arma;

namespace importanceSampling{
    /**
     * @struct ISResult
     * @brief This struct wraps the results of the importance sampling algorithm
     */
    struct ISResult{
        ISDiagnostic diagnostic{}; /**< @see ISDiagnostic ISDiagnostic*/
        vec covariance; /**< The variance of the prediction*/
        vec mean; /** The prediction */
    };
}


#endif //KERNELO_ISRESULT_H
