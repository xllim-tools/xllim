//
// Created by reverse-proxy on 19‏/4‏/2020.
//

#ifndef KERNELO_PREDICTIONRESULT_H
#define KERNELO_PREDICTIONRESULT_H

#include <armadillo>

using namespace arma;

namespace prediction{
    struct MeanPredictionResult{
        vec mean;
        vec variance;
        vec gmm_weights;
        mat gmm_means;
        cube gmm_covs;
    };

    struct CenterPredictionResult{
        vec weights;
        mat means;
        cube covs;

    };
    struct PredictionResult{
        MeanPredictionResult meanPredResult;
        CenterPredictionResult centerPredResult;
    };
}

#endif //KERNELO_PREDICTIONRESULT_H
