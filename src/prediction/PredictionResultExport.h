//
// Created by reverse-proxy on 26‏/4‏/2020.
//

#ifndef KERNELO_PREDICTIONRESULTEXPORT_H
#define KERNELO_PREDICTIONRESULTEXPORT_H

namespace prediction{
    struct MeanPredictionResultExport{
        double *mean;
        double *variance;
        double *gmm_weights;
        double *gmm_means;
        double *gmm_covs;
    };

    struct CenterPredictionResultExport{
        double *weights;
        double *means;
        double *covs;
    };
    struct PredictionResultExport{
        MeanPredictionResultExport meanPred;
        CenterPredictionResultExport centerPred;
    };
}

#endif //KERNELO_PREDICTIONRESULTEXPORT_H
