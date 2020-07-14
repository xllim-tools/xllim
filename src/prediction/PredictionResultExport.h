/**
 * @file PredictionResultExport.h
 * @brief External structures of the results of the prediction
 * @author Sami DJOUADI
 * @version 1.2
 * @date 26/04/2020
 */

#ifndef KERNELO_PREDICTIONRESULTEXPORT_H
#define KERNELO_PREDICTIONRESULTEXPORT_H

namespace prediction{
    /**
     * @struct MeanPredictionResultExport
     */
    class MeanPredictionResultExport{
    public:
        double *mean;/**< The mean of the GMM which stands for the prediction*/
        double *variance; /**< The variance of the prediction*/
        double *gmm_weights; /**< The weights of the components of the GMM*/
        double *gmm_means; /**< The means of each component in the GMM*/
        double *gmm_covs; /**< The covariance matrices of each component in the GMM*/
    };

    /**
     * @struct CenterPredictionResultExport
     */
    class CenterPredictionResultExport{
    public:
        double *weights; /**< The weights of the centers*/
        double *means; /**< The centers that stands for the predictions*/
        double *covs; /**< The covariance matrices of the centers*/
    };

    /**
     * @strcut PredictionResultExport
     */
    class PredictionResultExport{
    public:
        std::shared_ptr<MeanPredictionResultExport> meanPred; /**< @see MeanPredictionResultExport MeanPredictionResultExport*/
        std::shared_ptr<CenterPredictionResultExport> centerPred; /**< @see CenterPredictionResultExport CenterPredictionResultExport*/
    };
}

#endif //KERNELO_PREDICTIONRESULTEXPORT_H
