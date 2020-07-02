/**
 * @file IPredictor.h
 * @brief IPredictor class definition
 * @author Sami DJOUADI
 * @version 1.2
 * @date 26/03/2019
 */

#ifndef KERNELO_IPREDICTOR_H
#define KERNELO_IPREDICTOR_H

#include "PredictionResult.h"
#include "PredictionResultExport.h"

namespace prediction{

    /**
     * @class IPredictor
     *
     * @details This is the interface of the prediction module. It offers prediction and regularization methods in two format.
     * One using pointer to represent data structures like vectors, matrices and cube. This format is used for integration purposes
     * with third party language API. The second format uses data structure of the library Armadillo , and it is used for internal calls
     * within the modules of the library.
     */
    class IPredictor{
    public:

        /**
         * @details This method can be used by third party language API to compute a prediction of the given observation and using
         * a trained GLLiM model. It calls the internal prediction method based on data structures of the library Armadillo.
         * @param y_obs : a vector of low dimension data
         * @param var_obs : the errors in the measure of the observation
         * @param size : number of variables of the observation
         * @param resultExport : @see PredictionResultExport PredictionResultExport
         */
        void predict(double *y_obs, double *var_obs, unsigned size, const std::shared_ptr<PredictionResultExport>& resultExport){
            vec y_obs_arma(&y_obs[0], size,  false, true);
            vec var_obs_arma(&var_obs[0], size, false, true);

            PredictionResult result = predict(y_obs_arma, var_obs_arma);

            unsigned L = result.meanPredResult.mean.n_rows;
            unsigned k_merged = result.centerPredResult.weights.n_rows;
            unsigned k_pred_mean = result.meanPredResult.gmm_weights.n_rows;

            for(unsigned j=0 ; j<L; j++){
                resultExport->meanPred.mean[j] = result.meanPredResult.mean(j);
                resultExport->meanPred.variance[j] = result.meanPredResult.variance(j);
            }

            for(unsigned i=0 ; i<k_pred_mean ; i++){
                resultExport->meanPred.gmm_weights[i] = result.meanPredResult.gmm_weights(i);
                for(unsigned j=0 ; j<L; j++){
                    resultExport->meanPred.gmm_means[i + j*k_pred_mean] = result.meanPredResult.gmm_means(j,i);
                }
            }

            for(unsigned i=0; i<L * L * k_pred_mean; i++){
                resultExport->meanPred.gmm_covs[i] = result.meanPredResult.gmm_covs(
                        (i % (L * L))% (L),
                        (i % (L * L))/ (L),
                        i / (L * L)
                );
            }

            for(unsigned i=0 ; i<k_merged ; i++){
                resultExport->centerPred.weights[i] = result.centerPredResult.weights(i);
                for(unsigned j=0 ; j<L; j++){
                    resultExport->centerPred.means[i + j*k_merged] = result.centerPredResult.means(j,i);
                }
            }

            for(unsigned i=0; i<L * L * k_merged; i++){
                resultExport->centerPred.covs[i] = result.centerPredResult.covs(
                        (i % (L * L))% (L),
                        (i % (L * L))/ (L),
                        i / (L * L)
                );
            }
        }

        /**
         * @details This method can be used by third party language API to get a regularized series pf predictions if the context
         * of the study requires it. The method calls the internal regularization method based on data structures of the library Armadillo.
         * @param series : The centers computed by the prediction algorithm for all the observations.
         * @param rows : The number of variables in a prediction
         * @param cols : The number of centers per prediction.
         * @param slices : the number of tuples to predict
         * @param permutations : A pointer to a matrix(K,N) containing the permutations between the centers of each prediction.
         */
        void regularize(const double *series , unsigned rows, unsigned cols, unsigned slices, double *permutations){
            cube series_arma(rows, cols, slices);

            for(unsigned i=0; i<rows * cols * slices; i++){
                series_arma(
                        (i % (rows * cols))% (rows),
                        (i % (rows * cols))/ (rows),
                        i / (rows * cols)
                ) = series[i];
            }

            Mat<unsigned> permutations_arma = regularize(series_arma);

            for(unsigned i=0 ; i<cols ; i++){
                for(unsigned j=0 ; j<slices; j++){
                    permutations[i*slices+j] = permutations_arma(i,j);
                }
            }
        }

        /**
         * @details This methods computes the predictions that correspond to the given observation using data structures of the library
         * Armadillo. Should be used for internal calls within the library
         * @param y_obs : a vector of low dimension data
         * @param cov_obs : the errors in the measure of the observation
         * @return @see PredictionResult PredictionResult
         */
        virtual PredictionResult predict(const vec &y_obs, const vec &cov_obs) = 0;

        /**
         * @details This method returns permutations of the predictions of all the observations given in parameter. These permutations
         * make the predictions more adapter to a context where regularity is required.
         * @param series : A cube (L,K,N) of centers computed by the prediction algorithm for all the observations.
         * @return arma::Mat<unsigned>
         */
        virtual Mat<unsigned> regularize(const cube &series) = 0;
    };

}

#endif //KERNELO_IPREDICTOR_H
