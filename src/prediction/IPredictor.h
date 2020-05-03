//
// Created by reverse-proxy on 26‏/3‏/2020.
//

#ifndef KERNELO_IPREDICTOR_H
#define KERNELO_IPREDICTOR_H

#include "PredictionResult.h"
#include "PredictionResultExport.h"

namespace prediction{

    class IPredictor{
    public:
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
                    resultExport->meanPred.gmm_means[i*L + j] = result.meanPredResult.gmm_means(j,i);
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
                    resultExport->centerPred.means[i* L + j] = result.centerPredResult.means(j,i);
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
        virtual PredictionResult predict(const vec &y_obs, const vec &cov_obs) = 0;
        virtual Mat<unsigned> regularize(const cube &series) = 0;

    };

}

#endif //KERNELO_IPREDICTOR_H
